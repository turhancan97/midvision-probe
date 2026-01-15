import os
import numpy as np
import torch
import pickle
import scipy
from datasets import load_dataset
import json
from .utils import get_nyu_transforms  # Assuming you have custom transforms in utils.py
from PIL import Image


def NYU(
    train_path,
    test_path,
    split,
    name="nyu",
    image_mean="imagenet",
    center_crop=False,
    rotateflip=False,
    augment_train=False,
):
    assert split in ["train", "trainval", "valid", "test"]
    if split == "test":
        return NYU_test(test_path, image_mean, center_crop)
    else:
        return NYU_geonet(
            train_path,
            split,
            image_mean,
            center_crop,
            augment_train,
            rotateflip=rotateflip,
        )


def make_serializable(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(i) for i in data]
    else:
        return data


class NYU_test(torch.utils.data.Dataset):
    """
    Dataset loader based on Ishan Misra's SSL benchmark
    """

    def __init__(self, path, image_mean="imagenet", center_crop=False):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            False,
            rotateflip=False,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        # parse data
        with open(path, "rb") as f:
            data_dict = pickle.load(f)

        self.indices = data_dict["test_indices"]
        self.depths = [data_dict["depths"][_i] for _i in self.indices]
        self.images = [data_dict["images"][_i] for _i in self.indices]
        self.scenes = [data_dict["scene_types"][_i][0] for _i in self.indices]
        self.snorms = [data_dict["snorms"][_i] for _i in self.indices]
        self.segmentations = [data_dict["instances"][_i] for _i in self.indices]

        num_instances = len(self.indices)
        print(f"NYUv2 labeled test set: {num_instances} instances")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.images[index]
        depth = self.depths[index]
        snorm = self.snorms[index]
        room = self.scenes[index]
        nyu_index = self.indices[index]
        segmentation = self.segmentations[index]
        
        # transform image
        image = np.transpose(image, (1, 2, 0))
        image = self.image_transform(image)

        # set max depth to 10
        depth[depth > 10] = 0

        # center crop
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]
            segmentation = segmentation[..., 80:-80]
        
        # move to tensor
        depth = torch.tensor(depth).float()[None, :, :]
        snorm = torch.tensor(snorm).float()
        segmentation = torch.tensor(segmentation)

        return {
            "image": image,
            "depth": depth,
            "snorm": snorm,
            "room": room,
            "nyu_index": nyu_index,
            "segmentation": segmentation,
        }


class NYU_geonet(torch.utils.data.Dataset):
    """
    Dataset loader for train/validation set using Parquet files with streaming
    """

    def __init__(
        self,
        path,
        split,
        image_mean="imagenet",
        center_crop=False,
        augment_train=False,
        rotateflip=False,
    ):
        super().__init__()
        self.name = "NYUv2"
        self.center_crop = center_crop
        self.max_depth = 10.0

        # get transforms
        image_size = (480, 480) if center_crop else (480, 640)
        augment = augment_train and "train" in split
        self.image_transform, self.shared_transform = get_nyu_transforms(
            image_mean,
            image_size,
            augment,
            rotateflip=rotateflip,
            additional_targets={"depth": "image", "snorm": "image"},
        )

        # parse dataset
        self.root_dir = path
        insts = os.listdir(path)
        insts.sort()

        # remove bad indices
        del insts[21181]
        del insts[6919]

        assert split in ["train", "valid", "trainval"]
        if split == "train":
            self.instances = [x for i, x in enumerate(insts) if i % 20 != 0]
        elif split == "valid":
            self.instances = [x for i, x in enumerate(insts) if i % 20 == 0]
        elif split == "trainval":
            self.instances = insts
        else:
            raise ValueError()

        print(f"NYU-GeoNet {split}: {len(self.instances)} instances.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        file_name = self.instances[index]
        room = "_".join(file_name.split("-")[0].split("_")[:-2])

        # extract elements from the matlab thing
        instance = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        image = instance["img"][:480, :640]
        depth = instance["depth"][:480, :640]
        snorm = torch.tensor(instance["norm"][:480, :640]).permute(2, 0, 1)

        # process image
        image[:, :, 0] = image[:, :, 0] + 2 * 122.175
        image[:, :, 1] = image[:, :, 1] + 2 * 116.169
        image[:, :, 2] = image[:, :, 2] + 2 * 103.508
        image = image.astype(np.uint8)
        image = self.image_transform(image)

        # set max depth to 10
        depth[depth > self.max_depth] = 0

        # center crop
        if self.center_crop:
            image = image[..., 80:-80]
            depth = depth[..., 80:-80]
            snorm = snorm[..., 80:-80]

        if self.shared_transform:
            # put in correct format (h, w, feat)
            image = image.permute(1, 2, 0).numpy()
            snorm = snorm.permute(1, 2, 0).numpy()
            depth = depth[:, :, None]

            # transform
            transformed = self.shared_transform(image=image, depth=depth, snorm=snorm)

            # get back in (feat_dim x height x width)
            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
        else:
            # move to torch tensors
            depth = torch.tensor(depth).float()[None, :, :]
            snorm = torch.tensor(snorm).float()

        return {"image": image, "depth": depth, "snorm": snorm, "room": room}