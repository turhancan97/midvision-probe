from __future__ import annotations

import torch
import torchvision.transforms as T
from torchvision.datasets import Imagenette, FGVCAircraft, Flowers102


class ImagenetteDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around torchvision.datasets.Imagenette to fit this repo's dataset API.

    Returns dicts with keys: "image" (Tensor CxHxW), "label" (int), and "class_id" (int).
    Supports splits: "train", "valid", "test", and "trainval" (mapped to Imagenette's
    'train'/'val').
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        load_size: int = 224,
        image_mean: str = "imagenet",
        download: bool = True,
        name: str = "imagenette",
    ):
        super().__init__()
        self.root = root
        self.name = name

        # Map repo-style splits to Imagenette splits
        if split in ["train", "trainval"]:
            imagenette_split = "train"
        elif split in ["valid", "test"]:
            imagenette_split = "val"
        else:
            raise ValueError(f"Unsupported split for Imagenette: {split}")

        if image_mean == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif image_mean == "clip":
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]

        self.transform = T.Compose(
            [
                T.Resize((load_size, load_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        if name == "imagenette":
            self.dataset = Imagenette(
                root=self.root,
                split=imagenette_split,
                download=download,
                transform=self.transform,
            )
        elif name == "fgvcaircraft":
            self.dataset = FGVCAircraft(
                root=self.root,
                split=imagenette_split,
                download=download,
                transform=self.transform,
            )
        elif name == "flowers102":
            self.dataset = Flowers102(
                root=self.root,
                split=imagenette_split,
                download=download,
                transform=self.transform,
            )
        else:
            raise ValueError(f"Unsupported dataset: {name}")

        # Expose num classes if needed
        try:
            self.num_classes = len(self.dataset.classes)
        except AttributeError:
            self.num_classes = 102

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(label, dtype=torch.long),
        }

