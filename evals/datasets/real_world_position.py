from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


CLASS_FOLDER_TO_LABEL: Dict[str, str] = {
    "front": "Front",
    "back": "Back",
    "left": "Left",
    "right": "Right",
}
LABEL_TO_INDEX: Dict[str, int] = {"Front": 0, "Back": 1, "Left": 2, "Right": 3}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_MODES = {
    "random",
    "time_series",
    "clip_block_random",
    "clip_block_time_series",
}


def _normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split == "val":
        return "valid"
    return split


def _collect_images(class_dir: Path, recursive: bool) -> List[Path]:
    iterator = class_dir.rglob("*") if recursive else class_dir.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.as_posix())
    return files


def _split_counts(n: int, train_ratio: float, valid_ratio: float) -> Tuple[int, int, int]:
    n_train = int(np.floor(n * train_ratio))
    n_valid = int(np.floor(n * valid_ratio))
    n_test = max(0, n - n_train - n_valid)
    return n_train, n_valid, n_test


def _extract_numeric_id(path: Path) -> Optional[int]:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


class DeprecatedSpatialSenseDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "dataset=spatialsense_position is deprecated and removed. "
            "Use dataset=real_world_position with "
            "launch_script/spatialsense_rebuttal.py (now real-world rebuttal launcher)."
        )

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError("Deprecated dataset is not usable.")


class RealWorldRelativePosition(torch.utils.data.Dataset):
    """
    Folder-based 4-way relative position dataset.
    Directory layout:
      root/
        front/*.jpg
        back/*.jpg
        left/*.jpg
        right/*.jpg
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        split_mode: str = "random",
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        perspective: Optional[str] = None,
        seed: int = 8,
        recursive: bool = True,
        clip_block_size: int = 20,
        clip_id_source: str = "filename_numeric",
        name: str = "real_world_position",
        **_unused_kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.name = str(name)
        self.seed = int(seed)
        self.split = _normalize_split(split)
        self.split_mode = str(split_mode).strip().lower()
        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)
        self.test_ratio = float(test_ratio)
        self.perspective = str(perspective) if perspective is not None else self.split_mode
        self.recursive = bool(recursive)
        self.clip_block_size = int(clip_block_size)
        self.clip_id_source = str(clip_id_source).strip().lower()

        if self.split not in {"train", "valid", "test", "trainval"}:
            raise ValueError(f"Unsupported split={split}. Use train, valid, test, or trainval.")
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(
                "Unsupported split_mode="
                f"{split_mode}. Use random, time_series, clip_block_random, or clip_block_time_series."
            )
        if self.clip_block_size <= 0:
            raise ValueError(f"clip_block_size must be positive, got {self.clip_block_size}.")
        if self.clip_id_source not in {"filename_numeric", "parent_folder"}:
            raise ValueError(
                f"Unsupported clip_id_source={clip_id_source}. "
                "Use filename_numeric or parent_folder."
            )
        ratio_sum = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(
                f"Split ratios must sum to 1.0, got train+valid+test={ratio_sum:.6f}."
            )

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
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.class_names: List[str] = ["Front", "Back", "Left", "Right"]
        self.class_order: List[int] = [LABEL_TO_INDEX[name] for name in self.class_names]
        self.label_to_index: Dict[str, int] = dict(LABEL_TO_INDEX)
        self.index_to_label: Dict[int, str] = dict(INDEX_TO_LABEL)
        self.num_classes = len(self.class_names)

        class_files = self._index_files_by_class()
        self.samples = self._build_split_samples(class_files)

    def _index_files_by_class(self) -> Dict[str, List[Path]]:
        class_files: Dict[str, List[Path]] = {}
        for folder_name, class_label in CLASS_FOLDER_TO_LABEL.items():
            class_dir = self.root / folder_name
            if not class_dir.exists():
                class_files[class_label] = []
                continue
            class_files[class_label] = _collect_images(class_dir, recursive=self.recursive)
        return class_files

    def _select_split_indices(self, n: int, rng: Optional[np.random.RandomState]) -> Dict[str, np.ndarray]:
        order = np.arange(n, dtype=np.int64)
        if self.split_mode == "random" and rng is not None:
            rng.shuffle(order)
        n_train, n_valid, _ = _split_counts(n, self.train_ratio, self.valid_ratio)
        if self.split_mode == "time_series":
            # Keep earliest samples for train, then randomly split only the tail to val/test.
            train_idx = order[:n_train]
            tail_idx = order[n_train:]
            if rng is not None:
                tail_idx = tail_idx.copy()
                rng.shuffle(tail_idx)
            valid_idx = tail_idx[:n_valid]
            test_idx = tail_idx[n_valid:]
            return {"train": train_idx, "valid": valid_idx, "test": test_idx}
        train_idx = order[:n_train]
        valid_idx = order[n_train:n_train + n_valid]
        test_idx = order[n_train + n_valid:]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def _build_split_samples(self, class_files: Dict[str, Sequence[Path]]) -> List[Tuple[Path, int]]:
        if self.split_mode in {"clip_block_random", "clip_block_time_series"}:
            return self._build_clip_block_split_samples(class_files)

        rng = np.random.RandomState(self.seed) if self.split_mode in {"random", "time_series"} else None
        selected: List[Tuple[Path, int]] = []

        for class_label in self.class_names:
            files = list(class_files.get(class_label, []))
            if len(files) == 0:
                continue
            split_to_idx = self._select_split_indices(len(files), rng=rng)
            if self.split == "trainval":
                idx = np.concatenate([split_to_idx["train"], split_to_idx["valid"]], axis=0)
            else:
                idx = split_to_idx[self.split]
            for i in idx.tolist():
                selected.append((files[int(i)], int(LABEL_TO_INDEX[class_label])))

        return selected

    def _clip_block_id(self, path: Path, class_min_frame_id: Optional[int]):
        if self.clip_id_source == "parent_folder":
            return f"dir::{path.parent.as_posix()}"
        frame_id = _extract_numeric_id(path)
        if frame_id is None:
            return -1
        min_id = class_min_frame_id if class_min_frame_id is not None else frame_id
        return int((frame_id - min_id) // self.clip_block_size)

    def _build_clip_block_split_samples(
        self, class_files: Dict[str, Sequence[Path]]
    ) -> List[Tuple[Path, int]]:
        rng = (
            np.random.RandomState(self.seed)
            if self.split_mode in {"clip_block_random", "clip_block_time_series"}
            else None
        )
        selected: List[Tuple[Path, int]] = []

        for class_label in self.class_names:
            files = list(class_files.get(class_label, []))
            if len(files) == 0:
                continue

            frame_ids = [_extract_numeric_id(p) for p in files]
            valid_ids = [x for x in frame_ids if x is not None]
            class_min_frame_id = min(valid_ids) if valid_ids else None

            block_to_indices: Dict[int, List[int]] = {}
            for i, p in enumerate(files):
                block_id = self._clip_block_id(p, class_min_frame_id)
                block_to_indices.setdefault(block_id, []).append(i)

            block_ids = sorted(block_to_indices.keys())
            if not block_ids:
                continue
            order = np.arange(len(block_ids), dtype=np.int64)
            if self.split_mode == "clip_block_random" and rng is not None:
                rng.shuffle(order)
            n_train, n_valid, _ = _split_counts(len(block_ids), self.train_ratio, self.valid_ratio)
            if self.split_mode == "clip_block_time_series":
                # Keep earliest blocks for train, then randomly split tail blocks to val/test.
                train_ord = order[:n_train]
                tail_ord = order[n_train:]
                if rng is not None:
                    tail_ord = tail_ord.copy()
                    rng.shuffle(tail_ord)
                valid_ord = tail_ord[:n_valid]
                test_ord = tail_ord[n_valid:]
            else:
                train_ord = order[:n_train]
                valid_ord = order[n_train : n_train + n_valid]
                test_ord = order[n_train + n_valid :]

            if self.split == "train":
                selected_blocks = train_ord
            elif self.split == "valid":
                selected_blocks = valid_ord
            elif self.split == "test":
                selected_blocks = test_ord
            elif self.split == "trainval":
                selected_blocks = np.concatenate([train_ord, valid_ord], axis=0)
            else:
                raise ValueError(f"Unsupported split: {self.split}")

            keep_indices: List[int] = []
            for ord_idx in selected_blocks.tolist():
                block_id = block_ids[int(ord_idx)]
                keep_indices.extend(block_to_indices[block_id])
            keep_indices.sort()

            for i in keep_indices:
                selected.append((files[int(i)], int(LABEL_TO_INDEX[class_label])))

        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        with Image.open(image_path).convert("RGB") as im:
            image = im.copy()
        image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(label, dtype=torch.long),
        }
