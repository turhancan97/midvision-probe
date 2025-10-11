"""
Dataset utilities for the equivariance regression tasks.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


@dataclass
class _Sample:
    image_path: Path
    orientation: str
    environment: str
    level: str
    object_class: str
    frame_number: int
    target: Optional[torch.Tensor] = None


class UnrealEquivarianceTask(Dataset):
    """Dataset that exposes samples grouped by environment/level/object.

    Each sample returns the normalized regression target (line or circle) along with
    metadata that allows grouping in downstream training code.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        regression_type: str = "circle",
        frame_count: int = 540,
        split_ratio: float = 0.95,
        test_ratio: float = 0.0,
        orientation: Optional[str] = None,
        environments: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[str]] = None,
        seed: int = 8,
        name: str = "unreal_equivariance",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.regression_type = regression_type.lower()
        self.frame_count = int(frame_count)
        self.split_ratio = float(split_ratio)
        self.test_ratio = float(test_ratio)
        self.orientation = orientation or ("Orbit" if self.regression_type == "circle" else "Line")
        self.allowed_environments = set(environments) if environments else None
        self.allowed_levels = set(levels) if levels else None
        self.name = name
        self.seed = seed

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

        self.samples: List[_Sample] = []
        self._groups: DefaultDict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.group_indices_by_split: Dict[str, DefaultDict[Tuple[str, str, str], List[int]]] = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }
        self.split_indices: Dict[str, List[int]] = {
            "train": [],
            "valid": [],
            "test": [],
        }

        self._index_samples()
        self._assign_targets()
        self._build_splits()
        self._set_active_split(split)

    # ------------------------------------------------------------------
    # Dataset index helpers
    # ------------------------------------------------------------------
    def _iter_image_files(self, directory: Path) -> Iterable[Path]:
        for pattern in _IMAGE_PATTERNS:
            for image_path in sorted(directory.glob(pattern)):
                if image_path.is_file():
                    yield image_path

    def _index_samples(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        frame_regex = re.compile(r"(\d+)\.[^.]+$")
        for orientation_dir in sorted(self.root.glob("Frames_*")):
            orientation_name = orientation_dir.name.replace("Frames_", "")
            if self.orientation and orientation_name != self.orientation:
                print(f"Skipping {orientation_name} because it doesn't match {self.orientation}")
                continue

            for environment_dir in sorted(p for p in orientation_dir.iterdir() if p.is_dir()):
                environment_name = environment_dir.name
                if self.allowed_environments and environment_name not in self.allowed_environments:
                    continue

                for level_dir in sorted(p for p in environment_dir.iterdir() if p.is_dir()):
                    level_name = level_dir.name
                    if self.allowed_levels and level_name not in self.allowed_levels:
                        continue

                    for object_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                        object_name = object_dir.name
                        for image_path in self._iter_image_files(object_dir):
                            match = frame_regex.search(image_path.name)
                            if not match:
                                continue
                            frame_number = int(match.group(1))
                            sample_index = len(self.samples)
                            self.samples.append(
                                _Sample(
                                    image_path=image_path,
                                    orientation=orientation_name,
                                    environment=environment_name,
                                    level=level_name,
                                    object_class=object_name,
                                    frame_number=frame_number,
                                )
                            )
                            key = (environment_name, level_name, object_name)
                            self._groups[key].append(sample_index)

        self.available_environments = sorted({s.environment for s in self.samples})
        self.available_levels = sorted({s.level for s in self.samples})

    def _assign_targets(self) -> None:
        if not self.samples:
            return
        for key, indices in self._groups.items():
            sorted_indices = sorted(indices, key=lambda idx: self.samples[idx].frame_number)
            if self.regression_type == "circle":
                theta = torch.linspace(0.0, 2.0 * math.pi, len(sorted_indices), dtype=torch.float32)
                coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
                for position, sample_idx in enumerate(sorted_indices):
                    self.samples[sample_idx].target = coords[position]
            else:
                for sample_idx in sorted_indices:
                    fraction = float(self.samples[sample_idx].frame_number) / float(self.frame_count)
                    clipped = max(0.0, min(1.0, fraction))
                    self.samples[sample_idx].target = torch.tensor([clipped], dtype=torch.float32)

    def _build_splits(self) -> None:
        rng = np.random.RandomState(self.seed)
        for key, indices in self._groups.items():
            sorted_indices = sorted(indices, key=lambda idx: self.samples[idx].frame_number)
            n = len(sorted_indices)
            if n == 0:
                continue

            train_cut = int(round(self.split_ratio * n))
            train_cut = max(1, min(train_cut, n))
            if train_cut >= n and n > 1:
                train_cut = n - 1

            train_indices = sorted_indices[:train_cut]
            remaining = sorted_indices[train_cut:]

            test_count = int(round(self.test_ratio * n)) if self.test_ratio > 0 else 0
            test_count = min(test_count, len(remaining))
            if test_count > 0:
                rng.shuffle(remaining)
                test_indices = remaining[:test_count]
                val_indices = remaining[test_count:]
            else:
                test_indices = []
                val_indices = remaining

            for idx in train_indices:
                self.split_indices["train"].append(idx)
                self.group_indices_by_split["train"][key].append(idx)
            for idx in val_indices:
                self.split_indices["valid"].append(idx)
                self.group_indices_by_split["valid"][key].append(idx)
            for idx in test_indices:
                self.split_indices["test"].append(idx)
                self.group_indices_by_split["test"][key].append(idx)

        # Build combined splits for convenience.
        self.split_indices["trainval"] = self.split_indices["train"] + self.split_indices["valid"]

    def _set_active_split(self, split: str) -> None:
        split_key = "valid" if split in {"val", "valid"} else split
        if split_key not in self.split_indices:
            raise ValueError(f"Unsupported split: {split}")
        self.active_split = split_key
        self.active_indices = self.split_indices[split_key]

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample_idx = self.active_indices[index]
        sample = self.samples[sample_idx]
        with Image.open(sample.image_path).convert("RGB") as image:
            image_t = image.copy()
        image_t = self.transform(image_t)
        target = sample.target.clone() if sample.target is not None else torch.tensor([0.0], dtype=torch.float32)

        return {
            "image": image_t,
            "target": target,
            "environment": sample.environment,
            "level": sample.level,
            "object_class": sample.object_class,
            "orientation": sample.orientation,
            "frame_number": torch.tensor(sample.frame_number, dtype=torch.long),
            "frame_fraction": torch.tensor(float(sample.frame_number) / float(self.frame_count), dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def set_split(self, split: str) -> None:
        self._set_active_split(split)

    def groups_for_split(self, split: str) -> Dict[Tuple[str, str, str], List[int]]:
        split_key = "valid" if split in {"val", "valid"} else split
        return self.group_indices_by_split.get(split_key, {})

