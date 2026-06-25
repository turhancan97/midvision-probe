from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .real_world_position import RealWorldRelativePosition
from .unreal_position import LABEL_TO_INDEX, UnrealRelativePosition


INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def _normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split == "val":
        return "valid"
    return split


class UnrealRealHoldoutTransfer(Dataset):
    """
    Sim-to-real transfer dataset:
      - protocol=loto_source_to_target:
          train/valid from Unreal source pool, test from real-world dataset
      - protocol=target_only:
          train/valid/test from real-world dataset
    """

    def __init__(
        self,
        root: str,
        real_world_root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        perspective: str = "human",
        real_world_perspective: str = "lego_human",
        human_label: Optional[str] = "Human",
        exclude_ambiguous: bool = True,
        ambiguity_degrees: int = 20,
        front_degrees: int = 45,
        back_degrees: int = 135,
        unreal_split_ratio: float = 0.2,
        unreal_test_ratio: float = 0.1,
        split_mode: str = "time_series",
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        clip_block_size: int = 20,
        clip_id_source: str = "filename_numeric",
        recursive: bool = True,
        seed: int = 8,
        name: str = "unreal_real_holdout_transfer",
        protocol: str = "loto_source_to_target",
        holdout_environment: str = "real_world",
        environments: Optional[List[str]] = None,
        excluded_environments: Optional[List[str]] = None,
        environment_to_labels: Optional[Dict[str, Dict[str, str]]] = None,
        **_unused_kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.real_world_root = Path(real_world_root)
        self.split = _normalize_split(split)
        self.name = str(name)
        self.protocol = str(protocol).strip().lower()
        self.holdout_environment = str(holdout_environment).strip()
        self.perspective = str(perspective).strip().lower()
        self.real_world_perspective = str(real_world_perspective).strip()
        self.seed = int(seed)

        self.class_names: List[str] = ["Front", "Back", "Left", "Right"]
        self.class_order: List[int] = [LABEL_TO_INDEX[c] for c in self.class_names]
        self.label_to_index: Dict[str, int] = {
            "Front": LABEL_TO_INDEX["Front"],
            "Back": LABEL_TO_INDEX["Back"],
            "Left": LABEL_TO_INDEX["Left"],
            "Right": LABEL_TO_INDEX["Right"],
        }
        self.index_to_label: Dict[int, str] = {self.label_to_index[k]: k for k in self.label_to_index}
        self.num_classes = 4

        if self.split not in {"train", "valid", "test", "trainval"}:
            raise ValueError(f"Unsupported split={split}. Use train, valid, test, or trainval.")
        if self.protocol not in {"loto_source_to_target", "target_only"}:
            raise ValueError(
                f"Unsupported protocol={protocol}. Use loto_source_to_target or target_only."
            )
        if self.perspective != "human":
            raise ValueError(
                f"Unsupported perspective={perspective} for real-world holdout transfer. Use human."
            )
        if self.holdout_environment.lower() != "real_world":
            raise ValueError(
                "holdout_environment must be set to 'real_world' for this dataset."
            )
        if environment_to_labels is None:
            raise ValueError("environment_to_labels mapping must be provided.")

        env_to_labels = {str(k): dict(v) for k, v in dict(environment_to_labels).items()}
        all_envs = list(environments) if environments is not None else list(env_to_labels.keys())
        excluded = set(excluded_environments or [])
        all_envs = [e for e in all_envs if e not in excluded]
        if not all_envs:
            raise ValueError("No Unreal source environments available after exclusions.")
        missing = [e for e in all_envs if e not in env_to_labels]
        if missing:
            raise ValueError(f"Missing environment_to_labels entries for: {missing}")

        self.samples: List[Tuple[Dataset, int]] = []
        self.env_counts: Dict[str, int] = {}

        use_real_world = self.protocol == "target_only" or self.split == "test"
        if self.protocol == "loto_source_to_target":
            if self.split not in {"train", "valid", "test", "trainval"}:
                raise ValueError(f"Unsupported split={split}")
            if self.split == "trainval":
                raise ValueError(
                    "split=trainval is not supported for loto_source_to_target in this dataset."
                )

        if use_real_world:
            ds_real = RealWorldRelativePosition(
                root=str(self.real_world_root),
                split=self.split,
                image_size=image_size,
                image_mean=image_mean,
                split_mode=split_mode,
                train_ratio=float(train_ratio),
                valid_ratio=float(valid_ratio),
                test_ratio=float(test_ratio),
                perspective=self.real_world_perspective,
                seed=self.seed,
                recursive=bool(recursive),
                clip_block_size=int(clip_block_size),
                clip_id_source=str(clip_id_source),
                name=f"{self.name}:real_world",
            )
            self.env_counts["real_world"] = len(ds_real)
            for idx in range(len(ds_real)):
                self.samples.append((ds_real, idx))
        else:
            for env in all_envs:
                labels = env_to_labels[env]
                reference_label = str(labels["reference_label"])
                target_label = str(labels["target_label"])
                env_root = self.root / env / "mid-objects"
                ds_unreal = UnrealRelativePosition(
                    root=str(env_root),
                    split=self.split,
                    image_size=image_size,
                    image_mean=image_mean,
                    reference_label=reference_label,
                    target_label=target_label,
                    exclude_ambiguous=bool(exclude_ambiguous),
                    ambiguity_degrees=int(ambiguity_degrees),
                    front_degrees=int(front_degrees),
                    back_degrees=int(back_degrees),
                    split_ratio=float(unreal_split_ratio),
                    test_ratio=float(unreal_test_ratio),
                    seed=self.seed,
                    perspective=self.perspective,
                    human_label=human_label,
                    name=f"{self.name}:{env}",
                )
                self.env_counts[env] = len(ds_unreal)
                for idx in range(len(ds_unreal)):
                    self.samples.append((ds_unreal, idx))

        if not self.samples:
            raise ValueError(
                f"No samples found for protocol={self.protocol}, split={self.split}, "
                f"holdout_environment={self.holdout_environment}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ds, sample_idx = self.samples[idx]
        sample = ds[sample_idx]
        return {
            "image": sample["image"],
            "label": sample["label"],
            "class_id": sample["class_id"],
        }
