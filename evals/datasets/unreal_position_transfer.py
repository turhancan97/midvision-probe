from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .unreal_position import LABEL_TO_INDEX, UnrealRelativePosition


INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def _normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split == "val":
        return "valid"
    return split


class UnrealRelativePositionTransfer(Dataset):
    """
    Multi-environment Unreal relative position dataset for transfer protocols.

    Protocol modes:
      - loto_source_to_target:
          train/valid from non-holdout environments, test from holdout environment
      - target_only:
          train/valid/test from holdout environment only
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        perspective: str = "camera",
        human_label: Optional[str] = "Human",
        exclude_ambiguous: bool = True,
        ambiguity_degrees: int = 20,
        front_degrees: int = 45,
        back_degrees: int = 135,
        split_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 8,
        name: str = "unreal_position_transfer",
        protocol: str = "loto_source_to_target",
        holdout_environment: str = "bridge_2",
        environments: Optional[List[str]] = None,
        excluded_environments: Optional[List[str]] = None,
        environment_to_labels: Optional[Dict[str, Dict[str, str]]] = None,
        **_unused_kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = _normalize_split(split)
        self.name = name
        self.protocol = str(protocol).strip().lower()
        self.holdout_environment = str(holdout_environment).strip()
        self.perspective = str(perspective).strip().lower()
        self.class_names: List[str] = ["Front", "Back", "Left", "Right"]
        self.class_order: List[int] = [LABEL_TO_INDEX[c] for c in self.class_names]
        self.label_to_index: Dict[str, int] = dict(LABEL_TO_INDEX)
        self.index_to_label: Dict[int, str] = dict(INDEX_TO_LABEL)
        self.num_classes = 4

        if not self.holdout_environment:
            raise ValueError("holdout_environment must be provided.")
        if self.protocol not in {"loto_source_to_target", "target_only"}:
            raise ValueError(
                f"Unsupported protocol={protocol}. Use loto_source_to_target or target_only."
            )
        if environment_to_labels is None:
            raise ValueError("environment_to_labels mapping must be provided.")

        env_to_labels = {str(k): dict(v) for k, v in dict(environment_to_labels).items()}
        all_envs = list(environments) if environments is not None else list(env_to_labels.keys())
        excluded = set(excluded_environments or [])
        all_envs = [e for e in all_envs if e not in excluded]
        if self.holdout_environment not in all_envs:
            raise ValueError(
                f"holdout_environment={self.holdout_environment} is not in active environments={all_envs}."
            )
        missing = [e for e in all_envs if e not in env_to_labels]
        if missing:
            raise ValueError(f"Missing environment_to_labels entries for: {missing}")

        if self.protocol == "loto_source_to_target":
            if self.split in {"train", "valid", "trainval"}:
                active_envs = [e for e in all_envs if e != self.holdout_environment]
            elif self.split == "test":
                active_envs = [self.holdout_environment]
            else:
                raise ValueError(f"Unsupported split={split}")
        else:  # target_only
            active_envs = [self.holdout_environment]

        if not active_envs:
            raise ValueError(
                "No active environments selected for this split/protocol configuration."
            )

        self.samples: List[Tuple[UnrealRelativePosition, int, str]] = []
        self.env_counts: Dict[str, int] = {}
        for env in active_envs:
            labels = env_to_labels[env]
            reference_label = str(labels["reference_label"])
            target_label = str(labels["target_label"])
            env_root = self.root / env / "mid-objects"
            ds = UnrealRelativePosition(
                root=str(env_root),
                split=self.split,
                image_size=image_size,
                image_mean=image_mean,
                reference_label=reference_label,
                target_label=target_label,
                exclude_ambiguous=exclude_ambiguous,
                ambiguity_degrees=ambiguity_degrees,
                front_degrees=front_degrees,
                back_degrees=back_degrees,
                split_ratio=split_ratio,
                test_ratio=test_ratio,
                seed=seed,
                perspective=self.perspective,
                human_label=human_label,
                name=f"{name}:{env}",
            )
            self.env_counts[env] = len(ds)
            for idx in range(len(ds)):
                self.samples.append((ds, idx, env))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ds, sample_idx, _env = self.samples[idx]
        sample = ds[sample_idx]
        return {
            "image": sample["image"],
            "label": sample["label"],
            "class_id": sample["class_id"],
        }
