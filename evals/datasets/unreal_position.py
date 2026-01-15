from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


LABEL_TO_INDEX = {"Front": 0, "Back": 1, "Left": 2, "Right": 3, "Ambiguous": 4}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def classify_relative_direction(
    observer_xy: np.ndarray,
    ref_xy: np.ndarray,
    tgt_xy: np.ndarray,
    amb_deg: int = 20,
    front_deg: int = 45,
    back_deg: int = 135,
) -> str:
    # Vectors
    fwd = ref_xy - observer_xy  # observer -> reference
    rel = tgt_xy - ref_xy  # reference -> target

    nf = np.linalg.norm(fwd)
    nr = np.linalg.norm(rel)
    if nf == 0 or nr == 0:
        return "Ambiguous"

    f = fwd / nf
    r = rel / nr

    dot = float(np.dot(f, r))
    det = float(f[0] * r[1] - f[1] * r[0])
    ang = np.degrees(np.arctan2(det, dot))  # (-180, 180]
    abs_ang = abs(ang)

    if (front_deg - amb_deg) <= abs_ang <= (front_deg + amb_deg) or (
        back_deg - amb_deg
    ) <= abs_ang <= (back_deg + amb_deg):
        return "Ambiguous"

    if abs_ang < (front_deg - amb_deg):
        return "Front"
    if abs_ang > (back_deg + amb_deg):
        return "Back"
    return "Left" if ang > 0 else "Right"


def _matching_json_for_image(image_path: Path) -> Optional[Path]:
    patterns = [r"img_(\d+)\.jpg$", r"img_(\d+)\.jpeg$"]
    m = None
    for pat in patterns:
        m = re.search(pat, image_path.name)
        if m:
            break
    if not m:
        return None
    idx = m.group(1).lstrip("0")
    if idx == "":
        idx = "0"
    json_name = f"params_{idx}.json"
    jp = image_path.parent / json_name
    if not jp.exists():
        # fallback to zero-padding (4 or 5)
        for width in (4, 5):
            zp = image_path.parent / f"params_{int(idx):0{width}d}.json"
            if zp.exists():
                return zp
        return None
    return jp


def _load_positions(
    json_path: Path,
    reference_label: str,
    target_label: str,
    human_label: Optional[str] = None,
) -> Optional[Dict[str, Optional[np.ndarray]]]:
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
        cam_pos = metadata["camera"]["location"]
        actors = metadata["actors"]
        positions = {v["label"]: v["location"] for v in actors.values()}
        if reference_label not in positions or target_label not in positions:
            return None
        # invert X coordinate as provided
        cam = np.array([-1 * cam_pos["x"], cam_pos["y"]], dtype=float)
        ref = np.array(
            [-1 * positions[reference_label]["x"], positions[reference_label]["y"]],
            dtype=float,
        )
        tgt = np.array(
            [-1 * positions[target_label]["x"], positions[target_label]["y"]],
            dtype=float,
        )
        human = None
        if human_label:
            actor_loc = positions.get(human_label)
            if actor_loc is not None:
                human = np.array([-1 * actor_loc["x"], actor_loc["y"]], dtype=float)
        return {"camera": cam, "reference": ref, "target": tgt, "human": human}
    except Exception:
        return None


@dataclass
class SplitConfig:
    split_ratio: float = 0.1  # val fraction
    seed: int = 8


class UnrealRelativePosition(torch.utils.data.Dataset):
    """
    Unreal synthetic dataset for relative position classification between two objects.
    Returns dicts with keys: image (CxHxW), label (LongTensor), class_id (LongTensor).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        reference_label: str = "Rock",
        target_label: str = "Truck",
        exclude_ambiguous: bool = True,
        ambiguity_degrees: int = 20,
        front_degrees: int = 45,
        back_degrees: int = 135,
        split_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 8,
        name: str = "unreal_position",
        perspective: str = "camera",
        human_label: Optional[str] = "Human",
    ):
        super().__init__()
        self.root = Path(root)
        self.name = name
        self.split = split
        self.reference_label = reference_label
        self.target_label = target_label
        self.exclude_ambiguous = exclude_ambiguous
        self.amb_deg = ambiguity_degrees
        self.front_deg = front_degrees
        self.back_deg = back_degrees
        self.split_cfg = SplitConfig(split_ratio=split_ratio, seed=seed)
        self.test_ratio = float(test_ratio)
        self.perspective = perspective.lower()
        if self.perspective not in ("camera", "human"):
            raise ValueError(f"Unsupported perspective: {perspective}")
        self.human_label = human_label
        if self.human_label is not None:
            self.human_label = str(self.human_label)
        if self.perspective == "human" and not self.human_label:
            raise ValueError("human_label must be provided when perspective is 'human'.")

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

        self.samples: List[Tuple[Path, int]] = []
        self._index_samples(perspective=self.perspective)
        self._make_split_indices()

        # classes: 4 (without ambiguous) or 5 (with ambiguous)
        self.num_classes = 4 if self.exclude_ambiguous else 5

    def _index_samples(self, perspective: str = "camera"):
        imgs: List[Path] = []
        for pat in ("*.jpg", "*.jpeg"):
            if perspective == "camera":
                imgs.extend(sorted(self.root.glob(pat))[:5000]) # Because we test on 5000 images for easy task
            elif perspective == "human":
                imgs.extend(sorted(self.root.glob(pat))) # Because we test on all images for hard task
            else:
                raise ValueError(f"Unsupported perspective: {perspective}")
        for img_path in imgs:
            json_path = _matching_json_for_image(img_path)
            if json_path is None:
                continue
            positions = _load_positions(
                json_path,
                self.reference_label,
                self.target_label,
                human_label=self.human_label,
            )
            if positions is None:
                continue
            observer_key = "camera" if self.perspective == "camera" else "human"
            observer = positions.get(observer_key)
            if observer is None:
                continue
            ref = positions["reference"]
            tgt = positions["target"]
            label_str = classify_relative_direction(
                observer,
                ref,
                tgt,
                self.amb_deg,
                self.front_deg,
                self.back_deg,
            )
            if self.exclude_ambiguous and label_str == "Ambiguous":
                continue
            label_idx = LABEL_TO_INDEX[label_str]
            self.samples.append((img_path, label_idx))

    def _make_split_indices(self):
        n = len(self.samples)
        rng = np.random.RandomState(self.split_cfg.seed)
        perm = rng.permutation(n)
        val_n = int(self.split_cfg.split_ratio * n)
        test_n = int(self.test_ratio * n)
        self.idx_val = perm[:val_n].tolist()
        self.idx_train = perm[val_n : n - test_n].tolist() if test_n > 0 else perm[val_n:].tolist()
        self.idx_test = perm[n - test_n :].tolist() if test_n > 0 else []
        self.idx_trainval = self.idx_train + self.idx_val

    def __len__(self) -> int:
        if self.split in ("valid", "val"):
            return len(self.idx_val)
        elif self.split in ("test",):
            return len(self.idx_test)
        elif self.split in ("train",):
            return len(self.idx_train)
        elif self.split in ("trainval",):
            return len(self.idx_trainval)
        else:
            raise ValueError(f"Unsupported split: {self.split}")

    def __getitem__(self, idx: int):
        if self.split in ("valid", "val"):
            real_idx = self.idx_val[idx]
        elif self.split in ("test",):
            real_idx = self.idx_test[idx]
        elif self.split in ("train",):
            real_idx = self.idx_train[idx]
        elif self.split in ("trainval",):
            real_idx = self.idx_trainval[idx]
        else:
            raise ValueError(f"Unsupported split: {self.split}")
        img_path, label = self.samples[real_idx]
        with Image.open(img_path).convert("RGB") as im:
            image = im.copy()
        image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(label, dtype=torch.long),
        }
