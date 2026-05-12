from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


LABEL_TO_INDEX: Dict[str, int] = {"Left": 0, "Right": 1, "Front": 2, "Back": 3}
INDEX_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_INDEX.items()}
PREDICATE_TO_LABEL: Dict[str, str] = {
    "to the left of": "Left",
    "to the right of": "Right",
    "in front of": "Front",
    "behind": "Back",
}


def _normalize_split(split: str) -> str:
    split = str(split).strip().lower()
    if split == "val":
        return "valid"
    return split


def _resolve_image_path(root: Path, filename: str) -> Optional[Path]:
    candidates = [
        root / "images" / "flickr" / filename,
        root / "images" / "nyu" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _pair_key(annotation: dict) -> Tuple:
    subject = annotation.get("subject", {})
    obj = annotation.get("object", {})
    return (
        subject.get("name", ""),
        tuple(subject.get("bbox", [])),
        obj.get("name", ""),
        tuple(obj.get("bbox", [])),
    )


def _union_bbox(subject_bbox: List[int], object_bbox: List[int]) -> Tuple[int, int, int, int]:
    sy1, sy2, sx1, sx2 = [int(v) for v in subject_bbox]
    oy1, oy2, ox1, ox2 = [int(v) for v in object_bbox]
    y1 = min(sy1, oy1)
    y2 = max(sy2, oy2)
    x1 = min(sx1, ox1)
    x2 = max(sx2, ox2)
    return y1, y2, x1, x2


class SpatialSenseRelativePosition(torch.utils.data.Dataset):
    """
    SpatialSense-based relative position dataset for 4-way direction classification.
    Each sample corresponds to one (subject, object) pair annotation.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        image_mean: str = "imagenet",
        directional_only: bool = True,
        positive_only: bool = True,
        drop_multi_positive_pairs: bool = True,
        input_mode: str = "pair_union_crop",
        crop_padding_ratio: float = 0.10,
        perspective: Optional[str] = None,
        exclude_ambiguous: Optional[bool] = None,
        role_cue_enabled: bool = True,
        role_cue_target_role: str = "subject",
        role_cue_black_background: bool = True,
        seed: int = 8,
        name: str = "spatialsense_position",
        **_unused_kwargs,
    ):
        super().__init__()
        del seed, exclude_ambiguous  # shared config compatibility

        self.root = Path(root)
        self.name = name
        self.split = _normalize_split(split)
        self.directional_only = bool(directional_only)
        self.positive_only = bool(positive_only)
        self.drop_multi_positive_pairs = bool(drop_multi_positive_pairs)
        self.input_mode = str(input_mode).strip().lower()
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.perspective = str(perspective) if perspective is not None else self.input_mode
        self.role_cue_enabled = bool(role_cue_enabled)
        self.role_cue_target_role = str(role_cue_target_role).strip().lower()
        self.role_cue_black_background = bool(role_cue_black_background)

        if self.input_mode not in {"pair_union_crop", "full_image"}:
            raise ValueError(f"Unsupported input_mode={input_mode}. Use pair_union_crop or full_image.")
        if not self.directional_only:
            raise ValueError("SpatialSenseRelativePosition currently supports directional_only=True only.")
        if not self.positive_only:
            raise ValueError("SpatialSenseRelativePosition currently supports positive_only=True only.")
        if self.role_cue_target_role not in {"subject", "object"}:
            raise ValueError(
                f"Unsupported role_cue_target_role={role_cue_target_role}. Use subject or object."
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

        self.class_names: List[str] = ["Left", "Right", "Front", "Back"]
        self.class_order: List[int] = [LABEL_TO_INDEX[name] for name in self.class_names]
        self.label_to_index: Dict[str, int] = dict(LABEL_TO_INDEX)
        self.index_to_label: Dict[int, str] = dict(INDEX_TO_LABEL)
        self.num_classes = len(self.class_names)

        self.samples: List[dict] = []
        self.missing_images = 0
        self.dropped_multi_positive = 0
        self.total_pair_groups = 0
        self._index_samples()

    def _index_samples(self) -> None:
        annotation_path = self.root / "annotations.json"
        with annotation_path.open("r") as f:
            records = json.load(f)

        target_split = _normalize_split(self.split)
        if target_split == "trainval":
            valid_splits = {"train", "valid"}
        else:
            valid_splits = {target_split}

        for record_idx, record in enumerate(records):
            record_split = _normalize_split(record.get("split", ""))
            if record_split not in valid_splits:
                continue

            url = record.get("url", "")
            filename = url.rsplit("/", 1)[-1] if "/" in url else ""
            if not filename:
                continue
            image_path = _resolve_image_path(self.root, filename)
            if image_path is None:
                self.missing_images += 1
                continue

            grouped: Dict[Tuple, set] = defaultdict(set)
            grouped_meta: Dict[Tuple, Tuple[str, str, List[int], List[int]]] = {}
            for ann in record.get("annotations", []):
                pred = ann.get("predicate", "")
                if pred not in PREDICATE_TO_LABEL:
                    continue
                if self.positive_only and not bool(ann.get("label", False)):
                    continue
                key = _pair_key(ann)
                subject_bbox = ann.get("subject", {}).get("bbox", None)
                object_bbox = ann.get("object", {}).get("bbox", None)
                if not subject_bbox or not object_bbox or len(subject_bbox) != 4 or len(object_bbox) != 4:
                    continue
                grouped[key].add(PREDICATE_TO_LABEL[pred])
                grouped_meta[key] = (
                    str(ann.get("subject", {}).get("name", "")),
                    str(ann.get("object", {}).get("name", "")),
                    subject_bbox,
                    object_bbox,
                )

            for pair_key, classes in grouped.items():
                self.total_pair_groups += 1
                if len(classes) == 0:
                    continue
                if len(classes) > 1 and self.drop_multi_positive_pairs:
                    self.dropped_multi_positive += 1
                    continue
                class_name = sorted(classes)[0]
                sname, oname, subject_bbox, object_bbox = grouped_meta[pair_key]
                union_bbox = _union_bbox(subject_bbox, object_bbox)
                self.samples.append(
                    {
                        "record_index": record_idx,
                        "image_path": image_path,
                        "image_id": filename,
                        "subject_name": sname,
                        "object_name": oname,
                        "label_name": class_name,
                        "label_idx": LABEL_TO_INDEX[class_name],
                        "subject_bbox": [int(v) for v in subject_bbox],
                        "object_bbox": [int(v) for v in object_bbox],
                        "union_bbox": [int(v) for v in union_bbox],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_union_with_padding(self, image: Image.Image, union_bbox: List[int]) -> Image.Image:
        width, height = image.size
        y1, y2, x1, x2 = [int(v) for v in union_bbox]
        box_h = max(1, y2 - y1)
        box_w = max(1, x2 - x1)
        pad_h = int(round(box_h * self.crop_padding_ratio))
        pad_w = int(round(box_w * self.crop_padding_ratio))

        y1p = max(0, y1 - pad_h)
        y2p = min(height, y2 + pad_h)
        x1p = max(0, x1 - pad_w)
        x2p = min(width, x2 + pad_w)
        if y2p <= y1p or x2p <= x1p:
            return image
        return image.crop((x1p, y1p, x2p, y2p))

    @staticmethod
    def _clamp_bbox(bbox: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
        y1, y2, x1, x2 = [int(v) for v in bbox]
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        return y1, y2, x1, x2

    def _apply_role_cue(self, image: Image.Image, sample: dict) -> Image.Image:
        width, height = image.size
        target_key = "subject_bbox" if self.role_cue_target_role == "subject" else "object_bbox"
        reference_key = "object_bbox" if self.role_cue_target_role == "subject" else "subject_bbox"

        if self.role_cue_black_background:
            styled = Image.new("RGB", image.size, (0, 0, 0))
        else:
            styled = image.copy()

        ry1, ry2, rx1, rx2 = self._clamp_bbox(sample[reference_key], width, height)
        if ry2 > ry1 and rx2 > rx1:
            ref_patch = image.crop((rx1, ry1, rx2, ry2))
            styled.paste(ref_patch, (rx1, ry1))

        ty1, ty2, tx1, tx2 = self._clamp_bbox(sample[target_key], width, height)
        if ty2 > ty1 and tx2 > tx1:
            tgt_patch = image.crop((tx1, ty1, tx2, ty2))
            tgt_patch = ImageOps.grayscale(tgt_patch).convert("RGB")
            styled.paste(tgt_patch, (tx1, ty1))

        return styled

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path: Path = sample["image_path"]
        with Image.open(image_path).convert("RGB") as im:
            image = im.copy()

        if self.role_cue_enabled:
            image = self._apply_role_cue(image, sample)

        if self.input_mode == "pair_union_crop":
            image = self._crop_union_with_padding(image, sample["union_bbox"])

        image = self.transform(image)
        label = int(sample["label_idx"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "class_id": torch.tensor(label, dtype=torch.long),
        }
