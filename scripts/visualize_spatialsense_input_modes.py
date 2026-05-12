from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Ensure repo root is importable when running script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.datasets.spatialsense_position import SpatialSenseRelativePosition


RELATION_TEXT = {
    "Left": "to the left of",
    "Right": "to the right of",
    "Front": "in front of",
    "Back": "behind",
}


def resolve_roles(target_role: str) -> tuple[str, str]:
    if target_role not in {"subject", "object"}:
        raise ValueError(f"Unsupported target_role={target_role}")
    if target_role == "subject":
        return "subject", "object"
    return "object", "subject"


def mask_to_subject_object_only(image: Image.Image, sample: dict) -> Image.Image:
    black = Image.new("RGB", image.size, (0, 0, 0))
    sx1, sy1, sx2, sy2 = sample["subject_bbox"][2], sample["subject_bbox"][0], sample["subject_bbox"][3], sample["subject_bbox"][1]
    ox1, oy1, ox2, oy2 = sample["object_bbox"][2], sample["object_bbox"][0], sample["object_bbox"][3], sample["object_bbox"][1]

    subj_patch = image.crop((sx1, sy1, sx2, sy2))
    obj_patch = image.crop((ox1, oy1, ox2, oy2))
    black.paste(subj_patch, (sx1, sy1))
    black.paste(obj_patch, (ox1, oy1))
    return black


def render_target_grayscale_on_black(
    image: Image.Image,
    sample: dict,
    target_role: str = "subject",
) -> Image.Image:
    target_key, reference_key = resolve_roles(target_role)
    styled = Image.new("RGB", image.size, (0, 0, 0))

    ref_bbox_key = "subject_bbox" if reference_key == "subject" else "object_bbox"
    ry1, ry2, rx1, rx2 = sample[ref_bbox_key]
    ref_patch = image.crop((rx1, ry1, rx2, ry2))
    styled.paste(ref_patch, (rx1, ry1))

    tgt_bbox_key = "subject_bbox" if target_key == "subject" else "object_bbox"
    ty1, ty2, tx1, tx2 = sample[tgt_bbox_key]
    tgt_patch = image.crop((tx1, ty1, tx2, ty2))
    tgt_gray_patch = ImageOps.grayscale(tgt_patch).convert("RGB")
    styled.paste(tgt_gray_patch, (tx1, ty1))
    return styled


def draw_annotation(image: Image.Image, sample: dict) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    sy1, sy2, sx1, sx2 = sample["subject_bbox"]
    oy1, oy2, ox1, ox2 = sample["object_bbox"]

    # Draw only the two objects that define the spatial label.
    draw.rectangle((sx1, sy1, sx2, sy2), outline=(255, 60, 60), width=3)
    draw.rectangle((ox1, oy1, ox2, oy2), outline=(70, 130, 255), width=3)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SpatialSense input modes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/shared/sets/datasets/vision/SpatialSense"),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.10)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="target_grayscale",
        choices=["target_grayscale", "object_only_black"],
        help="Visualization style for role cues.",
    )
    parser.add_argument(
        "--target-role",
        type=str,
        default="subject",
        choices=["subject", "object"],
        help="Which role gets grayscale when render-mode=target_grayscale.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/outputs/spatialsense_input_modes_comparison.png"),
    )
    args = parser.parse_args()

    ds = SpatialSenseRelativePosition(
        root=str(args.dataset_root),
        split=args.split,
        input_mode="pair_union_crop",
        crop_padding_ratio=args.crop_padding_ratio,
        image_size=args.image_size,
    )

    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(ds.samples))
    if n == 0:
        raise RuntimeError("No samples available for visualization.")
    sample_indices = rng.choice(len(ds.samples), size=n, replace=False)

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(12, max(3, n * 3)))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_i, sample_idx in enumerate(sample_indices):
        sample = ds.samples[int(sample_idx)]
        image_path = Path(sample["image_path"])
        with Image.open(image_path).convert("RGB") as im:
            full = im.copy()

        if args.render_mode == "object_only_black":
            styled = mask_to_subject_object_only(full, sample)
        else:
            styled = render_target_grayscale_on_black(full, sample, target_role=args.target_role)

        annotated = draw_annotation(styled, sample)
        full_model_view = styled.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        crop_model_view = ds._crop_union_with_padding(styled, sample["union_bbox"]).resize(
            (args.image_size, args.image_size),
            Image.Resampling.BICUBIC,
        )
        subject_name = str(sample.get("subject_name", "subject"))
        object_name = str(sample.get("object_name", "object"))
        target_key, reference_key = resolve_roles(args.target_role)
        target_name = subject_name if target_key == "subject" else object_name
        reference_name = object_name if reference_key == "object" else subject_name
        label_name = str(sample.get("label_name", "relation"))
        relation = RELATION_TEXT.get(label_name, label_name.lower())

        axes[row_i, 0].imshow(np.asarray(annotated))
        axes[row_i, 0].set_title(
            f"Target(gray): {target_name}\nReference(color): {reference_name}",
            fontsize=9,
        )
        axes[row_i, 1].imshow(np.asarray(full_model_view))
        axes[row_i, 1].set_title(
            f"Model Input: full_image\n{subject_name} {relation} {object_name}",
            fontsize=9,
        )
        axes[row_i, 2].imshow(np.asarray(crop_model_view))
        axes[row_i, 2].set_title("Model Input: pair_union_crop", fontsize=9)

        for c in range(3):
            axes[row_i, c].axis("off")

    fig.suptitle(
        "SpatialSense Input Modes Comparison\n"
        + (
            "Black background + objects only: target grayscale, reference normal color (Red=subject, Blue=object)"
            if args.render_mode == "target_grayscale"
            else "Black background; only subject/object shown (Red=subject, Blue=object)"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"[DONE] Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
