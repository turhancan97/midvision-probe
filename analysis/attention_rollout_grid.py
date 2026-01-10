from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import hydra
import numpy as np
import torch
import matplotlib
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig

matplotlib.use("Agg")

from analysis.attention_stats import (
    ImageRecord,
    _chunk_tensor,
    _infer_patch_kernel,
    _load_images,
    _prepare_patch_vectors,
    _prepare_recorders,
    _resolve_mean_std,
    _reshape_attention,
    _disable_recorders,
)
from evals.utils.vit_explain import rollout as vit_rollout, show_mask_on_image


DISCARD_RATIO = 0.9
HEAD_FUSION = "mean"


def _denormalize_image(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    image = tensor.clone()
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    return image


def _build_display_cache(images: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> List[np.ndarray]:
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    cache = []
    for tensor in images:
        cache.append(_denormalize_image(tensor, mean_tensor[0], std_tensor[0]))
    return cache


def _resize_mask(mask: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
    if resized.max() > 0:
        resized = resized / resized.max()
    return resized


def _save_grid(
    image_pairs: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    output_dir: Path,
):
    if not image_pairs:
        logger.warning(f"No images available for model {model_name}; skipping grid.")
        return
    cols = 5
    pair_rows = int(np.ceil(len(image_pairs) / cols))
    rows = max(1, pair_rows * 2)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, max(1, pair_rows) * 5))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    elif axes.ndim == 1:
        axes = axes.reshape(rows, 1)

    used = np.zeros((rows, cols), dtype=bool)

    for idx, (original, overlay) in enumerate(image_pairs):
        row_pair = idx // cols
        col = idx % cols
        top_row = row_pair * 2
        bottom_row = top_row + 1
        if top_row >= rows:
            break
        top_ax = axes[top_row, col]
        bot_ax = axes[bottom_row, col] if bottom_row < rows else top_ax
        top_ax.imshow(np.clip(original, 0.0, 1.0))
        if row_pair == 0:
            top_ax.set_title(f"Image {col+1}")
        top_ax.axis("off")
        bot_ax.imshow(np.clip(overlay, 0.0, 1.0))
        bot_ax.axis("off")
        used[top_row, col] = True
        used[bottom_row, col] = True

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if not used[r, c]:
                ax.axis("off")

    fig.suptitle(f"{model_name}: Attention Rollout")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / f"{model_name}_attention_rollout_grid.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved rollout grid for {model_name} to {grid_path}")


def _compute_rollout_pairs(
    cfg: DictConfig,
    model_cfg: DictConfig,
    images: torch.Tensor,
    image_records: List[ImageRecord],
    display_cache: List[np.ndarray],
    device: torch.device,
    mask_cfg: DictConfig | None,
    zero_self_attention: bool,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    backbone = instantiate(model_cfg.backbone)
    backbone.eval()
    backbone.to(device)
    layers = list(model_cfg.layers)
    recorders, attn_names = _prepare_recorders(backbone, layers)
    logger.info(f"{model_cfg.name}: recording layers {layers}")
    logger.debug("Layer names: " + ", ".join(attn_names))
    image_h, image_w = images.shape[-2:]
    patch_kernel = _infer_patch_kernel(backbone)
    if image_h % patch_kernel[0] != 0 or image_w % patch_kernel[1] != 0:
        raise ValueError("Image size must be divisible by patch size for rollout computation.")
    patch_hw = (image_h // patch_kernel[0], image_w // patch_kernel[1])
    num_spatial = patch_hw[0] * patch_hw[1]

    image_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    discard_ratio = getattr(cfg, "discard_ratio", DISCARD_RATIO)
    head_fusion = getattr(cfg, "head_fusion", HEAD_FUSION)
    patch_vectors = None
    object_groups: Sequence[str] = ()
    object_pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    if mask_cfg is not None:
        patch_vectors = _prepare_patch_vectors(image_records, patch_kernel)
        object_groups = list(mask_cfg.groups)
        object_pairs = {group: [] for group in object_groups}

    try:
        offset = 0
        with torch.no_grad():
            for batch in _chunk_tensor(images, cfg.batch_size):
                batch = batch.to(device)
                batch_size_eff = batch.size(0)
                _ = backbone(batch)
                batch_vectors = None
                if patch_vectors is not None:
                    batch_vectors = patch_vectors[offset : offset + batch_size_eff]
                layer_attn: Dict[int, torch.Tensor] = {}
                for layer in layers:
                    records = recorders[layer].pop_records()
                    if not records:
                        raise RuntimeError(f"No attention records captured for layer {layer}.")
                    if len(records) != 1:
                        logger.warning(f"Layer {layer} produced {len(records)} records; using the last entry.")
                    attn = records[-1]
                    reshaped = _reshape_attention(attn, recorders[layer].num_heads)
                    layer_attn[layer] = reshaped.cpu()

                for sample_idx in range(batch_size_eff):
                    attentions = [
                        layer_attn[layer][sample_idx : sample_idx + 1] for layer in layers
                    ]
                    mask = vit_rollout(attentions, discard_ratio, head_fusion, num_spatial)
                    mask = _resize_mask(mask, (image_h, image_w))
                    image_index = offset + sample_idx
                    original = display_cache[image_index]
                    overlay = show_mask_on_image(original, mask)
                    overlay = overlay.astype(np.float32) / 255.0
                    image_pairs.append((original, overlay))

                    if batch_vectors is not None and object_groups:
                        sample_vectors = batch_vectors[sample_idx]
                        if not sample_vectors:
                            continue
                        object_layer_scores: Dict[str, List[torch.Tensor]] = {
                            group: [] for group in object_groups
                        }
                        seq_len = attentions[0].size(-1)
                        patch_start = seq_len - num_spatial
                        for layer in layers:
                            sample_attn = layer_attn[layer][sample_idx].mean(dim=0)
                            patch_attn = sample_attn[patch_start:, patch_start:]
                            if patch_attn.shape[0] != num_spatial:
                                continue
                            for group in object_groups:
                                vec = sample_vectors.get(group)
                                if vec is None:
                                    continue
                                if vec.numel() != num_spatial:
                                    continue
                                mask_indices = (vec > 0).nonzero(as_tuple=False).view(-1)
                                if mask_indices.numel() == 0:
                                    continue
                                per_patch_scores = patch_attn[mask_indices]
                                object_layer_scores[group].append(
                                    per_patch_scores.mean(dim=0)
                                )
                        for group, scores in object_layer_scores.items():
                            if not scores:
                                continue
                            combined = torch.stack(scores).mean(dim=0)
                            if zero_self_attention:
                                vec = sample_vectors.get(group)
                                if vec is not None and vec.numel() == num_spatial:
                                    mask_bool = (vec > 0).to(combined.device)
                                    if mask_bool.any():
                                        combined = combined.clone()
                                        combined[mask_bool.bool()] = 0.0
                            obj_mask = combined.view(patch_hw[0], patch_hw[1]).cpu().numpy()
                            if obj_mask.max() > 0:
                                obj_mask = obj_mask / obj_mask.max()
                            obj_mask = _resize_mask(obj_mask, (image_h, image_w))
                            obj_overlay = show_mask_on_image(original, obj_mask)
                            obj_overlay = obj_overlay.astype(np.float32) / 255.0
                            object_pairs[group].append((original, obj_overlay))
                offset += batch_size_eff
    finally:
        _disable_recorders(recorders)
        del backbone
    filtered_object_pairs = {k: v for k, v in object_pairs.items() if v}
    return image_pairs, filtered_object_pairs


@hydra.main(config_path="../configs", config_name="attention_stats", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mean = _resolve_mean_std(cfg.image_mean)
    std = _resolve_mean_std(cfg.image_std, fallback=[0.229, 0.224, 0.225])
    image_dir = Path(to_absolute_path(cfg.image_dir))
    output_dir = Path(to_absolute_path(cfg.output_dir))
    mask_cfg = cfg.get("object_analysis", None)
    zero_self_attention = False
    if mask_cfg is not None:
        zero_self_attention = bool(mask_cfg.get("zero_self_attention", False))
        if not mask_cfg.get("enable", True):
            mask_cfg = None

    images, image_records = _load_images(image_dir, cfg.image_size, mean, std, mask_cfg=mask_cfg)
    logger.info(f"Loaded {len(images)} images from {image_dir}")
    display_cache = _build_display_cache(images, mean, std)

    for model_cfg in cfg.models:
        logger.info(f"Generating attention rollouts for model: {model_cfg.name}")
        image_pairs, object_pairs = _compute_rollout_pairs(
            cfg,
            model_cfg,
            images,
            image_records,
            display_cache,
            device,
            mask_cfg,
            zero_self_attention,
        )
        _save_grid(image_pairs, model_cfg.name, output_dir)
        if object_pairs:
            for group, pairs in object_pairs.items():
                label = f"{model_cfg.name}_{group}"
                _save_grid(pairs, label, output_dir)


if __name__ == "__main__":
    main()
