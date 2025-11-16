from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from dataclasses import dataclass
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate, to_absolute_path
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")

from timm.layers import set_fused_attn
set_fused_attn(False)


@dataclass
class ImageRecord:
    path: Path
    masks: Dict[str, torch.Tensor]


class GroupAttentionAggregator:
    def __init__(self):
        self._sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self._count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def update(self, layer: int, query: str, target: str, value: float):
        self._sum[layer][query][target] += value
        self._count[layer][query][target] += 1

    def finalize(self) -> Dict[str, Dict[str, Dict[int, float]]]:
        per_query: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for layer, query_map in self._sum.items():
            for query, target_map in query_map.items():
                for target, total in target_map.items():
                    count = self._count[layer][query][target]
                    if count == 0:
                        continue
                    per_query[query][target][layer] = total / count
        result: Dict[str, Dict[str, Dict[int, float]]] = {}
        for query, targets in per_query.items():
            result[query] = {}
            for target, layers in targets.items():
                result[query][target] = dict(layers)
        return result



def _resolve_mean_std(value, fallback=None) -> Sequence[float]:
    if isinstance(value, (list, tuple, ListConfig)):
        return [float(v) for v in value]
    if isinstance(value, str):
        key = value.lower()
        if key == "imagenet":
            return fallback or [0.485, 0.456, 0.406]
        if key == "clip":
            return fallback or [0.48145466, 0.4578275, 0.40821073]
    if fallback is not None:
        return fallback
    raise ValueError(f"Unsupported normalization value: {value}")


def _resolve_image_hw(image_size) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
    raise ValueError(f"Unsupported image_size format: {image_size}")


def _build_transform(image_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _build_mask_transform(image_hw: Tuple[int, int]):
    return transforms.Compose(
        [
            transforms.Resize(image_hw, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_hw),
            transforms.ToTensor(),
        ]
    )


def _load_masks_for_image(
    image_path: Path,
    mask_cfg: DictConfig,
    mask_transform,
    image_hw: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    metadata_dir = image_path.parent / mask_cfg.metadata_dir
    zero_mask = torch.zeros(image_hw, dtype=torch.float32)
    masks: Dict[str, torch.Tensor] = {}
    groups = list(mask_cfg.groups)
    pattern = mask_cfg.get("mask_pattern", "mask_{name}.png")
    stacked = []
    for name in groups:
        mask_path = metadata_dir / pattern.format(name=name)
        if mask_path.exists():
            with Image.open(mask_path) as mask_img:
                mask_tensor = mask_transform(mask_img.convert("L")).squeeze(0)
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = zero_mask.clone()
        masks[name] = mask_tensor
        stacked.append(mask_tensor)

    if stacked:
        overlap = torch.stack(stacked, dim=0).sum(dim=0) > 1.0
        if overlap.any():
            logger.debug(f"Overlap detected among masks for image {image_path}")

    if mask_cfg.get("include_background", True):
        background_name = mask_cfg.get("background_name", "Background")
        if stacked:
            union = torch.stack(stacked, dim=0).max(dim=0).values
            background = torch.clamp(1.0 - union, min=0.0, max=1.0)
        else:
            background = torch.ones_like(zero_mask)
        masks[background_name] = background

    return masks


def _load_images(
    image_dir: Path,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    mask_cfg: Optional[DictConfig] = None,
) -> Tuple[torch.Tensor, List[ImageRecord]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    transform = _build_transform(image_size, mean, std)
    image_hw = _resolve_image_hw(image_size)
    mask_transform = _build_mask_transform(image_hw) if mask_cfg is not None else None
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        [
            p
            for p in image_dir.rglob("*")
            if p.suffix.lower() in supported and "metadata" not in p.parts
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    tensors = []
    records: List[ImageRecord] = []
    for path in image_paths:
        with Image.open(path) as img:
            tensors.append(transform(img.convert("RGB")))
        mask_map = {}
        if mask_cfg is not None:
            mask_map = _load_masks_for_image(path, mask_cfg, mask_transform, image_hw)
        records.append(ImageRecord(path=path, masks=mask_map))
    stacked = torch.stack(tensors, dim=0)
    return stacked, records


def _chunk_tensor(tensor: torch.Tensor, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, tensor.size(0), batch_size):
        yield tensor[start : start + batch_size]


def _infer_patch_kernel(backbone) -> Tuple[int, int]:
    patch_size = getattr(backbone, "patch_size", None)
    if patch_size is None:
        raise AttributeError("Backbone does not expose 'patch_size'; cannot map patches.")
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
        return int(patch_size[0]), int(patch_size[1])
    raise ValueError(f"Unsupported patch_size format: {patch_size}")


def _mask_to_patch_vector(mask: torch.Tensor, kernel: Tuple[int, int], group: str) -> torch.Tensor:
    mask = mask.unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(mask, kernel_size=kernel, stride=kernel)
    # if group == "Background": # TODO: check if it make the results better
    #     pooled = 1 - (pooled < 1).float()
    # else:
    #     pooled = (pooled > 0).float()
    return pooled.view(-1)


def _prepare_patch_vectors(
    records: List[ImageRecord],
    kernel: Tuple[int, int],
) -> List[Dict[str, torch.Tensor]]:
    vectors = []
    for record in records:
        entry = {}
        for name, mask in record.masks.items():
            entry[name] = _mask_to_patch_vector(mask, kernel, name)
        vectors.append(entry)
    return vectors


class AttentionMapRecorder:
    def __init__(self, attn_module: nn.Module):
        self.attn_module = attn_module
        self.num_heads = self._extract_num_heads(attn_module)
        if self.num_heads is None:
            raise ValueError("Attention module missing a usable head count attribute.")
        self._records: List[torch.Tensor] = []
        self._handle = None
        self._orig_forward = None
        self._mode = None
        if isinstance(attn_module, nn.MultiheadAttention):
            self._mode = "multihead"
            self._patch_multihead_forward()
        else:
            drop_module = self._resolve_dropout(attn_module)
            if drop_module is None:
                raise ValueError("Could not locate attention dropout module for recording.")
            self._mode = "dropout"
            self._handle = drop_module.register_forward_hook(self._hook)

    @staticmethod
    def supports(module: nn.Module) -> bool:
        if "cross" in module.__class__.__name__.lower():
            return False
        if isinstance(module, nn.MultiheadAttention):
            return True
        num_heads = AttentionMapRecorder._extract_num_heads(module)
        if num_heads is None:
            return False
        return AttentionMapRecorder._resolve_dropout(module) is not None

    @staticmethod
    def _extract_num_heads(attn_module: nn.Module):
        for attr in ("num_heads", "num_attention_heads", "n_heads", "heads"):
            value = getattr(attn_module, attr, None)
            if value is not None:
                return int(value)
        return None

    @staticmethod
    def _resolve_dropout(attn_module: nn.Module):
        direct = getattr(attn_module, "attn_drop", None) or getattr(attn_module, "attn_dropout", None)
        if isinstance(direct, nn.Module):
            return direct
        for name, module in attn_module.named_modules():
            if isinstance(module, nn.Dropout) and "attn" in name:
                return module
        for module in attn_module.modules():
            if isinstance(module, nn.Dropout):
                return module
        return None

    def _hook(self, _module, _inputs, output):
        self._records.append(output.detach().cpu())

    def _patch_multihead_forward(self):
        orig_forward = self.attn_module.forward

        def patched_forward(module_self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            attn_output, attn_weights = orig_forward(*args, **kwargs)
            self._records.append(attn_weights.detach().cpu())
            return attn_output

        self.attn_module.forward = types.MethodType(patched_forward, self.attn_module)
        self._orig_forward = orig_forward

    def close(self):
        if self._mode == "dropout" and self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._mode == "multihead" and self._orig_forward is not None:
            self.attn_module.forward = self._orig_forward
            self._orig_forward = None

    def pop_records(self) -> List[torch.Tensor]:
        records = self._records
        self._records = []
        return records


def _reshape_attention(attn: torch.Tensor, num_heads: int) -> torch.Tensor:
    if attn.dim() == 4:
        return attn
    if attn.dim() != 3:
        raise ValueError(f"Unexpected attention tensor shape: {attn.shape}")
    total = attn.size(0)
    tgt_len = attn.size(1)
    src_len = attn.size(2)
    if total % num_heads != 0:
        raise ValueError(f"Cannot reshape tensor of shape {attn.shape} with {num_heads} heads")
    batch = total // num_heads
    return attn.view(batch, num_heads, tgt_len, src_len)


def _accumulate_group_attention(
    layer: int,
    attn: torch.Tensor,
    batch_patch_vectors: List[Dict[str, torch.Tensor]],
    patch_start: int,
    num_patches: int,
    query_groups: Sequence[str],
    target_groups: Sequence[str],
    aggregator: Optional[GroupAttentionAggregator],
):
    if aggregator is None:
        return
    seq_len = attn.size(-1)
    for sample_idx, sample_attn in enumerate(attn):
        vectors = batch_patch_vectors[sample_idx] # get each sample's patch vectors - patches with the objects are 1, others are 0
        token_masks: Dict[str, torch.Tensor] = {}
        for group, vector in vectors.items():
            if vector.numel() == 0:
                continue
            if vector.numel() != num_patches:
                raise ValueError(
                    f"Patch vector length {vector.numel()} does not match expected {num_patches}"
                )
            token_vec = sample_attn.new_zeros(seq_len) # initialize token vector with 0s
            token_vec[patch_start:] = vector.to(sample_attn.device) # set the patch vectors to the token vector
            token_masks[group] = token_vec # store the token vector for the group

        for query in query_groups:
            if query == "CLS":
                query_vec = sample_attn.new_zeros(seq_len)
                query_vec[0] = 1.0
                attn_query = sample_attn[0]
            else:
                q_mask = token_masks.get(query) # get the token vector for the query, e.g. Human, Tree, Truck
                if q_mask is None:
                    continue
                denom = q_mask.sum() # sum of the token vector, number of patches with the object
                if denom <= 0:
                    continue
                query_vec = q_mask / denom # normalize to find average attention score
                attn_query = torch.matmul(query_vec, sample_attn)

            for target in target_groups:
                t_mask = token_masks.get(target) # get the token vector for the target, e.g. Human, Tree, Truck
                if t_mask is None:
                    continue
                target_sum = t_mask.sum() # sum of the token vector, number of patches with the object
                if target_sum <= 0:
                    continue
                target_vec = t_mask / target_sum
                score = torch.dot(attn_query, target_vec)
                aggregator.update(layer, query, target, float(score))


def _summarize_attention(attn: torch.Tensor, num_heads: int, cls_index: int, num_spatial: int) -> Tuple[float, int, float, int]:
    attn = _reshape_attention(attn, num_heads)
    if attn.size(-1) <= cls_index:
        raise ValueError(f"CLS index {cls_index} out of bounds for attention map with width {attn.size(-1)}")
    if attn.size(-2) <= 1:
        raise ValueError("Attention map does not contain spatial tokens beyond CLS.")
    patch_to_cls = attn[:, :, -num_spatial:, cls_index]
    cls_to_cls = attn[:, :, cls_index, cls_index]
    return patch_to_cls.sum().item(), patch_to_cls.numel(), cls_to_cls.sum().item(), cls_to_cls.numel()


def _discover_attention_modules(backbone: nn.Module) -> List[Tuple[str, nn.Module]]:
    discovered: List[Tuple[str, nn.Module]] = []
    for name, module in backbone.named_modules():
        if AttentionMapRecorder.supports(module):
            if getattr(backbone, "dino_name", None) == "dinov2" or getattr(backbone, "dino_name", None) == "dinov3" or backbone.__class__.__name__ == "VGGT1B":
                if 'blocks' not in name:
                    continue
            discovered.append((name, module))
    return discovered


def _prepare_recorders(backbone, layers: Sequence[int]) -> Tuple[Dict[int, AttentionMapRecorder], List[str]]:
    attn_modules = _discover_attention_modules(backbone)
    if not attn_modules:
        raise RuntimeError("No attention modules detected in backbone.")
    if len(set(layers)) != len(layers):
        raise ValueError("Layer indices must be unique.")
    recorders: Dict[int, AttentionMapRecorder] = {}
    for layer in layers:
        if layer < 0 or layer >= len(attn_modules):
            raise ValueError(f"Requested layer {layer} outside valid range [0, {len(attn_modules)-1}]")
        name, module = attn_modules[layer]
        recorder = AttentionMapRecorder(module)
        recorders[layer] = recorder
        logger.debug(f"Attached recorder to attention layer {layer} ({name})")
    return recorders, [name for name, _ in attn_modules]


def _disable_recorders(recorders: Dict[int, AttentionMapRecorder]):
    for recorder in recorders.values():
        recorder.close()


def _analyze_model(
    model_cfg,
    images: torch.Tensor,
    image_records: List[ImageRecord],
    device: torch.device,
    batch_size: int,
    cls_index: int,
    mask_cfg: Optional[DictConfig],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, Dict[str, Dict[int, float]]]]:
    backbone = instantiate(model_cfg.backbone)
    backbone.eval()
    backbone.to(device)
    layers = list(model_cfg.layers)
    recorders, attn_names = _prepare_recorders(backbone, layers)
    model_name = getattr(model_cfg, "name", getattr(model_cfg.backbone, "_target_", "model"))
    logger.info(f"{model_name}: found {len(attn_names)} attention modules; recording indices {layers}")
    for layer in layers:
        logger.info(f"  layer {layer}: {attn_names[layer]}")
    totals_patch = defaultdict(float)
    counts_patch = defaultdict(int)
    totals_cls = defaultdict(float)
    counts_cls = defaultdict(int)
    image_h, image_w = images.shape[-2:] # get image height and width
    patch_kernel = _infer_patch_kernel(backbone) # get patch sizes
    if image_h % patch_kernel[0] != 0 or image_w % patch_kernel[1] != 0:
        raise ValueError("Image size is not divisible by patch size; cannot align masks.")
    patch_hw = (image_h // patch_kernel[0], image_w // patch_kernel[1]) # get patch height and width
    num_spatial = patch_hw[0] * patch_hw[1] # get number of spatial tokens - 196 patches for standard ViT-B/16

    patch_vectors: Optional[List[Dict[str, torch.Tensor]]] = None
    aggregator: Optional[GroupAttentionAggregator] = None
    query_groups: Sequence[str] = []
    target_groups: Sequence[str] = []
    if mask_cfg is not None:
        patch_vectors = _prepare_patch_vectors(image_records, patch_kernel)
        if patch_vectors and patch_vectors[0]:
            aggregator = GroupAttentionAggregator()
            object_groups = list(mask_cfg.groups)
            query_groups = ["CLS"] + object_groups
            target_groups = ["CLS"] + list(object_groups)
            if mask_cfg.get("include_background", True):
                background_name = mask_cfg.get("background_name", "Background")
                if background_name not in target_groups:
                    target_groups.append(background_name)
        else:
            logger.warning("Mask configuration enabled but no mask data found; skipping object analysis.")

    try:
        image_offset = 0
        with torch.no_grad():
            for batch in _chunk_tensor(images, batch_size):
                batch = batch.to(device)
                batch_size_eff = batch.size(0)
                batch_vectors = None
                if patch_vectors is not None:
                    batch_vectors = patch_vectors[image_offset : image_offset + batch_size_eff]
                image_offset += batch_size_eff
                _ = backbone(batch)
                for layer, recorder in recorders.items():
                    for attn in recorder.pop_records():
                        patch_sum, patch_n, cls_sum, cls_n = _summarize_attention(attn, recorder.num_heads, cls_index, num_spatial)
                        totals_patch[layer] += patch_sum
                        counts_patch[layer] += patch_n
                        totals_cls[layer] += cls_sum
                        counts_cls[layer] += cls_n
                        if aggregator is not None and batch_vectors is not None:
                            attn_mean = attn.mean(dim=1)
                            seq_len = attn_mean.size(-1)
                            patch_start = seq_len - num_spatial
                            if patch_start < 1:
                                raise ValueError("Sequence shorter than expected patch tokens.")
                            _accumulate_group_attention(
                                layer,
                                attn_mean,
                                batch_vectors,
                                patch_start,
                                num_spatial,
                                query_groups,
                                target_groups,
                                aggregator,
                            )
    finally:
        _disable_recorders(recorders)

    results = {}
    for layer in layers:
        if counts_patch[layer] == 0:
            raise RuntimeError(f"No attention data recorded for layer {layer}")
        results[layer] = totals_patch[layer] / counts_patch[layer]
    cls_self = {}
    for layer in layers:
        if counts_cls[layer] == 0:
            raise RuntimeError(f"No CLS self-attention data recorded for layer {layer}")
        cls_self[layer] = totals_cls[layer] / counts_cls[layer]
    object_results = aggregator.finalize() if aggregator is not None else {}
    return results, cls_self, object_results


def _plot_single(model_name: str, layer_to_score: Dict[int, float], output_dir: Path, title_suffix: str, filename_suffix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_layers = sorted(layer_to_score.keys())
    scores = [layer_to_score[layer] for layer in sorted_layers]
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_layers, scores, marker="o")
    plt.title(f"{model_name}: {title_suffix}")
    plt.xlabel("Layer index")
    plt.ylabel("Mean attention score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_{filename_suffix}.png", dpi=200)
    plt.close()


def _plot_overlay(results: List[Tuple[str, Dict[int, float]]], output_dir: Path, title: str, filename: str):
    if len(results) < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, layer_scores in results:
        sorted_layers = sorted(layer_scores.keys())
        scores = [layer_scores[layer] for layer in sorted_layers]
        plt.plot(sorted_layers, scores, marker="o", label=name)
    plt.title(title)
    plt.xlabel("Layer index")
    plt.ylabel("Mean attention score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def _write_table(
    patch_results: List[Tuple[str, Dict[int, float]]],
    cls_self_results: List[Tuple[str, Dict[int, float]]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cls_map = {name: layers for name, layers in cls_self_results}
    lines = ["model,layer,mean_patch_to_cls_attention,mean_cls_to_cls_attention"]
    for name, layer_scores in patch_results:
        cls_layers = cls_map.get(name, {})
        for layer, score in sorted(layer_scores.items()):
            cls_val = cls_layers.get(layer, float("nan"))
            lines.append(f"{name},{layer},{score:.6f},{cls_val:.6f}")
    table_path = output_dir / "cls_attention_scores.csv"
    table_path.write_text("\n".join(lines))
    logger.info(f"Wrote summary table to {table_path}")


def _plot_object_attention(model_name: str, query_map: Dict[str, Dict[str, Dict[int, float]]], output_dir: Path):
    if not query_map:
        return
    layers = sorted({layer for targets in query_map.values() for scores in targets.values() for layer in scores})
    if not layers:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for query, targets in query_map.items():
        for target, layer_scores in targets.items():
            y = [layer_scores.get(layer, np.nan) for layer in layers]
            if all(np.isnan(y)):
                continue
            label = f"{query}→{target}"
            plt.plot(layers, y, marker="o", label=label)
    plt.title(f"{model_name}: Query→Group attention")
    plt.xlabel("Layer index")
    plt.ylabel("Mean attention score")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_object_attention.png", dpi=200)
    plt.close()


def _write_object_attention_table(model_name: str, query_map: Dict[str, Dict[str, Dict[int, float]]], cls_self: Dict[int, float], output_dir: Path):
    if not query_map and not cls_self:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["model,layer,query,target,mean_attention"]
    for query, targets in query_map.items():
        for target, layer_scores in targets.items():
            for layer in sorted(layer_scores.keys()):
                lines.append(f"{model_name},{layer},{query},{target},{layer_scores[layer]:.6f}")
    if cls_self:
        for layer in sorted(cls_self.keys()):
            lines.append(f"{model_name},{layer},CLS,CLS,{cls_self[layer]:.6f}")
    table_path = output_dir / f"{model_name}_object_attention.csv"
    table_path.write_text("\n".join(lines))
    logger.info(f"Wrote object attention table to {table_path}")


@hydra.main(config_path="../configs", config_name="attention_stats", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    mean = _resolve_mean_std(cfg.image_mean)
    std = _resolve_mean_std(cfg.image_std, fallback=[0.229, 0.224, 0.225])
    image_dir = Path(to_absolute_path(cfg.image_dir))
    output_dir = Path(to_absolute_path(cfg.output_dir))
    mask_cfg = cfg.get("object_analysis", None)
    if mask_cfg is not None and not mask_cfg.get("enable", True):
        mask_cfg = None

    images, image_records = _load_images(image_dir, cfg.image_size, mean, std, mask_cfg=mask_cfg)
    logger.info(f"Loaded {len(image_records)} images from {image_dir}")

    model_results: List[Tuple[str, Dict[int, float]]] = []
    cls_self_results: List[Tuple[str, Dict[int, float]]] = []
    for model_cfg in cfg.models:
        model_name = model_cfg.name
        logger.info(f"Analyzing model '{model_name}' with layers {list(model_cfg.layers)}")
        layer_scores, cls_self_scores, object_map = _analyze_model(
            model_cfg,
            images,
            image_records,
            device,
            cfg.batch_size,
            cfg.cls_index,
            mask_cfg,
        )
        model_results.append((model_name, layer_scores))
        cls_self_results.append((model_name, cls_self_scores))
        _plot_single(model_name, layer_scores, output_dir, "Patch→CLS attention", "patch_to_cls")
        _plot_single(model_name, cls_self_scores, output_dir, "CLS→CLS attention", "cls_to_cls")
        if object_map:
            _plot_object_attention(model_name, object_map, output_dir)
            _write_object_attention_table(model_name, object_map, cls_self_scores, output_dir)

    _plot_overlay(model_results, output_dir, "Patch→CLS attention across models", "models_patch_to_cls.png")
    _plot_overlay(cls_self_results, output_dir, "CLS→CLS attention across models", "models_cls_to_cls.png")
    _write_table(model_results, cls_self_results, output_dir)


if __name__ == "__main__":
    main()
