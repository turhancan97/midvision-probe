from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    from matplotlib import cm as mpl_cm
except Exception:
    matplotlib = None
    mpl_cm = None

try:
    import cv2
except Exception:
    cv2 = None


def to_uint8_rgb(image_f32: np.ndarray) -> np.ndarray:
    image_f32 = np.clip(image_f32, 0.0, 1.0)
    return (image_f32 * 255.0).round().astype(np.uint8)


def _jet_colormap(norm_2d: np.ndarray) -> np.ndarray:
    x = np.clip(norm_2d.astype(np.float32), 0.0, 1.0)
    if mpl_cm is not None:
        if matplotlib is not None and hasattr(matplotlib, "colormaps"):
            return matplotlib.colormaps["jet"](x)[..., :3].astype(np.float32)
        return mpl_cm.get_cmap("jet")(x)[..., :3].astype(np.float32)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def select_attention_map(attn: torch.Tensor, map_name: str) -> torch.Tensor:
    # attn shape: [num_queries, num_patches]
    if map_name == "mean":
        return attn.mean(dim=0)
    if map_name == "max":
        return attn.max(dim=0).values
    if map_name == "min":
        return attn.min(dim=0).values
    if map_name == "std":
        return attn.std(dim=0, unbiased=False)
    if map_name in {"q1", "q2", "q3", "q4"}:
        idx = int(map_name[1]) - 1
        if idx >= attn.shape[0]:
            raise RuntimeError(f"Requested {map_name} but attention has only {attn.shape[0]} queries.")
        return attn[idx]
    if map_name == "best_q":
        n_patches = float(attn.shape[1])
        baseline = 1.0 / max(n_patches, 1.0)
        score = torch.clamp(attn - baseline, min=0.0).max(dim=1).values
        return attn[int(torch.argmax(score))]
    raise ValueError(f"Unknown map_name: {map_name}")


def _normalize_2d(x: np.ndarray) -> np.ndarray:
    min_v = float(x.min())
    max_v = float(x.max())
    denom = max(max_v - min_v, 1e-8)
    return (x - min_v) / denom


def _resize_attention(attn_2d: np.ndarray, out_hw: Tuple[int, int], mode: str = "nearest") -> np.ndarray:
    t = torch.from_numpy(attn_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=out_hw, mode=mode)
    return t[0, 0].cpu().numpy()


def _blur_attention(attn_2d: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    if kernel_size <= 1:
        return attn_2d
    if cv2 is not None:
        return cv2.blur(attn_2d.astype(np.float32), (kernel_size, kernel_size))
    t = torch.from_numpy(attn_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.avg_pool2d(t, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    if kernel_size % 2 == 0:
        t = t[:, :, :-1, :-1]
    return t[0, 0].cpu().numpy()


def _enhance_attention(
    attn_2d: np.ndarray,
    n_patches: int,
    clip_percentile: float = 98.0,
    gamma: float = 0.65,
) -> np.ndarray:
    x = attn_2d.astype(np.float32)
    baseline = 1.0 / float(max(n_patches, 1))
    x = np.maximum(x - baseline, 0.0)

    p = float(np.percentile(x, np.clip(clip_percentile, 0.0, 100.0)))
    if p > 1e-12:
        x = np.clip(x / p, 0.0, 1.0)
    else:
        x = np.zeros_like(x, dtype=np.float32)

    x = np.power(x, max(float(gamma), 1e-3))
    return x


def overlay_attention(
    image_rgb_u8: np.ndarray,
    attn_2d: np.ndarray,
    alpha: float,
    n_patches: int,
    sharpen: bool = True,
    clip_percentile: float = 98.0,
    gamma: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb_u8.shape[:2]
    # Notebook-style: nearest upsample -> blur -> normalize -> jet.
    attn_resized = _resize_attention(attn_2d, out_hw=(h, w), mode="nearest")
    attn_blurred = _blur_attention(attn_resized, kernel_size=8)
    if sharpen:
        attn_norm = _enhance_attention(
            attn_blurred,
            n_patches=n_patches,
            clip_percentile=clip_percentile,
            gamma=gamma,
        )
    else:
        attn_norm = _normalize_2d(attn_blurred)
    heatmap = _jet_colormap(attn_norm)

    image_f32 = image_rgb_u8.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * image_f32 + alpha * heatmap

    return to_uint8_rgb(blended), to_uint8_rgb(heatmap)

