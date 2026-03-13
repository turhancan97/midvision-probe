from __future__ import annotations

import gc
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image
from timm.data import resolve_model_data_config

from src.attention import overlay_attention, select_attention_map, to_uint8_rgb
from src.geometry import _load_positions, _matching_json_for_image, classify_relative_direction
from src.gradio_compat import patch_gradio_schema_parser
from src.probes import EfficientProbing
from src.ui_components import (
    build_copy_status,
    build_footer_html,
    build_help_html,
    build_hero_html,
    build_label_card,
    build_legend_html,
    build_model_warning,
    build_prediction_card,
    build_prediction_placeholder,
    build_run_info_card,
    build_stage_indicator,
    build_triplet_card,
)
from src.ui_state import PERSISTENT_CONTROL_IDS, build_persistence_script, serialize_settings
from src.ui_theme import APP_CSS, build_head_html, path_to_data_uri

APP_VERSION = "1.1.0-ui-refresh"
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
MANIFEST_PATH = APP_DIR / "artifacts" / "manifest.yaml"

LOGO_SRC = path_to_data_uri(ASSETS_DIR / "logo.png")
FAVICON_PATH = ASSETS_DIR / "favicon.png"

DEFAULT_PERSPECTIVE = "camera"
DEFAULT_SOURCE_MODE = "Sample gallery"
REAL_WORLD_SOURCE_MODE = "Real-world gallery"
UPLOAD_SOURCE_MODE = "Upload image"
GALLERY_SOURCE_MODES = (DEFAULT_SOURCE_MODE, REAL_WORLD_SOURCE_MODE)
REAL_WORLD_SAMPLES_DIR = APP_DIR / "artifacts" / "real_world_samples"
REAL_WORLD_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_MAP_NAME = "mean"
DEFAULT_ALPHA = 0.25
DEFAULT_COMPARE_ENABLED = False
DEFAULT_COMPARE_MAP = "q1"
DEFAULT_VIEW_MODE = "Split"
DEFAULT_SHARPEN = True
DEFAULT_CLIP_PERCENTILE = 100.0
DEFAULT_GAMMA = 1.5

patch_gradio_schema_parser()


@dataclass
class LoadedBackbone:
    key: str
    model: nn.Module
    transform: Any
    mean: np.ndarray
    std: np.ndarray


@dataclass
class RuntimeState:
    device: torch.device
    amp_dtype: torch.dtype
    loaded_backbone: Optional[LoadedBackbone]
    head_cache: Dict[Tuple[str, str, str, int], EfficientProbing]


@dataclass
class ModelEntry:
    key: str
    hf_model_id: str
    label: str
    available: bool
    reason: str
    experimental: bool


def load_manifest() -> Dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


MANIFEST = load_manifest()
CLASSES: List[str] = list(MANIFEST["classes"])
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}


def _manifest_rel_path(rel_path: str) -> Path:
    return APP_DIR / rel_path


def _triplet_map() -> Dict[str, Dict[str, Any]]:
    return {entry["id"]: entry for entry in MANIFEST["triplets"]}


TRIPLETS_BY_ID = _triplet_map()


def _all_triplet_ids() -> List[str]:
    return [entry["id"] for entry in MANIFEST["triplets"]]


def _checkpoint_exists(model_key: str, perspective: str, triplet_id: str) -> bool:
    rel = (
        MANIFEST.get("models", {})
        .get(model_key, {})
        .get("checkpoints", {})
        .get(perspective, {})
        .get(triplet_id)
    )
    if not rel:
        return False
    return _manifest_rel_path(rel).exists()


def model_has_all_checkpoints(model_key: str) -> bool:
    for perspective in MANIFEST["perspectives"].keys():
        for triplet_id in _all_triplet_ids():
            if not _checkpoint_exists(model_key, perspective, triplet_id):
                return False
    return True


def validate_manifest_or_raise() -> None:
    if len(CLASSES) != 4:
        raise RuntimeError("Manifest must define exactly 4 classes: Front/Back/Left/Right.")

    for required in ("Front", "Back", "Left", "Right"):
        if required not in CLASS_TO_INDEX:
            raise RuntimeError(f"Missing required class in manifest: {required}")

    for triplet in MANIFEST["triplets"]:
        sample_dir = _manifest_rel_path(triplet["sample_dir"])
        if not sample_dir.exists():
            raise RuntimeError(f"Missing sample_dir: {sample_dir}")
        for image_name in triplet["sample_images"]:
            image_path = sample_dir / image_name
            if not image_path.exists():
                raise RuntimeError(f"Missing sample image: {image_path}")
            json_path = _matching_json_for_image(image_path)
            if json_path is None or not json_path.exists():
                raise RuntimeError(f"Missing JSON metadata for sample image: {image_path}")

    for model_key, cfg in MANIFEST["models"].items():
        if bool(cfg.get("experimental", False)):
            continue
        if not model_has_all_checkpoints(model_key):
            raise RuntimeError(
                f"Missing required checkpoints for non-experimental model '{model_key}'. "
                "Populate artifacts/checkpoints before startup."
            )


validate_manifest_or_raise()


def _init_runtime_state() -> RuntimeState:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    return RuntimeState(device=device, amp_dtype=amp_dtype, loaded_backbone=None, head_cache={})


RUNTIME = _init_runtime_state()


def _clear_loaded_backbone() -> None:
    if RUNTIME.loaded_backbone is not None:
        RUNTIME.loaded_backbone = None
    RUNTIME.head_cache.clear()
    gc.collect()
    if RUNTIME.device.type == "cuda":
        torch.cuda.empty_cache()


def _runtime_model_status(model_key: str) -> Tuple[bool, str]:
    cfg = MANIFEST["models"][model_key]
    is_experimental = bool(cfg.get("experimental", False))
    has_ckpt = model_has_all_checkpoints(model_key)

    if not has_ckpt:
        return False, "Packaged checkpoints are incomplete for this backbone."

    if is_experimental and RUNTIME.device.type != "cuda":
        return False, "Experimental backbones require a CUDA runtime."

    if is_experimental:
        return True, "Experimental backbone is enabled for this runtime."

    return True, "Model is ready."


def _model_entry(model_key: str) -> ModelEntry:
    cfg = MANIFEST["models"][model_key]
    hf_model_id = str(cfg["hf_model_id"])
    is_experimental = bool(cfg.get("experimental", False))
    available, reason = _runtime_model_status(model_key)

    label = hf_model_id
    if is_experimental:
        label += " [experimental]"
    if not available:
        label += " [unavailable]"

    return ModelEntry(
        key=model_key,
        hf_model_id=hf_model_id,
        label=label,
        available=available,
        reason=reason,
        experimental=is_experimental,
    )


def _model_entries(show_experimental: bool) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for model_key, cfg in MANIFEST["models"].items():
        is_experimental = bool(cfg.get("experimental", False))
        if is_experimental and not show_experimental:
            continue
        entries.append(_model_entry(model_key))
    return entries


def _model_choice_pairs(show_experimental: bool) -> List[Tuple[str, str]]:
    return [(entry.label, entry.key) for entry in _model_entries(show_experimental)]


def _initial_model_key() -> str:
    non_exp = [k for k, cfg in MANIFEST["models"].items() if not bool(cfg.get("experimental", False))]
    for key in non_exp:
        entry = _model_entry(key)
        if entry.available:
            return key
    if not non_exp:
        raise RuntimeError("No model entries available in manifest.")
    return non_exp[0]


def _resolve_model_value(show_experimental: bool, current_model: str) -> str:
    entries = _model_entries(show_experimental)
    keys = [entry.key for entry in entries]
    if not keys:
        return _initial_model_key()
    if current_model in keys:
        return current_model
    for entry in entries:
        if entry.available:
            return entry.key
    return keys[0]


def _get_backbone(model_key: str) -> LoadedBackbone:
    if RUNTIME.loaded_backbone is not None and RUNTIME.loaded_backbone.key == model_key:
        return RUNTIME.loaded_backbone

    _clear_loaded_backbone()

    cfg = MANIFEST["models"][model_key]
    hf_id = str(cfg["hf_model_id"])
    timm_model_name = hf_id if hf_id.startswith("hf_hub:") else f"hf_hub:{hf_id}"

    model = timm.create_model(timm_model_name, pretrained=True)
    model.eval().to(RUNTIME.device)

    data_cfg = resolve_model_data_config(model)
    mean = np.array(data_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
    std = np.array(data_cfg.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)

    transform = T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )

    loaded = LoadedBackbone(
        key=model_key,
        model=model,
        transform=transform,
        mean=mean,
        std=std,
    )
    RUNTIME.loaded_backbone = loaded
    return loaded


def _extract_patch_tokens(model: nn.Module, tensor_image: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Extract probe tokens aligned with training wrapper: [CLS] + spatial tail."""

    patch_size = getattr(model.patch_embed, "patch_size", 16)
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
            raise RuntimeError(f"Unsupported patch_size format: {patch_size}")
        patch_size = int(patch_size[0])
    else:
        patch_size = int(patch_size)

    h, w = tensor_image.shape[-2:]
    grid_h, grid_w = h // patch_size, w // patch_size
    n_patches = int(grid_h * grid_w)
    if n_patches <= 0:
        raise RuntimeError("Could not infer spatial patch count from input size.")

    # Mirror evals.models.dinov3_timm.DINOV3TIMM forward path.
    tokens = model.patch_embed(tensor_image)
    rope = None
    attn_mask = None
    if hasattr(model, "_pos_embed"):
        tokens = model._pos_embed(tokens)
        if isinstance(tokens, tuple):
            if len(tokens) >= 1:
                rope = tokens[1] if len(tokens) >= 2 else None
                attn_mask = tokens[2] if len(tokens) >= 3 else None
                tokens = tokens[0]
            else:
                raise RuntimeError("Unexpected empty tuple from _pos_embed.")
    else:
        if hasattr(model, "cls_token") and model.cls_token is not None:
            cls_tok = model.cls_token.expand(tensor_image.shape[0], -1, -1)
            tokens = torch.cat((cls_tok, tokens), dim=1)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            tokens = tokens + model.pos_embed

    patch_drop = getattr(model, "patch_drop", None)
    if patch_drop is not None:
        tokens = patch_drop(tokens)
    norm_pre = getattr(model, "norm_pre", None)
    if norm_pre is not None:
        tokens = norm_pre(tokens)

    for blk in model.blocks:
        if rope is not None or attn_mask is not None:
            try:
                tokens = blk(tokens, rope=rope, attn_mask=attn_mask)
            except TypeError:
                tokens = blk(tokens)
        else:
            tokens = blk(tokens)
        if isinstance(tokens, tuple):
            if len(tokens) == 0:
                raise RuntimeError("Transformer block returned an empty tuple.")
            tokens = tokens[0]

    if tokens.dim() != 3:
        raise RuntimeError(f"Expected token tensor with rank 3, got shape {tuple(tokens.shape)}")
    if tokens.shape[1] < n_patches:
        raise RuntimeError(f"Token sequence too short: seq={tokens.shape[1]}, patches={n_patches}.")

    spatial = tokens[:, -n_patches:, :]
    if tokens.shape[1] > n_patches:
        tokens = torch.cat([tokens[:, :1, :], spatial], dim=1)
    else:
        tokens = spatial

    side = int(round(math.sqrt(n_patches)))
    if side * side != n_patches:
        raise RuntimeError(
            f"Patch tokens are not square: n_patches={n_patches}. "
            "This app expects square patch grids from ViT patch tokens."
        )

    return tokens, side


def _build_head(feat_dim: int, checkpoint_path: Path) -> EfficientProbing:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    use_layernorm = "norm.weight" in state_dict

    head = EfficientProbing(
        feat_dim=feat_dim,
        num_classes=len(CLASSES),
        num_queries=4,
        d_out=8,
        use_layernorm=use_layernorm,
        dropout_rate=0.0,
    )
    head.load_state_dict(state_dict, strict=True)
    head.eval().to(RUNTIME.device)
    return head


def _get_head(model_key: str, perspective: str, triplet_id: str, feat_dim: int) -> EfficientProbing:
    cache_key = (model_key, perspective, triplet_id, feat_dim)
    if cache_key in RUNTIME.head_cache:
        return RUNTIME.head_cache[cache_key]

    rel = MANIFEST["models"][model_key]["checkpoints"][perspective][triplet_id]
    ckpt_path = _manifest_rel_path(rel)
    if not ckpt_path.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt_path}")

    head = _build_head(feat_dim=feat_dim, checkpoint_path=ckpt_path)
    RUNTIME.head_cache[cache_key] = head
    return head


def _sample_image_path(triplet_id: str, image_name: str) -> Path:
    cfg = TRIPLETS_BY_ID[triplet_id]
    return _manifest_rel_path(cfg["sample_dir"]) / image_name


def _real_world_dir(triplet_id: str) -> Path:
    return REAL_WORLD_SAMPLES_DIR / triplet_id


def _real_world_image_names(triplet_id: str) -> List[str]:
    rw_dir = _real_world_dir(triplet_id)
    if not rw_dir.exists():
        return []
    names = [
        path.name
        for path in rw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in REAL_WORLD_IMAGE_EXTENSIONS
    ]
    return sorted(names)


def _source_image_names(triplet_id: str, source_mode: str) -> List[str]:
    if source_mode == DEFAULT_SOURCE_MODE:
        return list(TRIPLETS_BY_ID[triplet_id]["sample_images"])
    if source_mode == REAL_WORLD_SOURCE_MODE:
        return _real_world_image_names(triplet_id)
    return []


def _source_mode_available(source_mode: str, triplet_id: str) -> bool:
    if source_mode != REAL_WORLD_SOURCE_MODE:
        return True
    return len(_real_world_image_names(triplet_id)) > 0


def _source_mode_warning_html(source_mode: str, triplet_id: str) -> str:
    if source_mode != REAL_WORLD_SOURCE_MODE:
        return ""
    if _source_mode_available(source_mode, triplet_id):
        return ""
    expected = (_real_world_dir(triplet_id)).as_posix()
    return (
        '<div class="banner banner-error">'
        "<strong>Real-world samples:</strong> none found for this triplet.<br />"
        f'<span class="banner-sub">Expected images in: {expected}</span>'
        "</div>"
    )


def _is_run_interactive(model_key: str, source_mode: str, triplet_id: str) -> bool:
    return _model_entry(model_key).available and _source_mode_available(source_mode, triplet_id)


def _resolve_valid_sample_image_path(triplet_id: str, image_name: str) -> Path:
    cfg = TRIPLETS_BY_ID[triplet_id]
    candidates: List[str] = []
    if image_name:
        candidates.append(image_name)
    for name in cfg["sample_images"]:
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        path = _sample_image_path(triplet_id, name)
        if path.exists():
            return path

    sample_dir = _manifest_rel_path(cfg["sample_dir"])
    any_images = sorted(sample_dir.glob("img_*.jpg"))
    if any_images:
        return any_images[0]

    raise gr.Error(f"No sample images found for triplet '{triplet_id}'.")


def _resolve_valid_real_world_image_path(triplet_id: str, image_name: str) -> Path:
    names = _real_world_image_names(triplet_id)
    if not names:
        raise gr.Error(
            f"No real-world images found for triplet '{triplet_id}' in {_real_world_dir(triplet_id)}."
        )
    if image_name and image_name in names:
        return _real_world_dir(triplet_id) / image_name
    return _real_world_dir(triplet_id) / names[0]


def _resolve_valid_gallery_image_path(triplet_id: str, source_mode: str, image_name: str) -> Path:
    if source_mode == DEFAULT_SOURCE_MODE:
        return _resolve_valid_sample_image_path(triplet_id, image_name)
    if source_mode == REAL_WORLD_SOURCE_MODE:
        return _resolve_valid_real_world_image_path(triplet_id, image_name)
    raise gr.Error(f"Unsupported gallery source mode: {source_mode}")


def _gallery_dropdown_update(
    triplet_id: str,
    source_mode: str,
    current_name: str,
    visible: bool,
) -> gr.Dropdown:
    names = _source_image_names(triplet_id, source_mode)
    value: Optional[str] = current_name if current_name in names else (names[0] if names else None)
    return gr.Dropdown(choices=names, value=value, visible=visible)


def _gallery_preview(triplet_id: str, source_mode: str, image_name: str) -> Optional[Image.Image]:
    if source_mode not in GALLERY_SOURCE_MODES:
        return None
    names = _source_image_names(triplet_id, source_mode)
    if not names:
        return None
    image_path = _resolve_valid_gallery_image_path(triplet_id, source_mode, image_name)
    return Image.open(image_path).convert("RGB")


def _build_triplet_html(triplet_id: str) -> str:
    return build_triplet_card(TRIPLETS_BY_ID[triplet_id])


def _compute_gt_label(image_path: Path, triplet_id: str, perspective: str) -> str:
    cfg = TRIPLETS_BY_ID[triplet_id]
    json_path = _matching_json_for_image(image_path)
    if json_path is None:
        return "unknown"

    positions = _load_positions(
        json_path,
        reference_label=cfg["reference_label"],
        target_label=cfg["target_label"],
        human_label=cfg["human_label"],
    )
    if positions is None:
        return "unknown"

    observer_key = "camera" if perspective == "camera" else "human"
    observer = positions.get(observer_key)
    if observer is None:
        return "unknown"

    label = classify_relative_direction(
        observer,
        positions["reference"],
        positions["target"],
        amb_deg=int(MANIFEST["perspectives"][perspective]["ambiguity_degrees"]),
        front_deg=45,
        back_deg=135,
    )
    if label in CLASS_TO_INDEX:
        return label
    return "Ambiguous"


def _to_display_image(normalized_tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    img = normalized_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return to_uint8_rgb(img)


def _prepare_input_image(
    perspective: str,
    triplet_id: str,
    source_mode: str,
    sample_image_name: str,
    uploaded_image: Optional[Image.Image],
) -> Tuple[Image.Image, str]:
    if source_mode in GALLERY_SOURCE_MODES:
        image_path = _resolve_valid_gallery_image_path(triplet_id, source_mode, sample_image_name)
        pil_image = Image.open(image_path).convert("RGB")
        gt_label = (
            _compute_gt_label(image_path, triplet_id, perspective)
            if source_mode == DEFAULT_SOURCE_MODE
            else "N/A"
        )
        return pil_image, gt_label

    if uploaded_image is None:
        raise gr.Error("Please upload an image first.")
    return uploaded_image.convert("RGB"), "N/A"


def _build_run_info_entries(
    model_entry: ModelEntry,
    perspective: str,
    triplet_id: str,
    map_name: str,
    sharpen: bool,
    clip_percentile: float,
    gamma: float,
    source_mode: str,
) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = [
        ("Backbone", model_entry.hf_model_id),
        ("Perspective", perspective),
        ("Triplet", triplet_id),
        ("Source", source_mode),
        ("Attention", map_name),
        ("Sharpen", str(bool(sharpen))),
    ]
    if sharpen:
        entries.append(("Clip %", f"{float(clip_percentile):.1f}"))
        entries.append(("Gamma", f"{float(gamma):.2f}"))
    return entries


def _infer_core(
    perspective: str,
    model_key: str,
    triplet_id: str,
    source_mode: str,
    sample_image_name: str,
    uploaded_image: Optional[Image.Image],
    map_name: str,
    alpha: float,
    compare_enabled: bool,
    compare_map_name: str,
    sharpen: bool,
    clip_percentile: float,
    gamma: float,
) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str, str, str, str]:
    if model_key not in MANIFEST["models"]:
        raise gr.Error("Invalid backbone selection.")
    if triplet_id not in TRIPLETS_BY_ID:
        raise gr.Error("Invalid triplet selection.")
    if perspective not in MANIFEST["perspectives"]:
        raise gr.Error("Invalid perspective selection.")

    model_entry = _model_entry(model_key)
    if not model_entry.available:
        raise gr.Error(model_entry.reason)

    pil_image, gt_label = _prepare_input_image(
        perspective=perspective,
        triplet_id=triplet_id,
        source_mode=source_mode,
        sample_image_name=sample_image_name,
        uploaded_image=uploaded_image,
    )

    backbone = _get_backbone(model_key)
    x = backbone.transform(pil_image).unsqueeze(0).to(RUNTIME.device)

    with torch.inference_mode():
        with torch.autocast(
            device_type=RUNTIME.device.type,
            dtype=RUNTIME.amp_dtype,
            enabled=(RUNTIME.device.type == "cuda"),
        ):
            patch_tokens, grid_side = _extract_patch_tokens(backbone.model, x)
            feat_dim = int(patch_tokens.shape[-1])
            expected = MANIFEST["models"][model_key].get("expected_feat_dim")
            if expected is not None and int(expected) != feat_dim:
                raise gr.Error(
                    f"Feature dim mismatch for {model_key}: expected {expected}, got {feat_dim}."
                )
            head = _get_head(model_key, perspective, triplet_id, feat_dim)
            logits = head(patch_tokens)

    if head.attention_map is None:
        raise gr.Error("Probe did not expose attention_map.")

    attn = head.attention_map[0].detach().float().cpu()
    selected = select_attention_map(attn, map_name=map_name)
    n_patches = grid_side * grid_side
    selected_np = selected.numpy().reshape(-1)
    if selected_np.size < n_patches:
        raise gr.Error(
            f"Attention map length {selected_np.size} is smaller than patch grid size {n_patches}."
        )
    selected_2d = selected_np[-n_patches:].reshape(grid_side, grid_side)

    display_image = _to_display_image(x[0], mean=backbone.mean, std=backbone.std)
    overlay, heatmap = overlay_attention(
        display_image,
        selected_2d,
        alpha=float(alpha),
        n_patches=n_patches,
        sharpen=bool(sharpen),
        clip_percentile=float(clip_percentile),
        gamma=float(gamma),
    )

    compare_primary: Optional[np.ndarray] = None
    compare_secondary: Optional[np.ndarray] = None
    if compare_enabled:
        compare_primary = overlay
        compare_selected = select_attention_map(attn, map_name=compare_map_name)
        compare_np = compare_selected.numpy().reshape(-1)
        if compare_np.size < n_patches:
            raise gr.Error(
                f"Compare attention map length {compare_np.size} is smaller than patch grid size {n_patches}."
            )
        compare_2d = compare_np[-n_patches:].reshape(grid_side, grid_side)
        compare_secondary, _ = overlay_attention(
            display_image,
            compare_2d,
            alpha=float(alpha),
            n_patches=n_patches,
            sharpen=bool(sharpen),
            clip_percentile=float(clip_percentile),
            gamma=float(gamma),
        )

    pred_html, pred_label, pred_prob = build_prediction_card(CLASSES, logits)
    labels_html = build_label_card(gt_label, pred_label, pred_prob, source_mode=source_mode)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        heatmap_path = tmp.name
    Image.fromarray(heatmap).save(heatmap_path)

    run_info_entries = _build_run_info_entries(
        model_entry=model_entry,
        perspective=perspective,
        triplet_id=triplet_id,
        map_name=map_name,
        sharpen=bool(sharpen),
        clip_percentile=float(clip_percentile),
        gamma=float(gamma),
        source_mode=source_mode,
    )
    run_info_html = build_run_info_card(run_info_entries)

    return (
        pred_html,
        labels_html,
        overlay,
        heatmap,
        overlay,
        heatmap,
        compare_primary,
        compare_secondary,
        heatmap_path,
        run_info_html,
        build_stage_indicator("Inference complete.", state="done"),
        _build_triplet_html(triplet_id),
    )


def _stage_only(message: str) -> Tuple[Any, ...]:
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        build_stage_indicator(message, state="running"),
        gr.update(),
    )


def _infer_stream(
    perspective: str,
    model_key: str,
    triplet_id: str,
    source_mode: str,
    sample_image_name: str,
    uploaded_image: Optional[Image.Image],
    map_name: str,
    alpha: float,
    compare_enabled: bool,
    compare_map_name: str,
    sharpen: bool,
    clip_percentile: float,
    gamma: float,
) -> Generator[Tuple[Any, ...], None, None]:
    yield _stage_only("Loading backbone...")
    yield _stage_only("Loading probe...")
    yield _stage_only("Running inference...")
    try:
        yield _infer_core(
            perspective=perspective,
            model_key=model_key,
            triplet_id=triplet_id,
            source_mode=source_mode,
            sample_image_name=sample_image_name,
            uploaded_image=uploaded_image,
            map_name=map_name,
            alpha=alpha,
            compare_enabled=compare_enabled,
            compare_map_name=compare_map_name,
            sharpen=sharpen,
            clip_percentile=clip_percentile,
            gamma=gamma,
        )
    except Exception as exc:
        yield (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            build_stage_indicator(f"Inference failed: {exc}", state="error"),
            gr.update(),
        )
        raise


def _on_gallery_image_change(triplet_id: str, source_mode: str, image_name: str) -> gr.Image:
    use_gallery = source_mode in GALLERY_SOURCE_MODES
    preview = _gallery_preview(triplet_id, source_mode, image_name) if use_gallery else None
    return gr.Image(value=preview, visible=use_gallery)


def _on_source_mode_change(
    source_mode: str,
    triplet_id: str,
    current_name: str,
    model_key: str,
) -> Tuple[gr.Dropdown, gr.Image, gr.Image, str, gr.Button]:
    use_gallery = source_mode in GALLERY_SOURCE_MODES
    dropdown = _gallery_dropdown_update(triplet_id, source_mode, current_name, visible=use_gallery)
    preview = gr.Image(value=_gallery_preview(triplet_id, source_mode, current_name), visible=use_gallery)
    uploaded = gr.Image(visible=(source_mode == UPLOAD_SOURCE_MODE))
    source_warning = _source_mode_warning_html(source_mode, triplet_id)
    run_btn = gr.Button(interactive=_is_run_interactive(model_key, source_mode, triplet_id))
    return dropdown, preview, uploaded, source_warning, run_btn


def _on_triplet_change(
    triplet_id: str,
    source_mode: str,
    current_name: str,
    model_key: str,
) -> Tuple[gr.Dropdown, gr.Image, str, str, gr.Button]:
    use_gallery = source_mode in GALLERY_SOURCE_MODES
    dropdown = _gallery_dropdown_update(triplet_id, source_mode, current_name, visible=use_gallery)
    preview = gr.Image(value=_gallery_preview(triplet_id, source_mode, current_name), visible=use_gallery)
    source_warning = _source_mode_warning_html(source_mode, triplet_id)
    run_btn = gr.Button(interactive=_is_run_interactive(model_key, source_mode, triplet_id))
    return dropdown, preview, _build_triplet_html(triplet_id), source_warning, run_btn


def _model_warning_and_button_state(
    model_key: str,
    source_mode: str,
    triplet_id: str,
) -> Tuple[str, str, gr.Button]:
    entry = _model_entry(model_key)
    warning_html = build_model_warning(
        model_label=entry.hf_model_id,
        is_experimental=entry.experimental,
        available=entry.available,
        reason=entry.reason,
    )
    source_warning = _source_mode_warning_html(source_mode, triplet_id)
    run_interactive = _is_run_interactive(model_key, source_mode, triplet_id)
    return warning_html, source_warning, gr.Button(interactive=run_interactive)


def _on_backbone_change(model_key: str, source_mode: str, triplet_id: str) -> Tuple[str, str, gr.Button]:
    return _model_warning_and_button_state(model_key, source_mode, triplet_id)


def _on_show_experimental_change(
    show_experimental: bool,
    current_model: str,
    source_mode: str,
    triplet_id: str,
) -> Tuple[gr.Dropdown, str, str, gr.Button]:
    value = _resolve_model_value(show_experimental, current_model)
    dropdown = gr.Dropdown(choices=_model_choice_pairs(show_experimental), value=value)
    warning_html, source_warning, run_btn = _model_warning_and_button_state(value, source_mode, triplet_id)
    return dropdown, warning_html, source_warning, run_btn


def _view_mode_visibility(view_mode: str) -> Tuple[gr.Group, gr.Group]:
    split_mode = view_mode == "Split"
    return gr.Group(visible=split_mode), gr.Group(visible=not split_mode)


def _compare_visibility(compare_enabled: bool) -> Tuple[gr.Dropdown, gr.Group]:
    return gr.Dropdown(visible=compare_enabled), gr.Group(visible=compare_enabled)


def _apply_recommended_preset() -> Tuple[str, float, bool, str, str, bool, float, float, str]:
    return (
        "mean",
        0.25,
        False,
        "q1",
        "Split",
        False,
        98.0,
        0.65,
        build_copy_status("Recommended preset loaded.", ok=True),
    )


def _serialize_current_settings(
    show_experimental: bool,
    perspective: str,
    model_key: str,
    triplet_id: str,
    source_mode: str,
    sample_name: str,
    map_name: str,
    alpha: float,
    compare_enabled: bool,
    compare_map: str,
    view_mode: str,
    sharpen: bool,
    clip_percentile: float,
    gamma: float,
) -> str:
    payload = {
        "show_experimental": bool(show_experimental),
        "perspective": perspective,
        "model_key": model_key,
        "triplet_id": triplet_id,
        "source_mode": source_mode,
        "sample_name": sample_name,
        "map_name": map_name,
        "alpha": float(alpha),
        "compare_enabled": bool(compare_enabled),
        "compare_map": compare_map,
        "view_mode": view_mode,
        "sharpen": bool(sharpen),
        "clip_percentile": float(clip_percentile),
        "gamma": float(gamma),
    }
    return serialize_settings(payload)


def _reset_controls(show_experimental: bool) -> Tuple[Any, ...]:
    initial_triplet = _all_triplet_ids()[0]
    initial_sample = _source_image_names(initial_triplet, DEFAULT_SOURCE_MODE)[0]
    initial_model = _resolve_model_value(show_experimental=show_experimental, current_model=_initial_model_key())

    backbone_update = gr.Dropdown(choices=_model_choice_pairs(show_experimental), value=initial_model)
    warning_html, source_warning, run_btn_update = _model_warning_and_button_state(
        initial_model,
        DEFAULT_SOURCE_MODE,
        initial_triplet,
    )

    return (
        DEFAULT_PERSPECTIVE,
        backbone_update,
        initial_triplet,
        DEFAULT_SOURCE_MODE,
        _gallery_dropdown_update(initial_triplet, DEFAULT_SOURCE_MODE, initial_sample, visible=True),
        None,
        DEFAULT_MAP_NAME,
        DEFAULT_ALPHA,
        DEFAULT_COMPARE_ENABLED,
        gr.Dropdown(
            choices=["mean", "max", "min", "std", "best_q", "q1", "q2", "q3", "q4"],
            value=DEFAULT_COMPARE_MAP,
            visible=False,
        ),
        DEFAULT_VIEW_MODE,
        DEFAULT_SHARPEN,
        DEFAULT_CLIP_PERCENTILE,
        DEFAULT_GAMMA,
        _gallery_preview(initial_triplet, DEFAULT_SOURCE_MODE, initial_sample),
        _build_triplet_html(initial_triplet),
        warning_html,
        source_warning,
        run_btn_update,
        build_copy_status("Controls reset to defaults.", ok=True),
        build_stage_indicator("Idle. Configure controls and run inference.", state="idle"),
        gr.Group(visible=True),
        gr.Group(visible=False),
        gr.Group(visible=False),
    )


def _build_demo() -> gr.Blocks:
    initial_triplet = _all_triplet_ids()[0]
    initial_sample = _source_image_names(initial_triplet, DEFAULT_SOURCE_MODE)[0]
    initial_model = _initial_model_key()
    initial_entry = _model_entry(initial_model)
    initial_warning = build_model_warning(
        model_label=initial_entry.hf_model_id,
        is_experimental=initial_entry.experimental,
        available=initial_entry.available,
        reason=initial_entry.reason,
    )
    initial_source_warning = _source_mode_warning_html(DEFAULT_SOURCE_MODE, initial_triplet)

    head_html = build_head_html(
        favicon_path=FAVICON_PATH,
        extra_script=build_persistence_script(PERSISTENT_CONTROL_IDS),
    )

    with gr.Blocks(title="SpaRRTA", css=APP_CSS, head=head_html) as demo:
        gr.HTML(build_hero_html(LOGO_SRC))

        with gr.Row(elem_id="layout-main"):
            with gr.Column(elem_id="controls-panel"):
                with gr.Group(elem_classes=["panel-card"]):
                    with gr.Accordion("Model", open=True):
                        show_experimental = gr.Checkbox(
                            label="Show experimental backbones",
                            value=False,
                            elem_id="ctl-show-experimental",
                        )
                        perspective = gr.Dropdown(
                            label="Perspective",
                            choices=list(MANIFEST["perspectives"].keys()),
                            value=DEFAULT_PERSPECTIVE,
                            elem_id="ctl-perspective",
                        )
                        backbone = gr.Dropdown(
                            label="Backbone",
                            choices=_model_choice_pairs(show_experimental=False),
                            value=initial_model,
                            elem_id="ctl-backbone",
                        )
                        warning_html = gr.HTML(value=initial_warning)

                    with gr.Accordion("Data", open=True):
                        triplet = gr.Dropdown(
                            label="Triplet",
                            choices=_all_triplet_ids(),
                            value=initial_triplet,
                            elem_id="ctl-triplet",
                        )
                        source_mode = gr.Radio(
                            label="Image Source",
                            choices=[DEFAULT_SOURCE_MODE, REAL_WORLD_SOURCE_MODE, UPLOAD_SOURCE_MODE],
                            value=DEFAULT_SOURCE_MODE,
                            elem_id="ctl-source-mode",
                        )
                        sample_name = gr.Dropdown(
                            label="Gallery Image",
                            choices=_source_image_names(initial_triplet, DEFAULT_SOURCE_MODE),
                            value=initial_sample,
                            visible=True,
                            elem_id="ctl-sample-name",
                        )
                        uploaded_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            visible=False,
                            show_fullscreen_button=True,
                        )
                        source_warning_html = gr.HTML(value=initial_source_warning)

                    with gr.Accordion("Attention", open=True):
                        map_name = gr.Dropdown(
                            label="Attention View",
                            choices=["mean", "max", "min", "std", "best_q", "q1", "q2", "q3", "q4"],
                            value=DEFAULT_MAP_NAME,
                            elem_id="ctl-map-name",
                        )
                        alpha = gr.Slider(
                            label="Overlay Alpha",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=DEFAULT_ALPHA,
                            elem_id="ctl-alpha",
                        )
                        compare_enabled = gr.Checkbox(
                            label="Enable Compare Mode",
                            value=DEFAULT_COMPARE_ENABLED,
                            elem_id="ctl-compare-enabled",
                        )
                        compare_map_name = gr.Dropdown(
                            label="Compare Attention View",
                            choices=["mean", "max", "min", "std", "best_q", "q1", "q2", "q3", "q4"],
                            value=DEFAULT_COMPARE_MAP,
                            visible=False,
                            elem_id="ctl-compare-map",
                        )
                        view_mode = gr.Radio(
                            label="Result View Mode",
                            choices=["Split", "Tabs"],
                            value=DEFAULT_VIEW_MODE,
                            elem_id="ctl-view-mode",
                        )

                    with gr.Accordion("Advanced", open=False):
                        sharpen = gr.Checkbox(
                            label="Sharpen Attention",
                            value=DEFAULT_SHARPEN,
                            elem_id="ctl-sharpen",
                        )
                        clip_percentile = gr.Slider(
                            label="Clip Percentile",
                            minimum=90.0,
                            maximum=100.0,
                            step=0.5,
                            value=DEFAULT_CLIP_PERCENTILE,
                            elem_id="ctl-clip-percentile",
                        )
                        gamma = gr.Slider(
                            label="Gamma (lower = sharper)",
                            minimum=0.30,
                            maximum=1.50,
                            step=0.05,
                            value=DEFAULT_GAMMA,
                            elem_id="ctl-gamma",
                        )

                    with gr.Accordion("Output", open=True):
                        run_btn = gr.Button(
                            "Run Inference",
                            variant="primary",
                            interactive=_is_run_interactive(initial_model, DEFAULT_SOURCE_MODE, initial_triplet),
                        )
                        with gr.Row():
                            preset_btn = gr.Button("Recommended Preset")
                            reset_btn = gr.Button("Reset All Controls")
                        with gr.Row():
                            copy_btn = gr.Button("Copy Settings")
                        copy_status_html = gr.HTML(value="", elem_id="copy-status")

                    stage_html = gr.HTML(
                        value=build_stage_indicator("Idle. Configure controls and run inference.", state="idle")
                    )

            with gr.Column():
                with gr.Row():
                    triplet_html = gr.HTML(value=_build_triplet_html(initial_triplet))
                    sample_preview = gr.Image(
                        label="Sample Preview",
                        value=_gallery_preview(initial_triplet, DEFAULT_SOURCE_MODE, initial_sample),
                        show_fullscreen_button=True,
                    )

                with gr.Row():
                    prediction_html = gr.HTML(value=build_prediction_placeholder())
                    labels_html = gr.HTML(
                        value=build_label_card(
                            gt_label="N/A",
                            pred_label="N/A",
                            pred_prob=0.0,
                            source_mode=UPLOAD_SOURCE_MODE,
                        )
                    )

                run_info_html = gr.HTML(value=build_run_info_card([("Status", "Waiting for inference")]))

                with gr.Row():
                    legend_html = gr.HTML(value=build_legend_html())
                    with gr.Accordion("How to Read Attention Maps", open=False):
                        gr.HTML(value=build_help_html())

                with gr.Group(visible=True) as split_results_group:
                    with gr.Row():
                        overlay_img = gr.Image(
                            label="Overlay (Selected Attention Map)",
                            show_fullscreen_button=True,
                        )
                        heatmap_img = gr.Image(
                            label="Selected Attention Heatmap",
                            show_fullscreen_button=True,
                        )

                with gr.Group(visible=False) as tabs_results_group:
                    with gr.Tabs():
                        with gr.Tab("Overlay"):
                            overlay_img_tab = gr.Image(
                                label="Overlay",
                                show_fullscreen_button=True,
                            )
                        with gr.Tab("Heatmap"):
                            heatmap_img_tab = gr.Image(
                                label="Heatmap",
                                show_fullscreen_button=True,
                            )

                with gr.Group(visible=False) as compare_group:
                    gr.Markdown("### Compare Attention Views")
                    with gr.Row():
                        compare_img_a = gr.Image(
                            label="Primary (Selected View)",
                            show_fullscreen_button=True,
                        )
                        compare_img_b = gr.Image(
                            label="Secondary (Compare View)",
                            show_fullscreen_button=True,
                        )

                download_file = gr.File(label="Download Selected Attention Heatmap")
                gr.HTML(value=build_footer_html(APP_VERSION))

        settings_json = gr.Textbox(visible=False)

        show_experimental.change(
            fn=_on_show_experimental_change,
            inputs=[show_experimental, backbone, source_mode, triplet],
            outputs=[backbone, warning_html, source_warning_html, run_btn],
        )

        backbone.change(
            fn=_on_backbone_change,
            inputs=[backbone, source_mode, triplet],
            outputs=[warning_html, source_warning_html, run_btn],
        )

        triplet.change(
            fn=_on_triplet_change,
            inputs=[triplet, source_mode, sample_name, backbone],
            outputs=[sample_name, sample_preview, triplet_html, source_warning_html, run_btn],
        )

        sample_name.change(
            fn=_on_gallery_image_change,
            inputs=[triplet, source_mode, sample_name],
            outputs=[sample_preview],
        )

        source_mode.change(
            fn=_on_source_mode_change,
            inputs=[source_mode, triplet, sample_name, backbone],
            outputs=[sample_name, sample_preview, uploaded_image, source_warning_html, run_btn],
        )

        view_mode.change(
            fn=_view_mode_visibility,
            inputs=[view_mode],
            outputs=[split_results_group, tabs_results_group],
        )

        compare_enabled.change(
            fn=_compare_visibility,
            inputs=[compare_enabled],
            outputs=[compare_map_name, compare_group],
        )

        preset_btn.click(
            fn=_apply_recommended_preset,
            inputs=None,
            outputs=[
                map_name,
                alpha,
                compare_enabled,
                compare_map_name,
                view_mode,
                sharpen,
                clip_percentile,
                gamma,
                copy_status_html,
            ],
        )

        reset_btn.click(
            fn=_reset_controls,
            inputs=[show_experimental],
            outputs=[
                perspective,
                backbone,
                triplet,
                source_mode,
                sample_name,
                uploaded_image,
                map_name,
                alpha,
                compare_enabled,
                compare_map_name,
                view_mode,
                sharpen,
                clip_percentile,
                gamma,
                sample_preview,
                triplet_html,
                warning_html,
                source_warning_html,
                run_btn,
                copy_status_html,
                stage_html,
                split_results_group,
                tabs_results_group,
                compare_group,
            ],
        )

        copy_event = copy_btn.click(
            fn=_serialize_current_settings,
            inputs=[
                show_experimental,
                perspective,
                backbone,
                triplet,
                source_mode,
                sample_name,
                map_name,
                alpha,
                compare_enabled,
                compare_map_name,
                view_mode,
                sharpen,
                clip_percentile,
                gamma,
            ],
            outputs=[settings_json],
        )

        copy_event.then(
            fn=None,
            inputs=[settings_json],
            outputs=[copy_status_html],
            js="""
            (txt) => {
              if (!txt) {
                return '<div class="copy-status copy-error">No settings to copy.</div>';
              }
              if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(txt).catch(() => {});
                return '<div class="copy-status copy-ok">Settings copied to clipboard.</div>';
              }
              return '<div class="copy-status copy-error">Clipboard API is unavailable in this browser.</div>';
            }
            """,
        )

        run_btn.click(
            fn=_infer_stream,
            inputs=[
                perspective,
                backbone,
                triplet,
                source_mode,
                sample_name,
                uploaded_image,
                map_name,
                alpha,
                compare_enabled,
                compare_map_name,
                sharpen,
                clip_percentile,
                gamma,
            ],
            outputs=[
                prediction_html,
                labels_html,
                overlay_img,
                heatmap_img,
                overlay_img_tab,
                heatmap_img_tab,
                compare_img_a,
                compare_img_b,
                download_file,
                run_info_html,
                stage_html,
                triplet_html,
            ],
        )

    return demo


def main() -> None:
    demo = _build_demo()
    demo.queue(max_size=16)
    demo.launch()


if __name__ == "__main__":
    main()
