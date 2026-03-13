from __future__ import annotations

import html
from datetime import date
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _esc(value: object) -> str:
    return html.escape(str(value))


def build_hero_html(logo_src: str) -> str:
    return f"""
<div class="hero-card">
  <div class="hero-logo-wrap">
    <img class="hero-logo" src="{_esc(logo_src)}" alt="SpaRRTA logo" />
  </div>
  <div class="hero-content">
    <h1>SpaRRTA</h1>
    <p class="hero-subtitle">Spatial relation probing with timm DINOv3 and efficient probing heads</p>
    <div class="hero-guide">
      <span>1) Pick model and triplet</span>
      <span>2) Choose attention view</span>
      <span>3) Run inference</span>
    </div>
  </div>
</div>
"""


def build_triplet_card(triplet_cfg: dict) -> str:
    return f"""
<div class="card triplet-card">
  <div class="card-title">Triplet</div>
  <div class="meta-grid">
    <div><span class="meta-k">Environment</span><span class="meta-v">{_esc(triplet_cfg["environment"])}</span></div>
    <div><span class="meta-k">Reference</span><span class="meta-v">{_esc(triplet_cfg["reference_label"])}</span></div>
    <div><span class="meta-k">Target</span><span class="meta-v">{_esc(triplet_cfg["target_label"])}</span></div>
    <div><span class="meta-k">Human Anchor</span><span class="meta-v">{_esc(triplet_cfg["human_label"])}</span></div>
  </div>
</div>
"""


def build_prediction_placeholder() -> str:
    return """
<div class="card prediction-card">
  <div class="card-title">Prediction</div>
  <div class="placeholder">Run inference to view top-2 probabilities and logits.</div>
</div>
"""


def build_prediction_card(classes: Sequence[str], logits: torch.Tensor) -> Tuple[str, str, float]:
    logits = logits.float()
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    logits_np = logits[0].detach().cpu().numpy()
    top2_idx = np.argsort(-probs)[:2]

    top1_idx = int(top2_idx[0])
    top2_idx_val = int(top2_idx[1])
    top1_label = str(classes[top1_idx])
    top1_prob = float(probs[top1_idx])
    top2_label = str(classes[top2_idx_val])
    top2_prob = float(probs[top2_idx_val])
    margin = top1_prob - top2_prob
    if margin >= 0.50:
        confidence = "High"
    elif margin >= 0.20:
        confidence = "Medium"
    else:
        confidence = "Low"

    logits_row = " | ".join(
        f"{_esc(cls)}: {float(value):.4f}" for cls, value in zip(classes, logits_np.tolist())
    )

    card = f"""
<div class="card prediction-card">
  <div class="card-title">Prediction</div>
  <div class="pred-chip-row">
    <span class="chip chip-strong">Top-1: {_esc(top1_label)} ({top1_prob*100:.2f}%)</span>
    <span class="chip chip-soft">Top-2: {_esc(top2_label)} ({top2_prob*100:.2f}%)</span>
  </div>
  <div class="prob-row">
    <div class="prob-label">{_esc(top1_label)}</div>
    <div class="prob-track"><div class="prob-fill p1" style="width: {top1_prob*100:.2f}%"></div></div>
    <div class="prob-val">{top1_prob*100:.2f}%</div>
  </div>
  <div class="prob-row">
    <div class="prob-label">{_esc(top2_label)}</div>
    <div class="prob-track"><div class="prob-fill p2" style="width: {top2_prob*100:.2f}%"></div></div>
    <div class="prob-val">{top2_prob*100:.2f}%</div>
  </div>
  <div class="pred-chip-row">
    <span class="chip chip-soft">Margin: {margin*100:.2f}%</span>
    <span class="chip chip-soft">Confidence: {_esc(confidence)}</span>
  </div>
  <div class="logits-line"><span>Raw logits</span><code>{logits_row}</code></div>
</div>
"""
    return card, top1_label, top1_prob


def _chip_class(label: str) -> str:
    name = label.lower()
    if name == "front":
        return "chip-front"
    if name == "back":
        return "chip-back"
    if name == "left":
        return "chip-left"
    if name == "right":
        return "chip-right"
    if name == "ambiguous":
        return "chip-ambiguous"
    return "chip-unknown"


def build_label_card(gt_label: str, pred_label: str, pred_prob: float, source_mode: str) -> str:
    if source_mode == "Sample gallery":
        gt_text = gt_label
        gt_class = _chip_class(gt_label)
        source_note = ""
    elif source_mode == "Real-world gallery":
        gt_text = "N/A (real-world sample)"
        gt_class = _chip_class("unknown")
        source_note = '<div class="banner banner-warn"><strong>OOD mode:</strong> real-world sample outside simulation distribution.</div>'
    else:
        gt_text = "N/A (uploaded image)"
        gt_class = _chip_class("unknown")
        source_note = ""
    pred_class = _chip_class(pred_label)

    return f"""
<div class="card labels-card">
  <div class="card-title">Ground Truth vs Prediction</div>
  {source_note}
  <div class="label-grid">
    <div class="label-block">
      <div class="label-k">Ground Truth</div>
      <div class="chip {gt_class}">{_esc(gt_text)}</div>
    </div>
    <div class="label-block">
      <div class="label-k">Predicted</div>
      <div class="chip {pred_class}">{_esc(pred_label)} ({pred_prob*100:.2f}%)</div>
    </div>
  </div>
</div>
"""


def build_run_info_card(entries: Iterable[Tuple[str, str]]) -> str:
    badges = "".join(
        f'<span class="badge"><span class="badge-k">{_esc(k)}</span><span class="badge-v">{_esc(v)}</span></span>'
        for k, v in entries
    )
    return f"""
<div class="card runinfo-card">
  <div class="card-title">Run Info</div>
  <div class="badge-wrap">{badges}</div>
</div>
"""


def build_model_warning(model_label: str, is_experimental: bool, available: bool, reason: str) -> str:
    if available and not is_experimental:
        return """
<div class="banner banner-ok">
  <strong>Model status:</strong> ready.
</div>
"""

    level_class = "banner-warn" if available else "banner-error"
    title = "Experimental model" if is_experimental else "Model availability"
    return f"""
<div class="banner {level_class}">
  <strong>{_esc(title)}:</strong> {_esc(reason)}<br />
  <span class="banner-sub">{_esc(model_label)}</span>
</div>
"""


def build_stage_indicator(stage: str, state: str = "idle") -> str:
    state_class = {
        "idle": "stage-idle",
        "running": "stage-running",
        "done": "stage-done",
        "error": "stage-error",
    }.get(state, "stage-idle")
    return f"""
<div class="stage-card {state_class}">
  <span class="stage-dot"></span>
  <span class="stage-text">{_esc(stage)}</span>
</div>
"""


def build_legend_html() -> str:
    return """
<div class="card legend-card">
  <div class="card-title">Attention Legend</div>
  <div class="legend-gradient"></div>
  <div class="legend-labels"><span>Low Attention</span><span>High Attention</span></div>
</div>
"""


def build_help_html() -> str:
    return """
<div class="help-list">
  <p><strong>How to read the map:</strong></p>
  <ul>
    <li>Warm colors indicate stronger probe attention concentration.</li>
    <li>`mean` is the default robust view across query heads.</li>
    <li>`q1..q4` isolates a single query head for inspection.</li>
    <li>Use compare mode to contrast two views side by side.</li>
  </ul>
</div>
"""


def build_footer_html(version: str) -> str:
    return f"""
<footer class="app-footer">
  <span><a href="https://huggingface.co/spaces" target="_blank" rel="noopener noreferrer">Hugging Face Spaces</a></span>
  <span><a href="https://github.com" target="_blank" rel="noopener noreferrer">Project Repository</a></span>
  <span>Version: {_esc(version)}</span>
  <span>Last Updated: {_esc(date.today().isoformat())}</span>
</footer>
"""


def build_copy_status(message: str, ok: bool = True) -> str:
    cls = "copy-ok" if ok else "copy-error"
    return f'<div class="copy-status {cls}">{_esc(message)}</div>'
