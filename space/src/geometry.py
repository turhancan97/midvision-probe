from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def classify_relative_direction(
    observer_xy: np.ndarray,
    ref_xy: np.ndarray,
    tgt_xy: np.ndarray,
    amb_deg: int = 20,
    front_deg: int = 45,
    back_deg: int = 135,
) -> str:
    fwd = ref_xy - observer_xy
    rel = tgt_xy - ref_xy

    nf = np.linalg.norm(fwd)
    nr = np.linalg.norm(rel)
    if nf == 0 or nr == 0:
        return "Ambiguous"

    f = fwd / nf
    r = rel / nr

    dot = float(np.dot(f, r))
    det = float(f[0] * r[1] - f[1] * r[0])
    ang = np.degrees(np.arctan2(det, dot))
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
    match = None
    for pat in patterns:
        match = re.search(pat, image_path.name)
        if match:
            break
    if not match:
        return None

    idx = match.group(1).lstrip("0")
    if idx == "":
        idx = "0"

    json_path = image_path.parent / f"params_{idx}.json"
    if json_path.exists():
        return json_path

    for width in (4, 5):
        padded = image_path.parent / f"params_{int(idx):0{width}d}.json"
        if padded.exists():
            return padded
    return None


def _load_positions(
    json_path: Path,
    reference_label: str,
    target_label: str,
    human_label: Optional[str] = None,
) -> Optional[Dict[str, Optional[np.ndarray]]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        cam_pos = metadata["camera"]["location"]
        actors = metadata["actors"]
        positions = {v["label"]: v["location"] for v in actors.values()}
        if reference_label not in positions or target_label not in positions:
            return None

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
            loc = positions.get(human_label)
            if loc is not None:
                human = np.array([-1 * loc["x"], loc["y"]], dtype=float)

        return {"camera": cam, "reference": ref, "target": tgt, "human": human}
    except Exception:
        return None

