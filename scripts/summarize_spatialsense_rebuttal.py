from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REAL_WORLD_DATASET_NAME = "real_world_position"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_FOLDER_TO_LABEL = {
    "front": "Front",
    "back": "Back",
    "left": "Left",
    "right": "Right",
}
LABEL_TO_INDEX = {"Front": 0, "Back": 1, "Left": 2, "Right": 3}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def method_from_probe_name(probe_name: str) -> str:
    probe_name = str(probe_name).lower().strip()
    if "efficient" in probe_name:
        return "EfficientProbing"
    if "abmilp" in probe_name:
        return "ABMILP"
    return "GAP"


def parse_percent(value: str) -> float:
    return float(value) / 100.0


def parse_percent_or_none(value: str) -> Optional[float]:
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return float(raw) / 100.0
    except ValueError:
        return None


def summarize_metric_percent(grp: List[Dict[str, str]], key: str) -> Tuple[str, str, str]:
    vals: List[float] = []
    for row in grp:
        val = parse_percent_or_none(row.get(key, ""))
        if val is not None:
            vals.append(val)
    if not vals:
        return "", "", "0"
    arr = np.asarray(vals, dtype=np.float64)
    return f"{arr.mean()*100:.2f}", f"{arr.std(ddof=0)*100:.2f}", str(len(vals))


def split_mode_from_row(row: Dict[str, str]) -> str:
    return row.get("Split Mode", "") or row.get("Perspective", "unknown")


def _run_stage_from_row(row: Dict[str, str]) -> str:
    blob = (
        f"{row.get('Predictions Path', '')} "
        f"{row.get('Head Path', '')}"
    ).lower()
    if "_final_" in blob:
        return "final"
    if "_tune_" in blob:
        return "tune"
    return "unknown"


def _safe_int(raw: str, default: int = -1) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def select_final_rows_one_per_seed(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Keep only final-stage runs and deduplicate to one row per
    # (backbone, split_mode, method, seed), preferring the latest appended row.
    latest: Dict[Tuple[str, str, str, int], Dict[str, str]] = {}
    for row in reversed(rows):
        if _run_stage_from_row(row) != "final":
            continue
        method = method_from_probe_name(row.get("Probe Name", ""))
        key = (
            row.get("Backbone", "unknown"),
            split_mode_from_row(row),
            method,
            _safe_int(row.get("Random Seed", "-1")),
        )
        if key in latest:
            continue
        latest[key] = row
    final_rows = list(latest.values())
    final_rows.sort(
        key=lambda r: (
            r.get("Backbone", "unknown"),
            split_mode_from_row(r),
            method_from_probe_name(r.get("Probe Name", "")),
            _safe_int(r.get("Random Seed", "-1")),
            r.get("Timestamp", ""),
        )
    )
    return final_rows


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = []
    for row in rows:
        if row.get("Train Dataset", "") != REAL_WORLD_DATASET_NAME:
            continue
        keep.append(row)
    return keep


def make_summary_by_backbone_split(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = method_from_probe_name(row.get("Probe Name", ""))
        key = (row.get("Backbone", "unknown"), split_mode_from_row(row), method)
        groups[key].append(row)

    summary = []
    for (backbone, split_mode, method), grp in sorted(groups.items()):
        top1_mean, top1_std, top1_n = summarize_metric_percent(grp, "Top1 Test")
        top2_mean, top2_std, top2_n = summarize_metric_percent(grp, "Top2 Test")
        bal_mean, bal_std, bal_n = summarize_metric_percent(grp, "Balanced Acc Test")
        macro_f1_mean, macro_f1_std, macro_f1_n = summarize_metric_percent(grp, "Macro F1 Test")
        top1_val_mean, top1_val_std, top1_val_n = summarize_metric_percent(grp, "Top1 Val")
        top2_val_mean, top2_val_std, top2_val_n = summarize_metric_percent(grp, "Top2 Val")
        bal_val_mean, bal_val_std, bal_val_n = summarize_metric_percent(grp, "Balanced Acc Val")
        seeds = sorted({int(r["Random Seed"]) for r in grp})
        summary.append(
            {
                "Backbone": backbone,
                "Split Mode": split_mode,
                "Method": method,
                "Seeds": ",".join(str(s) for s in seeds),
                "N": str(len(grp)),
                "Top1 Mean": top1_mean,
                "Top1 Std": top1_std,
                "Top1 N": top1_n,
                "Top2 Mean": top2_mean,
                "Top2 Std": top2_std,
                "Top2 N": top2_n,
                "Balanced Acc Mean": bal_mean,
                "Balanced Acc Std": bal_std,
                "Balanced Acc N": bal_n,
                "Macro F1 Mean": macro_f1_mean,
                "Macro F1 Std": macro_f1_std,
                "Macro F1 N": macro_f1_n,
                "Top1 Val Mean": top1_val_mean,
                "Top1 Val Std": top1_val_std,
                "Top1 Val N": top1_val_n,
                "Top2 Val Mean": top2_val_mean,
                "Top2 Val Std": top2_val_std,
                "Top2 Val N": top2_val_n,
                "Balanced Acc Val Mean": bal_val_mean,
                "Balanced Acc Val Std": bal_val_std,
                "Balanced Acc Val N": bal_val_n,
            }
        )
    return summary


def make_summary_by_split_mode(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = method_from_probe_name(row.get("Probe Name", ""))
        key = (split_mode_from_row(row), method)
        groups[key].append(row)

    summary = []
    for (split_mode, method), grp in sorted(groups.items()):
        top1_mean, top1_std, top1_n = summarize_metric_percent(grp, "Top1 Test")
        top2_mean, top2_std, top2_n = summarize_metric_percent(grp, "Top2 Test")
        bal_mean, bal_std, bal_n = summarize_metric_percent(grp, "Balanced Acc Test")
        top1_val_mean, top1_val_std, top1_val_n = summarize_metric_percent(grp, "Top1 Val")
        top2_val_mean, top2_val_std, top2_val_n = summarize_metric_percent(grp, "Top2 Val")
        bal_val_mean, bal_val_std, bal_val_n = summarize_metric_percent(grp, "Balanced Acc Val")
        summary.append(
            {
                "Split Mode": split_mode,
                "Method": method,
                "N": str(len(grp)),
                "Top1 Mean": top1_mean,
                "Top1 Std": top1_std,
                "Top1 N": top1_n,
                "Top2 Mean": top2_mean,
                "Top2 Std": top2_std,
                "Top2 N": top2_n,
                "Balanced Acc Mean": bal_mean,
                "Balanced Acc Std": bal_std,
                "Balanced Acc N": bal_n,
                "Top1 Val Mean": top1_val_mean,
                "Top1 Val Std": top1_val_std,
                "Top1 Val N": top1_val_n,
                "Top2 Val Mean": top2_val_mean,
                "Top2 Val Std": top2_val_std,
                "Top2 Val N": top2_val_n,
                "Balanced Acc Val Mean": bal_val_mean,
                "Balanced Acc Val Std": bal_val_std,
                "Balanced Acc Val N": bal_val_n,
            }
        )
    return summary


def make_summary_overall(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = method_from_probe_name(row.get("Probe Name", ""))
        groups[method].append(row)

    summary = []
    for method, grp in sorted(groups.items()):
        top1_mean, top1_std, top1_n = summarize_metric_percent(grp, "Top1 Test")
        top2_mean, top2_std, top2_n = summarize_metric_percent(grp, "Top2 Test")
        bal_mean, bal_std, bal_n = summarize_metric_percent(grp, "Balanced Acc Test")
        top1_val_mean, top1_val_std, top1_val_n = summarize_metric_percent(grp, "Top1 Val")
        top2_val_mean, top2_val_std, top2_val_n = summarize_metric_percent(grp, "Top2 Val")
        bal_val_mean, bal_val_std, bal_val_n = summarize_metric_percent(grp, "Balanced Acc Val")
        summary.append(
            {
                "Method": method,
                "N": str(len(grp)),
                "Top1 Mean": top1_mean,
                "Top1 Std": top1_std,
                "Top1 N": top1_n,
                "Top2 Mean": top2_mean,
                "Top2 Std": top2_std,
                "Top2 N": top2_n,
                "Balanced Acc Mean": bal_mean,
                "Balanced Acc Std": bal_std,
                "Balanced Acc N": bal_n,
                "Top1 Val Mean": top1_val_mean,
                "Top1 Val Std": top1_val_std,
                "Top1 Val N": top1_val_n,
                "Top2 Val Mean": top2_val_mean,
                "Top2 Val Std": top2_val_std,
                "Top2 Val N": top2_val_n,
                "Balanced Acc Val Mean": bal_val_mean,
                "Balanced Acc Val Std": bal_val_std,
                "Balanced Acc Val N": bal_val_n,
            }
        )
    return summary


def bootstrap_delta_top1(
    paired_runs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seed_diffs = []
        for y_true, y_pred_eff, y_pred_gap in paired_runs:
            n = y_true.shape[0]
            sample_idx = rng.integers(0, n, size=n)
            yt = y_true[sample_idx]
            ye = y_pred_eff[sample_idx]
            yg = y_pred_gap[sample_idx]
            top1_eff = float((ye == yt).mean())
            top1_gap = float((yg == yt).mean())
            seed_diffs.append(top1_eff - top1_gap)
        diffs[b] = float(np.mean(seed_diffs)) if seed_diffs else 0.0

    point = float(np.mean(diffs))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_left = float(np.mean(diffs <= 0.0))
    p_right = float(np.mean(diffs >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "delta_mean": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_two_sided,
    }


def load_npz_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    y_true = arr["y_true"].astype(np.int64)
    y_pred = arr["y_pred"].astype(np.int64)
    return y_true, y_pred


def run_significance_top1(rows: List[Dict[str, str]], n_boot: int, seed: int) -> List[Dict[str, str]]:
    by_group_seed: Dict[Tuple[str, str, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = method_from_probe_name(row.get("Probe Name", ""))
        key = (row.get("Backbone", "unknown"), split_mode_from_row(row), int(row.get("Random Seed", -1)))
        by_group_seed[key][method] = row

    grouped_pairs: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (backbone, split_mode, _seed), methods in by_group_seed.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        pred_eff = Path(methods["EfficientProbing"].get("Predictions Path", ""))
        pred_gap = Path(methods["GAP"].get("Predictions Path", ""))
        if not pred_eff.exists() or not pred_gap.exists():
            continue
        y_true_e, y_pred_e = load_npz_predictions(pred_eff)
        y_true_g, y_pred_g = load_npz_predictions(pred_gap)
        if y_true_e.shape != y_true_g.shape or np.any(y_true_e != y_true_g):
            continue
        grouped_pairs[(backbone, split_mode)].append((y_true_e, y_pred_e, y_pred_g))

    results = []
    for (backbone, split_mode), paired_runs in sorted(grouped_pairs.items()):
        if not paired_runs:
            continue
        sig = bootstrap_delta_top1(paired_runs, n_boot=n_boot, seed=seed)
        results.append(
            {
                "Backbone": backbone,
                "Split Mode": split_mode,
                "Paired Seeds": str(len(paired_runs)),
                "Delta Top1 Mean": f"{sig['delta_mean']*100:.2f}",
                "CI Low 2.5%": f"{sig['ci_low']*100:.2f}",
                "CI High 97.5%": f"{sig['ci_high']*100:.2f}",
                "P Value": f"{sig['p_value']:.6f}",
            }
        )
    return results


def bootstrap_delta_scalar(
    deltas: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    if deltas.size == 0:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, deltas.size, size=deltas.size)
        boots[b] = float(np.mean(deltas[idx]))
    point = float(np.mean(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_left = float(np.mean(boots <= 0.0))
    p_right = float(np.mean(boots >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(p_left, p_right))
    return {
        "delta_mean": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_two_sided,
    }


def run_significance_val_top1(rows: List[Dict[str, str]], n_boot: int, seed: int) -> List[Dict[str, str]]:
    by_group_seed: Dict[Tuple[str, str, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = method_from_probe_name(row.get("Probe Name", ""))
        key = (row.get("Backbone", "unknown"), split_mode_from_row(row), int(row.get("Random Seed", -1)))
        by_group_seed[key][method] = row

    grouped_deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (backbone, split_mode, _seed), methods in by_group_seed.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        top1_eff = parse_percent_or_none(methods["EfficientProbing"].get("Top1 Val", ""))
        top1_gap = parse_percent_or_none(methods["GAP"].get("Top1 Val", ""))
        if top1_eff is None or top1_gap is None:
            continue
        grouped_deltas[(backbone, split_mode)].append(float(top1_eff - top1_gap))

    results = []
    for (backbone, split_mode), deltas in sorted(grouped_deltas.items()):
        if not deltas:
            continue
        arr = np.asarray(deltas, dtype=np.float64)
        sig = bootstrap_delta_scalar(arr, n_boot=n_boot, seed=seed)
        results.append(
            {
                "Backbone": backbone,
                "Split Mode": split_mode,
                "Paired Seeds": str(arr.size),
                "Delta Top1 Val Mean": f"{sig['delta_mean']*100:.2f}",
                "CI Low 2.5%": f"{sig['ci_low']*100:.2f}",
                "CI High 97.5%": f"{sig['ci_high']*100:.2f}",
                "P Value": f"{sig['p_value']:.6f}",
            }
        )
    return results


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w") as f:
            f.write("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_top1_comparison(
    summary_rows: List[Dict[str, str]],
    output_path: Path,
    metric_prefix: str,
    title_suffix: str,
    ylabel_suffix: str,
) -> None:
    mean_key = f"{metric_prefix} Mean"
    std_key = f"{metric_prefix} Std"
    grouped = defaultdict(dict)
    for row in summary_rows:
        key = f"{row['Backbone']} | {row['Split Mode']}"
        grouped[key][row["Method"]] = row

    labels = []
    eff_means = []
    eff_stds = []
    gap_means = []
    gap_stds = []
    for key, methods in grouped.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        eff_mean = methods["EfficientProbing"].get(mean_key, "")
        gap_mean = methods["GAP"].get(mean_key, "")
        eff_std = methods["EfficientProbing"].get(std_key, "")
        gap_std = methods["GAP"].get(std_key, "")
        if eff_mean == "" or gap_mean == "" or eff_std == "" or gap_std == "":
            continue
        labels.append(key)
        eff_means.append(float(eff_mean))
        eff_stds.append(float(eff_std))
        gap_means.append(float(gap_mean))
        gap_stds.append(float(gap_std))

    if not labels:
        return

    x = np.arange(len(labels), dtype=np.float64)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5.5))
    ax.bar(x - w / 2, eff_means, width=w, yerr=eff_stds, capsize=3, label="EfficientProbing")
    ax.bar(x + w / 2, gap_means, width=w, yerr=gap_stds, capsize=3, label="GAP/MeanPool")
    ax.set_ylabel(f"{ylabel_suffix} (%)")
    ax.set_title(f"Real-World Rebuttal: EfficientProbing vs GAP ({title_suffix})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _collect_class_files(dataset_root: Path) -> Dict[str, List[Path]]:
    class_files: Dict[str, List[Path]] = {}
    for folder_name, class_label in CLASS_FOLDER_TO_LABEL.items():
        folder = dataset_root / folder_name
        files: List[Path] = []
        if folder.exists():
            files = [
                p
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            files.sort(key=lambda p: p.as_posix())
        class_files[class_label] = files
    return class_files


def _extract_numeric_id(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return -1
    try:
        return int(matches[-1])
    except ValueError:
        return -1


def _select_split_file_indices(
    files: Sequence[Path],
    split_mode: str,
    train_ratio: float,
    valid_ratio: float,
    rng: np.random.RandomState,
    clip_block_size: int = 20,
    clip_id_source: str = "filename_numeric",
) -> Dict[str, np.ndarray]:
    n = len(files)
    order = np.arange(n, dtype=np.int64)
    if split_mode == "random":
        rng.shuffle(order)
        n_train = int(np.floor(n * train_ratio))
        n_valid = int(np.floor(n * valid_ratio))
        train_idx = order[:n_train]
        valid_idx = order[n_train : n_train + n_valid]
        test_idx = order[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}
    if split_mode == "time_series":
        n_train = int(np.floor(n * train_ratio))
        n_valid = int(np.floor(n * valid_ratio))
        train_idx = order[:n_train]
        tail_idx = order[n_train:].copy()
        rng.shuffle(tail_idx)
        valid_idx = tail_idx[:n_valid]
        test_idx = tail_idx[n_valid:]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    if split_mode not in {"clip_block_random", "clip_block_time_series"}:
        raise ValueError(f"Unsupported split mode for sanity report: {split_mode}")

    clip_block_size = max(1, int(clip_block_size))
    clip_id_source = str(clip_id_source).strip().lower()
    numeric_ids = [_extract_numeric_id(p) for p in files]
    valid_numeric_ids = [x for x in numeric_ids if x >= 0]
    min_numeric_id = min(valid_numeric_ids) if valid_numeric_ids else 0

    block_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(files):
        if clip_id_source == "parent_folder":
            block_key = p.parent.as_posix()
        else:
            num_id = numeric_ids[i]
            if num_id < 0:
                num_id = i
            block_key = f"b{(num_id - min_numeric_id) // clip_block_size}"
        block_to_indices[block_key].append(i)

    block_keys = sorted(block_to_indices.keys())
    block_order = np.arange(len(block_keys), dtype=np.int64)
    if split_mode == "clip_block_random":
        rng.shuffle(block_order)

    n_blocks = len(block_keys)
    n_train_blocks = int(np.floor(n_blocks * train_ratio))
    n_valid_blocks = int(np.floor(n_blocks * valid_ratio))
    if split_mode == "clip_block_time_series":
        train_blocks = block_order[:n_train_blocks]
        tail_blocks = block_order[n_train_blocks:].copy()
        rng.shuffle(tail_blocks)
        valid_blocks = tail_blocks[:n_valid_blocks]
        test_blocks = tail_blocks[n_valid_blocks:]
    else:
        train_blocks = block_order[:n_train_blocks]
        valid_blocks = block_order[n_train_blocks : n_train_blocks + n_valid_blocks]
        test_blocks = block_order[n_train_blocks + n_valid_blocks :]

    def _expand(block_indices: np.ndarray) -> np.ndarray:
        out: List[int] = []
        for block_pos in block_indices.tolist():
            out.extend(block_to_indices[block_keys[int(block_pos)]])
        out.sort()
        return np.asarray(out, dtype=np.int64)

    return {
        "train": _expand(train_blocks),
        "valid": _expand(valid_blocks),
        "test": _expand(test_blocks),
    }


def dataset_sanity_report(
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    clip_block_size: int = 20,
    clip_id_source: str = "filename_numeric",
) -> None:
    class_files = _collect_class_files(dataset_root)
    rows: List[Dict[str, str]] = []

    # Raw class counts from folder scan
    folder_counts: Dict[str, int] = {
        class_name: len(files) for class_name, files in class_files.items()
    }
    rows.append(
        {
            "Section": "raw_folder_counts",
            "Split Mode": "all",
            "Split": "all",
            "Total": str(sum(folder_counts.values())),
            "Front": str(folder_counts["Front"]),
            "Back": str(folder_counts["Back"]),
            "Left": str(folder_counts["Left"]),
            "Right": str(folder_counts["Right"]),
        }
    )

    for split_mode in ["random", "time_series", "clip_block_random", "clip_block_time_series"]:
        split_counts = {
            "train": {"Front": 0, "Back": 0, "Left": 0, "Right": 0},
            "valid": {"Front": 0, "Back": 0, "Left": 0, "Right": 0},
            "test": {"Front": 0, "Back": 0, "Left": 0, "Right": 0},
        }
        rng = np.random.RandomState(seed)
        for class_name, files in class_files.items():
            idx = _select_split_file_indices(
                files=files,
                split_mode=split_mode,
                train_ratio=0.8,
                valid_ratio=0.1,
                rng=rng,
                clip_block_size=clip_block_size,
                clip_id_source=clip_id_source,
            )
            split_counts["train"][class_name] += int(idx["train"].shape[0])
            split_counts["valid"][class_name] += int(idx["valid"].shape[0])
            split_counts["test"][class_name] += int(idx["test"].shape[0])

        for split in ["train", "valid", "test"]:
            counts = split_counts[split]
            total = sum(counts.values())
            rows.append(
                {
                    "Section": "split_counts",
                    "Split Mode": split_mode,
                    "Split": split,
                    "Total": str(total),
                    "Front": str(counts["Front"]),
                    "Back": str(counts["Back"]),
                    "Left": str(counts["Left"]),
                    "Right": str(counts["Right"]),
                }
            )
        trainval_counts = {
            class_name: split_counts["train"][class_name] + split_counts["valid"][class_name]
            for class_name in ["Front", "Back", "Left", "Right"]
        }
        rows.append(
            {
                "Section": "split_counts",
                "Split Mode": split_mode,
                "Split": "trainval",
                "Total": str(sum(trainval_counts.values())),
                "Front": str(trainval_counts["Front"]),
                "Back": str(trainval_counts["Back"]),
                "Left": str(trainval_counts["Left"]),
                "Right": str(trainval_counts["Right"]),
            }
        )

    write_csv(output_dir / "dataset_sanity.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real-world rebuttal runs.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--clip-block-size", type=int, default=20)
    parser.add_argument("--clip-id-source", type=str, default="filename_numeric")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images"
        ),
    )
    args = parser.parse_args()

    rows = read_rows(args.results_csv)
    if not rows:
        raise RuntimeError(f"No {REAL_WORLD_DATASET_NAME} rows found in {args.results_csv}")
    rows_final_only = select_final_rows_one_per_seed(rows)

    summary_backbone_split = make_summary_by_backbone_split(rows)
    summary_backbone_split_final_only = make_summary_by_backbone_split(rows_final_only)
    summary_split_mode = make_summary_by_split_mode(rows)
    summary_overall = make_summary_overall(rows)
    sig_rows_test = run_significance_top1(rows, n_boot=args.n_bootstrap, seed=args.seed)
    sig_rows_val = run_significance_val_top1(rows, n_boot=args.n_bootstrap, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_by_backbone_split.csv", summary_backbone_split)
    write_csv(
        args.output_dir / "summary_by_backbone_split_final_only.csv",
        summary_backbone_split_final_only,
    )
    write_csv(args.output_dir / "summary_by_split_mode.csv", summary_split_mode)
    write_csv(args.output_dir / "summary_overall.csv", summary_overall)
    write_csv(args.output_dir / "significance_top1.csv", sig_rows_test)
    write_csv(args.output_dir / "significance_top1_val.csv", sig_rows_val)
    plot_top1_comparison(
        summary_backbone_split,
        args.output_dir / "top1_comparison.png",
        metric_prefix="Top1",
        title_suffix="Test Top-1",
        ylabel_suffix="Top-1 Test",
    )
    plot_top1_comparison(
        summary_backbone_split,
        args.output_dir / "top1_val_comparison.png",
        metric_prefix="Top1 Val",
        title_suffix="Validation Top-1",
        ylabel_suffix="Top-1 Val",
    )
    dataset_sanity_report(
        args.dataset_root,
        args.output_dir,
        seed=args.seed,
        clip_block_size=args.clip_block_size,
        clip_id_source=args.clip_id_source,
    )

    report_path = args.output_dir / "report.md"
    with report_path.open("w") as f:
        f.write("# Real-World Rebuttal Report\n\n")
        f.write("## Summary Files\n")
        f.write("- `summary_by_backbone_split.csv`\n")
        f.write("- `summary_by_backbone_split_final_only.csv`\n")
        f.write("- `summary_by_split_mode.csv`\n")
        f.write("- `summary_overall.csv`\n")
        f.write("- `significance_top1.csv`\n")
        f.write("- `significance_top1_val.csv`\n")
        f.write("- `dataset_sanity.csv`\n")
        f.write("- `top1_comparison.png`\n")
        f.write("- `top1_val_comparison.png`\n")
    print(f"[DONE] Wrote report artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
