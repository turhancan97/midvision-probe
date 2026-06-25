from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_TRAIN_DATASETS = {
    "unreal_position_transfer",
    "unreal_real_holdout_transfer",
}


def method_from_probe_name(probe_name: str) -> str:
    if "efficient" in probe_name.lower():
        return "EfficientProbing"
    return "GAP"


def parse_percent(value: str) -> float:
    return float(value) / 100.0


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def load_npz_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return arr["y_true"].astype(np.int64), arr["y_pred"].astype(np.int64)


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    keep: List[Dict[str, str]] = []
    for row in rows:
        if row.get("Train Dataset", "") not in SUPPORTED_TRAIN_DATASETS:
            continue
        keep.append(row)
    return keep


def stage_from_row(row: Dict[str, str]) -> Optional[str]:
    protocol = row.get("Protocol", "")
    init_type = row.get("Init Type", "")
    fewshot_k = int(row.get("Fewshot K", "0"))
    if protocol == "loto_source_to_target" and init_type == "zero_shot" and fewshot_k == 0:
        return "zero_shot"
    if protocol == "target_only" and init_type in {"transfer", "scratch"} and fewshot_k > 0:
        return init_type
    return None


def _metric_columns(metric_split: str) -> Tuple[str, str, str]:
    split = str(metric_split).strip().lower()
    if split == "val":
        return ("Top1 Val", "Top2 Val", "Balanced Acc Val")
    if split == "test":
        return ("Top1 Test", "Top2 Test", "Balanced Acc Test")
    raise ValueError(f"Unsupported metric_split={metric_split}. Use 'test' or 'val'.")


def summarize_rows(
    rows: List[Dict[str, str]],
    perspective: Optional[str],
    metric_split: str = "test",
) -> List[Dict[str, str]]:
    top1_col, top2_col, bal_col = _metric_columns(metric_split)
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if perspective is not None and row.get("Perspective", "") != perspective:
            continue
        stage = stage_from_row(row)
        if stage is None:
            continue
        key = (
            row.get("Backbone", ""),
            method_from_probe_name(row.get("Probe Name", "")),
            stage,
            int(row.get("Fewshot K", "0")),
        )
        grouped[key].append(row)

    out: List[Dict[str, str]] = []
    for (backbone, method, stage, fewshot_k), grp in sorted(grouped.items()):
        valid_grp = [r for r in grp if r.get(top1_col, "") != ""]
        if not valid_grp:
            continue
        top1 = np.array([parse_percent(r[top1_col]) for r in valid_grp], dtype=np.float64)
        top2 = np.array([parse_percent(r[top2_col]) for r in valid_grp], dtype=np.float64)
        bal = np.array([parse_percent(r[bal_col]) for r in valid_grp], dtype=np.float64)
        out.append(
            {
                "Perspective": perspective if perspective is not None else "macro",
                "Backbone": backbone,
                "Method": method,
                "Stage": stage,
                "Fewshot K": str(fewshot_k),
                "N": str(len(valid_grp)),
                "Top1 Mean": f"{top1.mean()*100:.4f}",
                "Top1 Std": f"{top1.std(ddof=0)*100:.4f}",
                "Top2 Mean": f"{top2.mean()*100:.4f}",
                "Top2 Std": f"{top2.std(ddof=0)*100:.4f}",
                "Balanced Acc Mean": f"{bal.mean()*100:.4f}",
                "Balanced Acc Std": f"{bal.std(ddof=0)*100:.4f}",
            }
        )
    return out


def bootstrap_delta_top1(
    paired_runs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        run_deltas: List[float] = []
        for y_true, y_pred_eff, y_pred_gap in paired_runs:
            n = y_true.shape[0]
            idx = rng.integers(0, n, size=n)
            te = top1_accuracy(y_true[idx], y_pred_eff[idx])
            tg = top1_accuracy(y_true[idx], y_pred_gap[idx])
            run_deltas.append(te - tg)
        diffs[b] = float(np.mean(run_deltas)) if run_deltas else 0.0
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


def run_significance(rows: List[Dict[str, str]], n_boot: int, seed: int) -> List[Dict[str, str]]:
    per_run: Dict[Tuple[str, str, str, int, str, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        stage = stage_from_row(row)
        if stage is None:
            continue
        key = (
            row.get("Backbone", ""),
            row.get("Perspective", ""),
            stage,
            int(row.get("Fewshot K", "0")),
            row.get("Holdout Environment", ""),
            int(row.get("Random Seed", "-1")),
        )
        per_run[key][method_from_probe_name(row.get("Probe Name", ""))] = row

    grouped_pairs: Dict[Tuple[str, str, str, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for (backbone, perspective, stage, fewshot_k, _holdout, _seed), methods in per_run.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        p_eff = Path(methods["EfficientProbing"].get("Predictions Path", ""))
        p_gap = Path(methods["GAP"].get("Predictions Path", ""))
        if not p_eff.exists() or not p_gap.exists():
            continue
        y_true_e, y_pred_e = load_npz_predictions(p_eff)
        y_true_g, y_pred_g = load_npz_predictions(p_gap)
        if y_true_e.shape != y_true_g.shape or np.any(y_true_e != y_true_g):
            continue
        grouped_pairs[(backbone, perspective, stage, fewshot_k)].append((y_true_e, y_pred_e, y_pred_g))

    out: List[Dict[str, str]] = []
    for (backbone, perspective, stage, fewshot_k), paired_runs in sorted(grouped_pairs.items()):
        if not paired_runs:
            continue
        stats = bootstrap_delta_top1(paired_runs, n_boot=n_boot, seed=seed)
        out.append(
            {
                "Backbone": backbone,
                "Perspective": perspective,
                "Stage": stage,
                "Fewshot K": str(fewshot_k),
                "Paired Runs": str(len(paired_runs)),
                "Delta Top1 Mean (Eff-GAP)": f"{stats['delta_mean']*100:.4f}",
                "CI Low 2.5%": f"{stats['ci_low']*100:.4f}",
                "CI High 97.5%": f"{stats['ci_high']*100:.4f}",
                "P Value": f"{stats['p_value']:.8f}",
            }
        )
    return out


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


def _table_to_map(rows: List[Dict[str, str]], perspective: str) -> Dict[Tuple[str, str, str, int], float]:
    m: Dict[Tuple[str, str, str, int], float] = {}
    for row in rows:
        if row.get("Perspective", "") != perspective:
            continue
        key = (
            row["Backbone"],
            row["Method"],
            row["Stage"],
            int(row["Fewshot K"]),
        )
        m[key] = float(row["Top1 Mean"])
    return m


def plot_curves(
    summary_rows: List[Dict[str, str]],
    output_dir: Path,
    perspective: str,
    metric_split: str = "test",
) -> None:
    table = _table_to_map(summary_rows, perspective)
    backbones = sorted({row["Backbone"] for row in summary_rows if row.get("Perspective", "") == perspective})
    ks = [0, 10, 25, 50, 100, 250, 500, 800]
    split = str(metric_split).strip().lower()
    if split not in {"test", "val"}:
        raise ValueError(f"Unsupported metric_split={metric_split}. Use 'test' or 'val'.")
    split_title = "Top-1" if split == "test" else "Val Top-1"
    split_file_tag = "top1" if split == "test" else "top1_val"
    for backbone in backbones:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for method, color in [("EfficientProbing", "tab:blue"), ("GAP", "tab:orange")]:
            zero = table.get((backbone, method, "zero_shot", 0), None)
            if zero is None:
                continue
            transfer_vals = [zero]
            scratch_vals = [zero]
            for k in ks[1:]:
                tv = table.get((backbone, method, "transfer", k), None)
                sv = table.get((backbone, method, "scratch", k), None)
                transfer_vals.append(tv if tv is not None else np.nan)
                scratch_vals.append(sv if sv is not None else np.nan)
            if np.any(np.isfinite(np.asarray(transfer_vals, dtype=np.float64))):
                ax.plot(ks, transfer_vals, marker="o", color=color, label=f"{method} transfer")
                plotted = True
            if np.any(np.isfinite(np.asarray(scratch_vals, dtype=np.float64))):
                ax.plot(ks, scratch_vals, marker="s", linestyle="--", color=color, label=f"{method} scratch")
                plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{split_title} Few-shot Curves | {perspective} | {backbone}")
        ax.set_xlabel("Few-shot K (includes K=0 zero-shot)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"curves_{split_file_tag}_{perspective}_{backbone}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Unreal LOTO + few-shot rebuttal results.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(args.results_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_camera = summarize_rows(rows, perspective="camera", metric_split="test")
    per_human = summarize_rows(rows, perspective="human", metric_split="test")
    per_macro = summarize_rows(rows, perspective=None, metric_split="test")
    per_camera_val = summarize_rows(rows, perspective="camera", metric_split="val")
    per_human_val = summarize_rows(rows, perspective="human", metric_split="val")
    per_macro_val = summarize_rows(rows, perspective=None, metric_split="val")
    significance = run_significance(rows, n_boot=args.n_boot, seed=args.seed)

    write_csv(args.output_dir / "summary_camera.csv", per_camera)
    write_csv(args.output_dir / "summary_human.csv", per_human)
    write_csv(args.output_dir / "summary_macro.csv", per_macro)
    write_csv(args.output_dir / "summary_camera_val.csv", per_camera_val)
    write_csv(args.output_dir / "summary_human_val.csv", per_human_val)
    write_csv(args.output_dir / "summary_macro_val.csv", per_macro_val)
    write_csv(args.output_dir / "significance_top1.csv", significance)

    combined = per_camera + per_human + per_macro
    combined_val = per_camera_val + per_human_val + per_macro_val
    plot_curves(combined, args.output_dir, perspective="camera", metric_split="test")
    plot_curves(combined, args.output_dir, perspective="human", metric_split="test")
    plot_curves(combined, args.output_dir, perspective="macro", metric_split="test")
    plot_curves(combined_val, args.output_dir, perspective="camera", metric_split="val")
    plot_curves(combined_val, args.output_dir, perspective="human", metric_split="val")
    plot_curves(combined_val, args.output_dir, perspective="macro", metric_split="val")


if __name__ == "__main__":
    main()
