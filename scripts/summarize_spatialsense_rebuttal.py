from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LABEL_ORDER = ["Left", "Right", "Front", "Back"]
PREDICATE_TO_LABEL = {
    "to the left of": "Left",
    "to the right of": "Right",
    "in front of": "Front",
    "behind": "Back",
}


def method_from_probe_name(probe_name: str) -> str:
    if "efficient" in probe_name.lower():
        return "EfficientProbing"
    return "GAP"


def parse_percent(value: str) -> float:
    return float(value) / 100.0


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_order: List[int]) -> float:
    recalls = []
    for cls in class_order:
        mask = y_true == cls
        denom = int(mask.sum())
        if denom == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).sum()) / float(denom))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def bootstrap_delta_balanced_accuracy(
    paired_runs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seed_diffs = []
        for y_true, y_pred_eff, y_pred_gap, class_order in paired_runs:
            n = y_true.shape[0]
            sample_idx = rng.integers(0, n, size=n)
            yt = y_true[sample_idx]
            ye = y_pred_eff[sample_idx]
            yg = y_pred_gap[sample_idx]
            bal_eff = balanced_accuracy(yt, ye, class_order)
            bal_gap = balanced_accuracy(yt, yg, class_order)
            seed_diffs.append(bal_eff - bal_gap)
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


def load_npz_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    arr = np.load(path, allow_pickle=False)
    y_true = arr["y_true"].astype(np.int64)
    y_pred = arr["y_pred"].astype(np.int64)
    class_order = [int(v) for v in arr["class_order"].astype(np.int64).tolist()]
    return y_true, y_pred, class_order


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = []
    for row in rows:
        if row.get("Train Dataset", "") != "spatialsense_position":
            continue
        keep.append(row)
    return keep


def make_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = method_from_probe_name(row["Probe Name"])
        key = (row["Model Checkpoint"], row.get("Input Mode", "unknown"), method)
        groups[key].append(row)

    summary = []
    for (model, input_mode, method), grp in sorted(groups.items()):
        bal = np.array([parse_percent(r["Balanced Acc Test"]) for r in grp], dtype=np.float64)
        top1 = np.array([parse_percent(r["Top1 Test"]) for r in grp], dtype=np.float64)
        top2 = np.array([parse_percent(r["Top2 Test"]) for r in grp], dtype=np.float64)
        macro_f1 = np.array([parse_percent(r.get("Macro F1 Test", "0")) for r in grp], dtype=np.float64)
        seeds = sorted({int(r["Random Seed"]) for r in grp})
        summary.append(
            {
                "Model Checkpoint": model,
                "Input Mode": input_mode,
                "Method": method,
                "Seeds": ",".join(str(s) for s in seeds),
                "N": str(len(grp)),
                "Balanced Acc Mean": f"{bal.mean()*100:.2f}",
                "Balanced Acc Std": f"{bal.std(ddof=0)*100:.2f}",
                "Top1 Mean": f"{top1.mean()*100:.2f}",
                "Top1 Std": f"{top1.std(ddof=0)*100:.2f}",
                "Top2 Mean": f"{top2.mean()*100:.2f}",
                "Top2 Std": f"{top2.std(ddof=0)*100:.2f}",
                "Macro F1 Mean": f"{macro_f1.mean()*100:.2f}",
                "Macro F1 Std": f"{macro_f1.std(ddof=0)*100:.2f}",
            }
        )
    return summary


def run_significance(rows: List[Dict[str, str]], n_boot: int, seed: int) -> List[Dict[str, str]]:
    by_group_seed: Dict[Tuple[str, str, int], Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        method = method_from_probe_name(row["Probe Name"])
        key = (row["Model Checkpoint"], row.get("Input Mode", "unknown"), int(row["Random Seed"]))
        by_group_seed[key][method] = row

    grouped_pairs: Dict[Tuple[str, str], List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]] = defaultdict(list)
    for (model, input_mode, _seed), methods in by_group_seed.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        pred_eff = Path(methods["EfficientProbing"]["Predictions Path"])
        pred_gap = Path(methods["GAP"]["Predictions Path"])
        if not pred_eff.exists() or not pred_gap.exists():
            continue
        y_true_e, y_pred_e, class_order_e = load_npz_predictions(pred_eff)
        y_true_g, y_pred_g, class_order_g = load_npz_predictions(pred_gap)
        if y_true_e.shape != y_true_g.shape or np.any(y_true_e != y_true_g):
            continue
        class_order = class_order_e if class_order_e == class_order_g else sorted(set(class_order_e) | set(class_order_g))
        grouped_pairs[(model, input_mode)].append((y_true_e, y_pred_e, y_pred_g, class_order))

    results = []
    for (model, input_mode), paired_runs in sorted(grouped_pairs.items()):
        if not paired_runs:
            continue
        sig = bootstrap_delta_balanced_accuracy(paired_runs, n_boot=n_boot, seed=seed)
        results.append(
            {
                "Model Checkpoint": model,
                "Input Mode": input_mode,
                "Paired Seeds": str(len(paired_runs)),
                "Delta Balanced Acc Mean": f"{sig['delta_mean']*100:.2f}",
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


def plot_balanced_accuracy(summary_rows: List[Dict[str, str]], output_path: Path) -> None:
    grouped = defaultdict(dict)
    for row in summary_rows:
        key = f"{row['Model Checkpoint']} | {row['Input Mode']}"
        grouped[key][row["Method"]] = row

    labels = []
    eff_means = []
    eff_stds = []
    gap_means = []
    gap_stds = []
    for key, methods in grouped.items():
        if "EfficientProbing" not in methods or "GAP" not in methods:
            continue
        labels.append(key)
        eff_means.append(float(methods["EfficientProbing"]["Balanced Acc Mean"]))
        eff_stds.append(float(methods["EfficientProbing"]["Balanced Acc Std"]))
        gap_means.append(float(methods["GAP"]["Balanced Acc Mean"]))
        gap_stds.append(float(methods["GAP"]["Balanced Acc Std"]))

    if not labels:
        return

    x = np.arange(len(labels), dtype=np.float64)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.4), 5))
    ax.bar(x - w / 2, eff_means, width=w, yerr=eff_stds, capsize=3, label="EfficientProbing")
    ax.bar(x + w / 2, gap_means, width=w, yerr=gap_stds, capsize=3, label="GAP/MeanPool")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title("SpatialSense: EfficientProbing vs GAP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def dataset_sanity_report(dataset_root: Path, output_dir: Path) -> None:
    ann_path = dataset_root / "annotations.json"
    if not ann_path.exists():
        return
    with ann_path.open("r") as f:
        records = json.load(f)

    split_counts = defaultdict(int)
    split_class_counts = defaultdict(lambda: defaultdict(int))
    dropped_multi = defaultdict(int)
    missing_images = defaultdict(int)

    for rec in records:
        split = rec.get("split", "unknown")
        filename = rec.get("url", "").rsplit("/", 1)[-1]
        image_path = None
        if filename:
            cands = [
                dataset_root / "images" / "flickr" / filename,
                dataset_root / "images" / "nyu" / filename,
            ]
            for c in cands:
                if c.exists():
                    image_path = c
                    break
        if image_path is None:
            missing_images[split] += 1
            continue

        grouped = defaultdict(set)
        for ann in rec.get("annotations", []):
            pred = ann.get("predicate", "")
            if pred not in PREDICATE_TO_LABEL:
                continue
            if not bool(ann.get("label", False)):
                continue
            subj = ann.get("subject", {})
            obj = ann.get("object", {})
            key = (
                subj.get("name", ""),
                tuple(subj.get("bbox", [])),
                obj.get("name", ""),
                tuple(obj.get("bbox", [])),
            )
            grouped[key].add(PREDICATE_TO_LABEL[pred])

        for classes in grouped.values():
            if len(classes) == 0:
                continue
            if len(classes) > 1:
                dropped_multi[split] += 1
                continue
            cls_name = sorted(classes)[0]
            split_counts[split] += 1
            split_class_counts[split][cls_name] += 1

    rows = []
    for split in sorted(split_counts.keys() | missing_images.keys()):
        rows.append(
            {
                "Split": split,
                "Usable Samples": str(split_counts[split]),
                "Dropped Multi Positive": str(dropped_multi[split]),
                "Missing Images": str(missing_images[split]),
                "Left": str(split_class_counts[split]["Left"]),
                "Right": str(split_class_counts[split]["Right"]),
                "Front": str(split_class_counts[split]["Front"]),
                "Back": str(split_class_counts[split]["Back"]),
            }
        )
    write_csv(output_dir / "dataset_sanity.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SpatialSense rebuttal runs.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/shared/sets/datasets/vision/SpatialSense"),
    )
    args = parser.parse_args()

    rows = read_rows(args.results_csv)
    if not rows:
        raise RuntimeError(f"No SpatialSense rows found in {args.results_csv}")

    summary_rows = make_summary(rows)
    sig_rows = run_significance(rows, n_boot=args.n_bootstrap, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_mean_std.csv", summary_rows)
    write_csv(args.output_dir / "significance_bootstrap.csv", sig_rows)
    plot_balanced_accuracy(summary_rows, args.output_dir / "balanced_accuracy_comparison.png")
    dataset_sanity_report(args.dataset_root, args.output_dir)

    report_path = args.output_dir / "report.md"
    with report_path.open("w") as f:
        f.write("# SpatialSense Rebuttal Report\n\n")
        f.write("## Summary Files\n")
        f.write("- `summary_mean_std.csv`\n")
        f.write("- `significance_bootstrap.csv`\n")
        f.write("- `dataset_sanity.csv`\n")
        f.write("- `balanced_accuracy_comparison.png`\n")
    print(f"[DONE] Wrote report artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
