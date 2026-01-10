from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LayerStats = Dict[int, Dict[str, float]]
ModelSummary = Dict[str, Dict[Tuple[str, str], LayerStats]]


def discover_env_dirs(root: Path, include: List[str] | None = None) -> List[Path]:
    if include:
        dirs = [root / name for name in include]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs = [p for p in dirs if p.exists() and p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No environment directories found under {root}")
    return sorted(dirs)


def load_environment(csv_dir: Path) -> Dict[str, Dict[Tuple[str, str], Dict[int, float]]]:
    models: Dict[str, Dict[Tuple[str, str], Dict[int, float]]] = {}
    for csv_path in csv_dir.glob("*_object_attention.csv"):
        model = csv_path.stem.replace("_object_attention", "")
        records: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                query = row["query"]
                target = row["target"]
                value = float(row["mean_attention"])
                records[(query, target)][layer] = value
        if not records:
            continue
        models[model] = records
    if not models:
        raise FileNotFoundError(f"No *_object_attention.csv files found in {csv_dir}")
    return models


def aggregate_environments(env_dirs: List[Path]) -> Tuple[ModelSummary, List[str], List[int]]:
    env_data = []
    for env_dir in env_dirs:
        env_models = load_environment(env_dir)
        env_data.append(env_models)

    model_names = sorted(env_data[0].keys())
    for env_models in env_data:
        missing = set(model_names) - set(env_models.keys())
        if missing:
            raise ValueError(f"Missing models {missing} in environment {env_dirs[env_data.index(env_models)]}")

    # determine objects and layers from first environment/model
    first_model = next(iter(env_data[0]))
    combos = env_data[0][first_model].keys()
    objects = sorted({q for (q, _) in combos if q != "CLS"})
    layers = sorted({layer for stats in env_data[0][first_model].values() for layer in stats})

    aggregated: ModelSummary = defaultdict(dict)
    for model in model_names:
        combos = env_data[0][model].keys()
        for key in combos:
            all_layers = {}
            for layer in layers:
                values = []
                for env_models in env_data:
                    layer_map = env_models[model].get(key)
                    if layer_map is None or layer not in layer_map:
                        raise ValueError(
                            f"Missing data for model {model}, query {key[0]}, target {key[1]}, layer {layer}"
                        )
                    values.append(layer_map[layer])
                arr = np.array(values, dtype=np.float64)
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                se = float(std / math.sqrt(len(arr))) if len(arr) > 0 else 0.0
                ci = float(1.96 * se)
                all_layers[layer] = {
                    "mean": mean,
                    "std": std,
                    "se": se,
                    "ci95": ci,
                }
            aggregated[model][key] = all_layers
    return aggregated, objects, layers


def subplot_layout(n_panels: int, max_cols: int = 3) -> Tuple[int, int]:
    cols = min(max_cols, max(1, n_panels))
    rows = math.ceil(n_panels / cols)
    return rows, cols


def compute_bounds(stats: List[Dict[str, float]], interval: str) -> Tuple[List[float], List[float], List[float]]:
    means = [s["mean"] for s in stats]
    if interval == "std":
        offsets = [s["std"] for s in stats]
    elif interval == "se":
        offsets = [s["se"] for s in stats]
    else:  # ci95
        offsets = [s["ci95"] for s in stats]
    lowers = [max(0.0, m - o) for m, o in zip(means, offsets)]
    uppers = [min(1.0, m + o) for m, o in zip(means, offsets)]
    return means, lowers, uppers


def style_interval(ax, layers, means, lowers, uppers, label, color, log_scale: bool = False):
    if log_scale:
        ax.set_yscale("log")
    ax.plot(layers, means, marker="o", color=color, label=label)
    ax.fill_between(layers, lowers, uppers, color=color, alpha=0.2)


def plot_cls_to_targets(models: ModelSummary, targets: List[str], layers: List[int], interval: str, output: Path, log_scale: bool = False):
    if not models:
        return
    model_names = sorted(models.keys())
    nrows, ncols = subplot_layout(len(model_names))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    colors = plt.cm.tab10.colors
    for idx, model in enumerate(model_names):
        ax = axes[idx // ncols, idx % ncols]
        for j, target in enumerate(targets + ["CLS"]):
            stats = [models[model][("CLS", target)][layer] for layer in layers]
            means, lowers, uppers = compute_bounds(stats, interval)
            label = f"CLS→{target}"
            style_interval(ax, layers, means, lowers, uppers, label, colors[j % len(colors)], log_scale=log_scale)
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean attention")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(targets) + 1, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_object_to_others(
    models: ModelSummary,
    objects: List[str],
    targets: List[str],
    layers: List[int],
    interval: str,
    output_dir: Path,
    log_scale: bool = False,
):
    colors = plt.cm.tab10.colors
    model_names = sorted(models.keys())
    nrows, ncols = subplot_layout(len(model_names))
    for obj in objects:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(nrows, ncols)
        for idx, model in enumerate(model_names):
            ax = axes[idx // ncols, idx % ncols]
            for j, target in enumerate(targets + ["CLS"]):
                stats = [models[model][(obj, target)][layer] for layer in layers]
                means, lowers, uppers = compute_bounds(stats, interval)
                style_interval(ax, layers, means, lowers, uppers, f"{obj}→{target}", colors[j % len(colors)], log_scale=log_scale)
            ax.set_title(model)
            ax.grid(True, alpha=0.3)
        for ax in axes[-1, :]:
            ax.set_xlabel("Layer")
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean attention")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(targets) + 1, frameon=False)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(output_dir / f"{obj}_to_others_agg.png", dpi=200)
        plt.close(fig)


def plot_self_attention(models: ModelSummary, objects: List[str], layers: List[int], interval: str, output: Path, log_scale: bool = False):
    model_names = sorted(models.keys())
    columns = objects + ["CLS"]
    nrows = len(model_names)
    ncols = len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    colors = plt.cm.Set1.colors
    for i, model in enumerate(model_names):
        for j, col in enumerate(columns):
            ax = axes[i, j]
            stats = [models[model][(col if col != "CLS" else "CLS", col)][layer] for layer in layers]
            means, lowers, uppers = compute_bounds(stats, interval)
            style_interval(ax, layers, means, lowers, uppers, f"{col}→{col}", colors[j % len(colors)], log_scale=log_scale)
            if i == 0:
                ax.set_title(col)
            if j == 0:
                ax.set_ylabel(f"{model}\nMean attention")
            else:
                ax.set_ylabel("")
            ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(columns), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_summary_csv(models: ModelSummary, layers: List[int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for model, combos in models.items():
        lines = ["model,layer,query,target,mean,std,se,ci95"]
        for (query, target), stats in combos.items():
            for layer in layers:
                s = stats[layer]
                lines.append(
                    f"{model},{layer},{query},{target},{s['mean']:.6f},{s['std']:.6f},{s['se']:.6f},{s['ci95']:.6f}"
                )
        (output_dir / f"{model}_object_attention_agg.csv").write_text("\n".join(lines))


def main():
    model_type = "large"
    parser = argparse.ArgumentParser(description="Aggregate object attention across environments.")
    parser.add_argument("--input-root", type=str, default=f"analysis/output_{model_type}", help="Parent directory containing environment subfolders")
    parser.add_argument("--output-dir", type=str, default=f"analysis/aggregated_{model_type}", help="Where to save aggregated plots/CSVs")
    parser.add_argument("--interval", choices=["std", "se", "ci95"], default="std")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for the y-axis")
    parser.add_argument("--background-name", type=str, default="Background")
    parser.add_argument("--environments", nargs="*", default=None, help="Specific environment folder names to include")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    env_dirs = discover_env_dirs(input_root, args.environments)
    models, objects, layers = aggregate_environments(env_dirs)
    base_targets = list(objects) + [args.background_name]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(models, layers, output_dir)
    plot_cls_to_targets(models, base_targets, layers, args.interval, output_dir / "cls_to_targets_agg.png", args.log_scale)
    plot_object_to_others(models, objects, base_targets, layers, args.interval, output_dir, args.log_scale)
    plot_self_attention(models, objects, layers, args.interval, output_dir / "objects_self_agg.png", args.log_scale)
    print(f"Aggregated results saved to {output_dir}")


if __name__ == "__main__":
    main()
