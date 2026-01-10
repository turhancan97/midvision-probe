from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LayerSeries = Dict[int, float]
ModelData = Dict[str, Dict[Tuple[str, str], LayerSeries]]


def find_csv_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("*_object_attention.csv"))


def load_models(files: List[Path]) -> Tuple[ModelData, List[str], List[int]]:
    models: ModelData = {}
    object_set = None
    layers_set = None

    for csv_path in files:
        model_records: Dict[Tuple[str, str], LayerSeries] = defaultdict(dict)
        current_model = None
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                layer = int(row["layer"])
                query = row["query"]
                target = row["target"]
                value = float(row["mean_attention"])
                current_model = model
                model_records[(query, target)][layer] = value
        if current_model is None:
            raise ValueError(f"No records found in {csv_path}")
        models[current_model] = model_records

        queries = sorted({q for (q, _) in model_records.keys() if q != "CLS"})
        if object_set is None:
            object_set = queries
        elif object_set != queries:
            raise ValueError(f"Object sets differ between CSVs. Expected {object_set}, got {queries} in {csv_path}")

        layers = sorted({layer for series in model_records.values() for layer in series})
        if layers_set is None:
            layers_set = layers
        elif layers_set != layers:
            raise ValueError(f"Layer sets differ between CSVs. Expected {layers_set}, got {layers} in {csv_path}")

    if object_set is None or layers_set is None:
        raise ValueError("No valid CSV files found.")
    return models, object_set, layers_set


def ensure_series(model: str, data: ModelData, query: str, target: str, layers: List[int]) -> List[float]:
    series = []
    key = (query, target)
    if key not in data[model]:
        raise ValueError(f"Missing attention data for model '{model}' query '{query}' target '{target}'")
    layer_map = data[model][key]
    for layer in layers:
        if layer not in layer_map:
            raise ValueError(f"Missing layer {layer} for model '{model}', query '{query}', target '{target}'")
        series.append(layer_map[layer])
    return series


def subplot_layout(n_panels: int, max_cols: int = 3) -> Tuple[int, int]:
    cols = min(max_cols, n_panels)
    rows = math.ceil(n_panels / cols)
    return rows, cols


def style_series(ax, layers, values, label, color=None, linestyle="-", log_scale: bool = False):
    if log_scale:
        ax.set_yscale("log")
    ax.plot(layers, values, marker="o", label=label, color=color, linestyle=linestyle)


def plot_cls_to_targets(models: ModelData, targets: List[str], layers: List[int], output: Path, log_scale: bool = False):
    model_names = sorted(models.keys())
    nrows, ncols = subplot_layout(len(model_names))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    colors = plt.cm.tab10.colors

    for idx, model in enumerate(model_names):
        ax = axes[idx // ncols, idx % ncols]
        for j, target in enumerate(targets):
            series = ensure_series(model, models, "CLS", target, layers)
            style_series(ax, layers, series, f"CLS→{target}", color=colors[j % len(colors)], log_scale=log_scale)
        ax.set_title(model)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean attention")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=len(targets), bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_object_to_others(models: ModelData, objects: List[str], targets: List[str], layers: List[int], output_dir: Path, log_scale: bool = False):
    colors = plt.cm.tab10.colors
    model_names = sorted(models.keys())
    nrows, ncols = subplot_layout(len(model_names))
    for obj in objects:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(nrows, ncols)
        for idx, model in enumerate(model_names):
            ax = axes[idx // ncols, idx % ncols]
            for j, target in enumerate(targets):
                series = ensure_series(model, models, obj, target, layers)
                style_series(ax, layers, series, f"{obj}→{target}", color=colors[j % len(colors)], log_scale=log_scale)
            ax.set_title(model)
            ax.grid(True, alpha=0.3)
        for ax in axes[-1, :]:
            ax.set_xlabel("Layer")
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean attention")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(targets), bbox_to_anchor=(0.5, 0), frameon=False)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(output_dir / f"{obj}_to_others.png", dpi=200)
        plt.close(fig)


def plot_self_attention(models: ModelData, objects: List[str], layers: List[int], output: Path, log_scale: bool = False):
    model_names = sorted(models.keys())
    columns = objects + ["CLS"]
    nrows = len(model_names)
    ncols = len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for i, model in enumerate(model_names):
        for j, col in enumerate(columns):
            ax = axes[i, j]
            if col == "CLS":
                series = ensure_series(model, models, "CLS", "CLS", layers)
                label = "CLS→CLS"
            else:
                series = ensure_series(model, models, col, col, layers)
                label = f"{col}→{col}"
            style_series(ax, layers, series, label, log_scale=log_scale)
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
    # fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    environment = "winter_town"
    model_type = "large"
    parser = argparse.ArgumentParser(description="Plot object attention grids across models.")
    parser.add_argument("--input-dir", type=str, default=f"analysis/output_{model_type}/{environment}")
    parser.add_argument("--output-dir", type=str, default=f"analysis/output_{model_type}/{environment}/object_grids")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for the y-axis")
    parser.add_argument("--background-name", type=str, default="Background")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_files = find_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No *_object_attention.csv files found in {input_dir}")

    models, objects, layers = load_models(csv_files)
    base_targets = list(objects) + [args.background_name]

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cls_to_targets(models, base_targets, layers, output_dir / "cls_to_targets.png", args.log_scale)
    plot_object_to_others(models, objects, base_targets + ["CLS"], layers, output_dir, args.log_scale)
    plot_self_attention(models, objects, layers, output_dir / "objects_self.png", args.log_scale)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
