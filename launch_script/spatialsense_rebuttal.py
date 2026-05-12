from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


BACKBONES: Dict[str, str] = {
    "vit_small": "vit_small_patch16_dinov3.lvd1689m",
    "vit_base": "vit_base_patch16_dinov3.lvd1689m",
    "vit_large": "vit_large_patch16_dinov3.lvd1689m",
}

HEADS: Dict[str, Dict[str, str]] = {
    "EfficientProbing": {
        "probe_target": "evals.models.probes.EfficientProbing",
        "return_cls": "false",
        "mean_pool": "false",
        "efficient_probe": "true",
    },
    "GAP": {
        "probe_target": "evals.models.probes.ClassificationHead",
        "return_cls": "false",
        "mean_pool": "true",
        "efficient_probe": "false",
    },
}

PRIMARY_INPUT_MODES = ["pair_union_crop"]
SECONDARY_COMBOS = [("vit_base", "full_image")]


def method_from_probe_name(probe_name: str) -> str:
    probe_name = probe_name.lower()
    if "efficient" in probe_name:
        return "EfficientProbing"
    return "GAP"


def results_csv_path(output_dir: Path) -> Path:
    return (
        output_dir
        / "position_between_objects"
        / "spatialsense_position"
        / "position_between_objects_results_spatialsense_position.csv"
    )


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {rendered}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def find_matching_row(
    rows: List[Dict[str, str]],
    model_name: str,
    method: str,
    input_mode: str,
    seed: int,
    lr: float,
    wd: float,
    dropout: float,
) -> Dict[str, str]:
    def _model_matches(row_model: str, requested_model: str) -> bool:
        return row_model == requested_model or row_model.endswith(requested_model)

    for row in reversed(rows):
        try:
            row_seed = int(row.get("Random Seed", "-1"))
            row_model = row.get("Model Checkpoint", "")
            row_mode = row.get("Input Mode", "")
            row_lr = float(row.get("Probe LR", "nan"))
            row_wd = float(row.get("Weight Decay", "nan"))
            row_dropout = float(row.get("Dropout Rate", "nan"))
            row_method = method_from_probe_name(row.get("Probe Name", ""))
        except ValueError:
            continue
        if (
            row_seed == seed
            and _model_matches(row_model, model_name)
            and row_mode == input_mode
            and abs(row_lr - lr) < 1e-12
            and abs(row_wd - wd) < 1e-12
            and abs(row_dropout - dropout) < 1e-12
            and row_method == method
        ):
            return row
    raise RuntimeError(
        f"Could not find row for model={model_name}, method={method}, mode={input_mode}, "
        f"seed={seed}, lr={lr}, wd={wd}, dropout={dropout}."
    )


def build_train_command(
    model_name: str,
    head_name: str,
    input_mode: str,
    seed: int,
    lr: float,
    wd: float,
    dropout: float,
    n_epochs: int,
    warmup_epochs: float,
    batch_size: int,
    output_dir: Path,
    experiment_suffix: str,
) -> List[str]:
    head_cfg = HEADS[head_name]
    return [
        "python",
        "train_position_between_objects_with_cache.py",
        "dataset=spatialsense_position",
        "backbone=dinov3_timm",
        f"backbone.model_name={model_name}",
        f"probe._target_={head_cfg['probe_target']}",
        f"probe.dropout_rate={dropout}",
        "probe.num_classes=4",
        f"backbone.return_cls={head_cfg['return_cls']}",
        f"backbone.mean_pool={head_cfg['mean_pool']}",
        f"backbone.efficient_probe={head_cfg['efficient_probe']}",
        "backbone.layer=-1",
        f"optimizer.n_epochs={n_epochs}",
        f"optimizer.warmup_epochs={warmup_epochs}",
        f"optimizer.probe_lr={lr}",
        f"optimizer.weight_decay={wd}",
        f"batch_size={batch_size}",
        f"system.random_seed={seed}",
        "environment=spatialsense",
        "experiment_model=dinov3_timm",
        f"experiment_name=SpatialSense_Rebuttal_{experiment_suffix}",
        f"dataset.input_mode={input_mode}",
        f"dataset.perspective={input_mode}",
        "dataset.crop_padding_ratio=0.10",
        "visualization.enable_perspective_divergence=false",
        "visualization.sample_number=20",
        "visualization.correctly_classified=true",
        f"output_dir={output_dir.as_posix()}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpatialSense rebuttal matrix for SpaRRTa.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result_spatialsense_rebuttal"),
        help="Hydra output_dir override (relative to project-dir if not absolute).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-secondary", action="store_true", default=True)
    parser.add_argument("--skip-secondary", action="store_true")
    parser.add_argument("--rerun-seed8-final", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()

    seeds = [8, 42, 123]
    lr_grid = [1e-3, 5e-4]
    wd_grid = [1e-3, 1e-4]
    dropout_grid = [0.0, 0.5]

    combos: List[Tuple[str, str, str]] = []
    for backbone_key in BACKBONES.keys():
        for head_name in HEADS.keys():
            for input_mode in PRIMARY_INPUT_MODES:
                combos.append((backbone_key, head_name, input_mode))
    if args.run_secondary and not args.skip_secondary:
        for backbone_key, input_mode in SECONDARY_COMBOS:
            for head_name in HEADS.keys():
                combos.append((backbone_key, head_name, input_mode))

    best_params: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    csv_path = results_csv_path(output_dir)

    # Phase 1: tuning on seed=8
    for backbone_key, head_name, input_mode in combos:
        model_name = BACKBONES[backbone_key]
        best_val = -1.0
        best_cfg: Dict[str, float] = {}
        for lr in lr_grid:
            for wd in wd_grid:
                for dropout in dropout_grid:
                    cmd = build_train_command(
                        model_name=model_name,
                        head_name=head_name,
                        input_mode=input_mode,
                        seed=8,
                        lr=lr,
                        wd=wd,
                        dropout=dropout,
                        n_epochs=args.epochs,
                        warmup_epochs=args.warmup_epochs,
                        batch_size=args.batch_size,
                        output_dir=output_dir,
                        experiment_suffix=f"{backbone_key}_{head_name}_{input_mode}_tune",
                    )
                    run_command(cmd, cwd=project_dir, dry_run=args.dry_run)
                    if args.dry_run:
                        continue
                    rows = read_csv_rows(csv_path)
                    row = find_matching_row(
                        rows=rows,
                        model_name=model_name,
                        method=head_name,
                        input_mode=input_mode,
                        seed=8,
                        lr=lr,
                        wd=wd,
                        dropout=dropout,
                    )
                    val_bal = float(row["Balanced Acc Val"])
                    if val_bal > best_val:
                        best_val = val_bal
                        best_cfg = {"lr": lr, "wd": wd, "dropout": dropout}
        if args.dry_run:
            best_cfg = {"lr": lr_grid[0], "wd": wd_grid[0], "dropout": dropout_grid[0]}
        elif not best_cfg:
            raise RuntimeError(f"No tuning result collected for {(backbone_key, head_name, input_mode)}")
        best_params[(backbone_key, head_name, input_mode)] = best_cfg
        print(
            f"[TUNE] best {backbone_key} | {head_name} | {input_mode} -> "
            f"{best_cfg} (val_bal={best_val:.2f})"
        )

    if not args.dry_run:
        best_path = output_dir / "spatialsense_rebuttal_best_params.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            f"{k[0]}::{k[1]}::{k[2]}": v for k, v in best_params.items()
        }
        with best_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[SAVE] best hyperparameters -> {best_path}")

    # Phase 2: fixed runs for seeds 42/123 (+ optional 8 rerun)
    target_seeds = seeds if args.rerun_seed8_final else [42, 123]
    for backbone_key, head_name, input_mode in combos:
        model_name = BACKBONES[backbone_key]
        cfg = best_params[(backbone_key, head_name, input_mode)]
        for seed in target_seeds:
            cmd = build_train_command(
                model_name=model_name,
                head_name=head_name,
                input_mode=input_mode,
                seed=seed,
                lr=float(cfg["lr"]),
                wd=float(cfg["wd"]),
                dropout=float(cfg["dropout"]),
                n_epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                output_dir=output_dir,
                experiment_suffix=f"{backbone_key}_{head_name}_{input_mode}_final",
            )
            run_command(cmd, cwd=project_dir, dry_run=args.dry_run)

    report_cmd = [
        "python",
        "scripts/summarize_spatialsense_rebuttal.py",
        "--results-csv",
        str(csv_path),
        "--output-dir",
        str(output_dir / "position_between_objects" / "spatialsense_position" / "reports"),
    ]
    run_command(report_cmd, cwd=project_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
