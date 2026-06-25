from __future__ import annotations

import argparse
import csv
import io
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


REAL_WORLD_DATASET_NAME = "real_world_position"
REAL_WORLD_DATASET_ROOT = (
    "/shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images"
)

BACKBONE_CHOICES: List[str] = [
    "dino_b16",
    "dinov2_b14",
    "dinov2_b14_reg",
    "dinov2_l14_reg",
    "dinov3_b16",
    "dinov3_timm",
    "croco_b16",
    "crocov2_b16",
    "mae_b16",
    "maskfeat_vitb16",
    "spa_b16",
    "spa_l16",
    "vggt_l16",
    "deit3_b16",
    "clip_b16_laion",
]

HEADS: Dict[str, Dict[str, str]] = {
    "EfficientProbing": {
        "probe_target": "evals.models.probes.EfficientProbing",
        "return_cls": "false",
        "mean_pool": "false",
        "efficient_probe": "true",
    },
    "ABMILP": {
        "probe_target": "evals.models.probes.ABMILPHead",
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

SPLIT_MODE_CHOICES = ["random", "time_series", "clip_block_random", "clip_block_time_series"]


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_float_csv(raw: str) -> List[float]:
    values = []
    for part in parse_csv_list(raw):
        values.append(float(part))
    return values


def parse_int_csv(raw: str) -> List[int]:
    values = []
    for part in parse_csv_list(raw):
        values.append(int(part))
    return values


def parse_selection(raw: str, allowed: List[str], name: str) -> List[str]:
    selected = parse_csv_list(raw)
    if not selected:
        raise ValueError(f"No values provided for {name}.")
    invalid = [x for x in selected if x not in allowed]
    if invalid:
        raise ValueError(f"Invalid {name}: {invalid}. Allowed: {allowed}")
    return selected


def method_from_probe_name(probe_name: str) -> str:
    probe_name = str(probe_name).lower().strip()
    if "efficient" in probe_name:
        return "EfficientProbing"
    if "abmilp" in probe_name:
        return "ABMILP"
    return "GAP"


def results_csv_path(output_dir: Path) -> Path:
    return (
        output_dir
        / "position_between_objects"
        / REAL_WORLD_DATASET_NAME
        / f"position_between_objects_results_{REAL_WORLD_DATASET_NAME}.csv"
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
    raw = csv_path.read_bytes()
    if not raw:
        return []
    nul_count = raw.count(b"\x00")
    if nul_count:
        raw = raw.replace(b"\x00", b"")
        print(f"[WARN] Removed {nul_count} NUL byte(s) from {csv_path} before CSV parsing.")
    text = raw.decode("utf-8", errors="replace")
    with io.StringIO(text, newline="") as f:
        return list(csv.DictReader(f))


def find_matching_row(
    rows: List[Dict[str, str]],
    backbone: str,
    method: str,
    split_mode: str,
    seed: int,
    lr: float,
    wd: float,
    dropout: float,
) -> Dict[str, str]:
    for row in reversed(rows):
        try:
            if row.get("Train Dataset", "") != REAL_WORLD_DATASET_NAME:
                continue
            row_seed = int(row.get("Random Seed", "-1"))
            row_backbone = row.get("Backbone", "")
            row_split_mode = row.get("Split Mode", "") or row.get("Perspective", "")
            row_lr = float(row.get("Probe LR", "nan"))
            row_wd = float(row.get("Weight Decay", "nan"))
            row_dropout = float(row.get("Dropout Rate", "nan"))
            row_method = method_from_probe_name(row.get("Probe Name", ""))
        except ValueError:
            continue
        if (
            row_seed == seed
            and row_backbone == backbone
            and row_method == method
            and row_split_mode == split_mode
            and abs(row_lr - lr) < 1e-12
            and abs(row_wd - wd) < 1e-12
            and abs(row_dropout - dropout) < 1e-12
        ):
            return row
    raise RuntimeError(
        f"Could not find row for backbone={backbone}, method={method}, split_mode={split_mode}, "
        f"seed={seed}, lr={lr}, wd={wd}, dropout={dropout}."
    )


def build_train_command(
    backbone: str,
    head_name: str,
    split_mode: str,
    seed: int,
    lr: float,
    wd: float,
    dropout: float,
    n_epochs: int,
    warmup_epochs: float,
    batch_size: int,
    sample_number: int,
    clip_block_size: int,
    clip_id_source: str,
    output_dir: Path,
    experiment_suffix: str,
) -> List[str]:
    head_cfg = HEADS[head_name]
    return [
        "python",
        "train_position_between_objects_with_cache.py",
        f"dataset={REAL_WORLD_DATASET_NAME}",
        f"backbone={backbone}",
        f"experiment_model={backbone}",
        f"experiment_name=RealWorld_Rebuttal_{experiment_suffix}",
        f"probe._target_={head_cfg['probe_target']}",
        "probe.num_classes=4",
        f"probe.dropout_rate={dropout}",
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
        "environment=real_world_images",
        f"dataset.split_mode={split_mode}",
        f"dataset.clip_block_size={clip_block_size}",
        f"dataset.clip_id_source={clip_id_source}",
        "dataset.perspective=lego_human",
        "visualization.enable_perspective_divergence=false",
        f"visualization.sample_number={sample_number}",
        "visualization.correctly_classified=true",
        f"output_dir={output_dir.as_posix()}",
    ]


def maybe_run_train(
    *,
    cmd: List[str],
    dry_run: bool,
    skip_existing: bool,
    csv_path: Path,
    backbone: str,
    head_name: str,
    split_mode: str,
    seed: int,
    lr: float,
    wd: float,
    dropout: float,
    project_dir: Path,
) -> None:
    if skip_existing and not dry_run:
        rows = read_csv_rows(csv_path)
        try:
            _ = find_matching_row(
                rows=rows,
                backbone=backbone,
                method=head_name,
                split_mode=split_mode,
                seed=seed,
                lr=lr,
                wd=wd,
                dropout=dropout,
            )
            print(
                f"[SKIP] Found existing row for backbone={backbone}, head={head_name}, "
                f"split_mode={split_mode}, seed={seed}, lr={lr}, wd={wd}, dropout={dropout}"
            )
            return
        except RuntimeError:
            pass
    run_command(cmd, cwd=project_dir, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-world rebuttal matrix (replaces SpatialSense pipeline)."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result_real_world_rebuttal"),
        help="Hydra output_dir override (relative to project-dir if not absolute).",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default=",".join(BACKBONE_CHOICES),
        help="Comma-separated backbone config keys.",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="EfficientProbing,ABMILP,GAP",
        help="Comma-separated heads: EfficientProbing,ABMILP,GAP",
    )
    parser.add_argument(
        "--split-modes",
        type=str,
        default="random,time_series",
        help=(
            "Comma-separated split modes: random,time_series,"
            "clip_block_random,clip_block_time_series"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--rerun-seed8-final",
        action="store_true",
        default=False,
        help=(
            "Backward-compatible flag: include tune seed in final stage. "
            "Now applies to whichever tune seed is selected."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="8,42,123",
        help="Comma-separated seeds for this run (example: 8 or 8,42,123).",
    )
    parser.add_argument(
        "--tune-seed",
        type=int,
        default=None,
        help=(
            "Seed used for hyperparameter tuning stage. "
            "Default: first value from --seeds."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-number", type=int, default=20)
    parser.add_argument("--clip-block-size", type=int, default=20)
    parser.add_argument(
        "--clip-id-source",
        type=str,
        default="filename_numeric",
        help="Block grouping source: filename_numeric or parent_folder",
    )
    parser.add_argument("--lr-grid", type=str, default="1e-4,5e-4")
    parser.add_argument("--wd-grid", type=str, default="1e-2,1e-3")
    parser.add_argument("--dropout-grid", type=str, default="0.0,0.2,0.4,0.6")
    args = parser.parse_args()
    args.clip_id_source = str(args.clip_id_source).strip().lower()
    if args.clip_id_source not in {"filename_numeric", "parent_folder"}:
        raise ValueError(
            f"Invalid --clip-id-source={args.clip_id_source}. "
            "Use filename_numeric or parent_folder."
        )

    selected_backbones = parse_selection(args.backbones, BACKBONE_CHOICES, "backbones")
    selected_heads = parse_selection(args.heads, list(HEADS.keys()), "heads")
    selected_split_modes = parse_selection(args.split_modes, SPLIT_MODE_CHOICES, "split-modes")

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()

    seeds = parse_int_csv(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided. Use --seeds.")
    tune_seed = int(args.tune_seed) if args.tune_seed is not None else int(seeds[0])
    if tune_seed not in seeds:
        raise ValueError(
            f"--tune-seed ({tune_seed}) must be included in --seeds ({seeds})."
        )
    lr_grid = parse_float_csv(args.lr_grid)
    wd_grid = parse_float_csv(args.wd_grid)
    dropout_grid = parse_float_csv(args.dropout_grid)

    combos: List[Tuple[str, str, str]] = []
    for backbone in selected_backbones:
        for head_name in selected_heads:
            for split_mode in selected_split_modes:
                combos.append((backbone, head_name, split_mode))

    print("Selected matrix:")
    print(f"  backbones: {selected_backbones}")
    print(f"  heads: {selected_heads}")
    print(f"  split_modes: {selected_split_modes}")
    print(f"  seeds: {seeds}")
    print(f"  tune_seed: {tune_seed}")
    print(f"  lr_grid: {lr_grid}")
    print(f"  wd_grid: {wd_grid}")
    print(f"  dropout_grid: {dropout_grid}")
    print(f"  clip_block_size: {args.clip_block_size}")
    print(f"  clip_id_source: {args.clip_id_source}")

    best_params: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    csv_path = results_csv_path(output_dir)

    # Phase 1: tune on selected tune seed
    for backbone, head_name, split_mode in combos:
        best_val = -1.0
        best_cfg: Dict[str, float] = {}
        for lr in lr_grid:
            for wd in wd_grid:
                for dropout in dropout_grid:
                    cmd = build_train_command(
                        backbone=backbone,
                        head_name=head_name,
                        split_mode=split_mode,
                        seed=tune_seed,
                        lr=lr,
                        wd=wd,
                        dropout=dropout,
                        n_epochs=args.epochs,
                        warmup_epochs=args.warmup_epochs,
                        batch_size=args.batch_size,
                        sample_number=args.sample_number,
                        clip_block_size=args.clip_block_size,
                        clip_id_source=args.clip_id_source,
                        output_dir=output_dir,
                        experiment_suffix=f"{backbone}_{head_name}_{split_mode}_tune",
                    )
                    maybe_run_train(
                        cmd=cmd,
                        dry_run=args.dry_run,
                        skip_existing=args.skip_existing,
                        csv_path=csv_path,
                        backbone=backbone,
                        head_name=head_name,
                        split_mode=split_mode,
                        seed=tune_seed,
                        lr=lr,
                        wd=wd,
                        dropout=dropout,
                        project_dir=project_dir,
                    )
                    if args.dry_run:
                        continue
                    rows = read_csv_rows(csv_path)
                    row = find_matching_row(
                        rows=rows,
                        backbone=backbone,
                        method=head_name,
                        split_mode=split_mode,
                        seed=tune_seed,
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
            raise RuntimeError(f"No tuning result collected for {(backbone, head_name, split_mode)}")

        best_params[(backbone, head_name, split_mode)] = best_cfg
        print(
            f"[TUNE] best {backbone} | {head_name} | {split_mode} -> "
            f"{best_cfg} (val_bal={best_val:.2f})"
        )

    if not args.dry_run:
        best_path = output_dir / "real_world_rebuttal_best_params.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            f"{k[0]}::{k[1]}::{k[2]}": v for k, v in best_params.items()
        }
        with best_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[SAVE] best hyperparameters -> {best_path}")

    # Phase 2: fixed runs for selected seeds.
    # For single-seed runs, include that seed by default.
    target_seeds = seeds if (args.rerun_seed8_final or len(seeds) == 1) else [s for s in seeds if s != tune_seed]
    if not target_seeds:
        print(
            "[INFO] No final-stage seeds selected after excluding tune seed. "
            "Use --rerun-seed8-final to run final stage on the tune seed as well."
        )
    for backbone, head_name, split_mode in combos:
        cfg = best_params[(backbone, head_name, split_mode)]
        for seed in target_seeds:
            cmd = build_train_command(
                backbone=backbone,
                head_name=head_name,
                split_mode=split_mode,
                seed=seed,
                lr=float(cfg["lr"]),
                wd=float(cfg["wd"]),
                dropout=float(cfg["dropout"]),
                n_epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                sample_number=args.sample_number,
                clip_block_size=args.clip_block_size,
                clip_id_source=args.clip_id_source,
                output_dir=output_dir,
                experiment_suffix=f"{backbone}_{head_name}_{split_mode}_final",
            )
            maybe_run_train(
                cmd=cmd,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
                csv_path=csv_path,
                backbone=backbone,
                head_name=head_name,
                split_mode=split_mode,
                seed=seed,
                lr=float(cfg["lr"]),
                wd=float(cfg["wd"]),
                dropout=float(cfg["dropout"]),
                project_dir=project_dir,
            )

    if args.skip_summary:
        return

    report_cmd = [
        "python",
        "scripts/summarize_spatialsense_rebuttal.py",
        "--results-csv",
        str(csv_path),
        "--output-dir",
        str(output_dir / "position_between_objects" / REAL_WORLD_DATASET_NAME / "reports"),
        "--dataset-root",
        REAL_WORLD_DATASET_ROOT,
    ]
    run_command(report_cmd, cwd=project_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
