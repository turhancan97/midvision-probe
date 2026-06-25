from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import yaml


BACKBONES: List[str] = [
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
UNREAL_TRANSFER_DATASET_NAME = "unreal_position_transfer"
REAL_WORLD_HOLDOUT_DATASET_NAME = "unreal_real_holdout_transfer"
REAL_WORLD_HOLDOUT_ID = "real_world"
REAL_WORLD_DATASET_ROOT = (
    "/shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images"
)
REAL_WORLD_SPLIT_MODE_CHOICES = [
    "random",
    "time_series",
    "clip_block_random",
    "clip_block_time_series",
]
HEADS: Dict[str, Dict[str, str]] = {
    "EfficientProbing": {
        "probe_target": "evals.models.probes.EfficientProbing",
        "probe_type": "EfficientProbing",
        "return_cls": "false",
        "mean_pool": "false",
        "efficient_probe": "true",
    },
    "GAP": {
        "probe_target": "evals.models.probes.ClassificationHead",
        "probe_type": "ClassificationHead",
        "return_cls": "false",
        "mean_pool": "true",
        "efficient_probe": "false",
    },
}
PERSPECTIVES: List[str] = ["camera", "human"]
HOLDOUT_FOLDS: List[str] = ["bridge_2", "city_2", "desert_2", "forest_2", "winter_town_2"]
EXCLUDED_ENVS: List[str] = ["desert_nonhuman"]
SEEDS: List[int] = [8] # [8, 42, 123]
FEWSHOT_KS: List[int] = [10, 25, 50, 100, 250, 500, 800]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def validate_selection(selected: List[str], valid: List[str], field_name: str) -> None:
    invalid = [x for x in selected if x not in valid]
    if invalid:
        raise ValueError(
            f"Invalid {field_name}: {invalid}. Valid options: {valid}"
        )


def results_csv_path(output_dir: Path, dataset_name: str, result_csv_suffix: str = "") -> Path:
    suffix = str(result_csv_suffix).strip()
    suffix_part = f"_{suffix}" if suffix else ""
    return (
        output_dir
        / "position_between_objects"
        / dataset_name
        / f"position_between_objects_results_{dataset_name}{suffix_part}.csv"
    )


def method_from_probe_name(probe_name: str) -> str:
    if "efficient" in probe_name.lower():
        return "EfficientProbing"
    return "GAP"


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def find_matching_row(
    rows: List[Dict[str, str]],
    *,
    backbone: str,
    method: str,
    perspective: str,
    holdout: str,
    seed: int,
    protocol: str,
    init_type: str,
    fewshot_k: int,
) -> Optional[Dict[str, str]]:
    for row in reversed(rows):
        try:
            if row.get("Backbone", "") != backbone:
                continue
            if row.get("Perspective", "") != perspective:
                continue
            if row.get("Holdout Environment", "") != holdout:
                continue
            if row.get("Protocol", "") != protocol:
                continue
            if row.get("Init Type", "") != init_type:
                continue
            if int(row.get("Fewshot K", "-1")) != int(fewshot_k):
                continue
            if int(row.get("Random Seed", "-1")) != int(seed):
                continue
            if method_from_probe_name(row.get("Probe Name", "")) != method:
                continue
            return row
        except Exception:
            continue
    return None


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[RUN] {rendered}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def load_position_config(project_dir: Path) -> Dict:
    config_path = project_dir / "launch_script" / "position_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def get_hparams(cfg: Dict, perspective: str, probe_type: str) -> Dict[str, float]:
    p_cfg = cfg["hyperparameters"][perspective]
    h_cfg = p_cfg[probe_type]
    return {
        "ambiguity_degree": int(p_cfg["ambiguity_degree"]),
        "epochs": int(h_cfg["epochs"]),
        "warmup_epochs": float(h_cfg["warmup_epochs"]),
        "dropout_rate": float(cfg["probe"]["dropout_rate"]),
    }


def build_train_command(
    *,
    dataset_key: str,
    backbone: str,
    method: str,
    perspective: str,
    holdout: str,
    seed: int,
    protocol: str,
    init_type: str,
    fewshot_k: int,
    train_subset_size: int,
    train_subset_stratified: bool,
    ckpt_path: str,
    save_head: bool,
    output_dir: Path,
    hparams: Dict[str, float],
    environment_subset: Optional[List[str]] = None,
    result_csv_suffix: str = "",
    real_world_split_mode: Optional[str] = None,
    real_world_dataset_root: Optional[str] = None,
) -> List[str]:
    head_cfg = HEADS[method]
    excluded = "[" + ",".join(EXCLUDED_ENVS) + "]"
    exp_name = (
        f"LOTO_{backbone}_{method}_{perspective}_{holdout}_seed{seed}"
        f"_{init_type}_k{fewshot_k}"
    )
    cmd = [
        "python",
        "train_position_between_objects_with_cache.py",
        f"dataset={dataset_key}",
        f"backbone={backbone}",
        f"experiment_model={backbone}",
        f"experiment_name={exp_name}",
        f"probe._target_={head_cfg['probe_target']}",
        "probe.num_classes=4",
        f"probe.dropout_rate={hparams['dropout_rate']}",
        f"backbone.return_cls={head_cfg['return_cls']}",
        f"backbone.mean_pool={head_cfg['mean_pool']}",
        f"backbone.efficient_probe={head_cfg['efficient_probe']}",
        "backbone.layer=-1",
        f"optimizer.n_epochs={hparams['epochs']}",
        f"optimizer.warmup_epochs={hparams['warmup_epochs']}",
        f"dataset.perspective={perspective}",
        f"dataset.ambiguity_degrees={hparams['ambiguity_degree']}",
        f"dataset.protocol={protocol}",
        f"dataset.holdout_environment={holdout}",
        f"dataset.excluded_environments={excluded}",
        f"train_subset_size={train_subset_size}",
        f"train_subset_stratified={'true' if train_subset_stratified else 'false'}",
        f"system.random_seed={seed}",
        f"environment={holdout}",
        f"protocol={protocol}",
        f"holdout_environment={holdout}",
        f"init_type={init_type}",
        f"fewshot_k={fewshot_k}",
        f"result_csv_suffix={result_csv_suffix}",
        f"ckpt_path={ckpt_path}",
        f"save_head={'true' if save_head else 'false'}",
        "visualization.enable_perspective_divergence=false",
        "visualization.sample_number=0",
        f"output_dir={output_dir.as_posix()}",
    ]
    if dataset_key == REAL_WORLD_HOLDOUT_DATASET_NAME:
        if real_world_split_mode is None:
            raise ValueError("real_world_split_mode must be set for real-world holdout mode.")
        if real_world_dataset_root is None:
            raise ValueError("real_world_dataset_root must be set for real-world holdout mode.")
        cmd.extend(
            [
                f"dataset.real_world_root={real_world_dataset_root}",
                f"dataset.split_mode={real_world_split_mode}",
                "dataset.real_world_perspective=lego_human",
            ]
        )
    if environment_subset:
        env_subset = "[" + ",".join(environment_subset) + "]"
        cmd.append(f"dataset.environments={env_subset}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Unreal LOTO + few-shot rebuttal matrix.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result_unreal_loto_rebuttal"),
        help="Hydra output_dir override (relative to project-dir if not absolute).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--backbones",
        type=str,
        default=",".join(BACKBONES),
        help=f"Comma-separated backbones. Default: {','.join(BACKBONES)}",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default=",".join(HEADS.keys()),
        help=f"Comma-separated heads. Default: {','.join(HEADS.keys())}",
    )
    parser.add_argument(
        "--perspectives",
        type=str,
        default=",".join(PERSPECTIVES),
        help=f"Comma-separated perspectives. Default: {','.join(PERSPECTIVES)}",
    )
    parser.add_argument(
        "--holdout-folds",
        type=str,
        default=",".join(HOLDOUT_FOLDS),
        help=f"Comma-separated holdout folds. Default: {','.join(HOLDOUT_FOLDS)}",
    )
    parser.add_argument(
        "--pair-swap-mode",
        action="store_true",
        help="Run pair-swap transfer mode: source env -> target env only.",
    )
    parser.add_argument(
        "--pair-swap-source-env",
        type=str,
        default="desert_2",
        help="Source environment for pair-swap mode (train/val source in zero-shot).",
    )
    parser.add_argument(
        "--pair-swap-target-env",
        type=str,
        default="desert_3",
        help="Target environment for pair-swap mode (test in zero-shot; all splits in target_only).",
    )
    parser.add_argument(
        "--real-world-holdout-mode",
        action="store_true",
        help="Enable sim-to-real mode: Unreal source train/val with real-world holdout test.",
    )
    parser.add_argument(
        "--real-world-split-mode",
        type=str,
        default="time_series",
        help=(
            "Real-world split mode used in holdout mode: "
            "random,time_series,clip_block_random,clip_block_time_series."
        ),
    )
    parser.add_argument(
        "--real-world-dataset-root",
        type=str,
        default=REAL_WORLD_DATASET_ROOT,
        help="Root path of real-world dataset used in holdout mode.",
    )
    parser.add_argument(
        "--result-csv-suffix",
        type=str,
        default="",
        help="Optional suffix appended to result CSV filename for non-legacy datasets.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()

    pos_cfg = load_position_config(project_dir)

    selected_backbones = parse_csv_list(args.backbones)
    selected_heads = parse_csv_list(args.heads)
    selected_perspectives = parse_csv_list(args.perspectives)
    real_world_split_mode = str(args.real_world_split_mode).strip().lower()
    validate_selection(
        [real_world_split_mode],
        REAL_WORLD_SPLIT_MODE_CHOICES,
        "real-world split mode",
    )
    available_envs = sorted(list(pos_cfg["environments"].keys()))
    pair_swap_environment_subset: Optional[List[str]] = None
    result_csv_suffix = str(args.result_csv_suffix).strip()
    dataset_name = UNREAL_TRANSFER_DATASET_NAME
    if args.real_world_holdout_mode:
        if args.pair_swap_mode:
            raise ValueError("pair-swap mode cannot be combined with real-world holdout mode.")
        if sorted(set(selected_perspectives)) != ["human"]:
            raise ValueError(
                "real-world holdout mode supports only human perspective. "
                "Use --perspectives human."
            )
        selected_holdouts = [REAL_WORLD_HOLDOUT_ID]
        dataset_name = REAL_WORLD_HOLDOUT_DATASET_NAME
        if not result_csv_suffix:
            result_csv_suffix = "real_world_holdout"
    elif args.pair_swap_mode:
        source_env = str(args.pair_swap_source_env).strip()
        target_env = str(args.pair_swap_target_env).strip()
        validate_selection([source_env], available_envs, "pair-swap source env")
        validate_selection([target_env], available_envs, "pair-swap target env")
        selected_holdouts = [target_env]
        pair_swap_environment_subset = [source_env, target_env]
        if not result_csv_suffix:
            result_csv_suffix = "cross_triple"
    else:
        selected_holdouts = parse_csv_list(args.holdout_folds)

    csv_path = results_csv_path(
        output_dir,
        dataset_name=dataset_name,
        result_csv_suffix=result_csv_suffix,
    )

    validate_selection(selected_backbones, BACKBONES, "backbones")
    validate_selection(selected_heads, list(HEADS.keys()), "heads")
    validate_selection(selected_perspectives, PERSPECTIVES, "perspectives")
    if args.pair_swap_mode:
        validate_selection(selected_holdouts, available_envs, "pair-swap holdout folds")
    elif not args.real_world_holdout_mode:
        validate_selection(selected_holdouts, HOLDOUT_FOLDS, "holdout folds")

    print("Selected matrix:")
    print(f"  backbones: {selected_backbones}")
    print(f"  heads: {selected_heads}")
    print(f"  perspectives: {selected_perspectives}")
    print(f"  holdout_folds: {selected_holdouts}")
    print(f"  dataset_key: {dataset_name}")
    print(f"  result_csv_suffix: {result_csv_suffix}")
    print(f"  real_world_holdout_mode: {'true' if args.real_world_holdout_mode else 'false'}")
    if args.real_world_holdout_mode:
        print(f"  real_world_split_mode: {real_world_split_mode}")
        print(f"  real_world_dataset_root: {args.real_world_dataset_root}")
    if args.pair_swap_mode:
        print(f"  pair_swap_mode: true")
        print(f"  pair_swap_source_env: {pair_swap_environment_subset[0]}")
        print(f"  pair_swap_target_env: {pair_swap_environment_subset[1]}")
    else:
        print(f"  pair_swap_mode: false")
    print(f"  seeds: {SEEDS}")
    print(f"  fewshot_ks: {FEWSHOT_KS}")
    print(
        "  train_subset_stratified: "
        f"{'true' if args.real_world_holdout_mode else 'false'}"
    )

    for backbone in selected_backbones:
        for perspective in selected_perspectives:
            for method in selected_heads:
                head_cfg = HEADS[method]
                hparams = get_hparams(pos_cfg, perspective, head_cfg["probe_type"])
                for holdout in selected_holdouts:
                    for seed in SEEDS:
                        # 1) Zero-shot LOTO run (K=0)
                        if args.skip_existing:
                            existing = find_matching_row(
                                read_csv_rows(csv_path),
                                backbone=backbone,
                                method=method,
                                perspective=perspective,
                                holdout=holdout,
                                seed=seed,
                                protocol="loto_source_to_target",
                                init_type="zero_shot",
                                fewshot_k=0,
                            )
                            if existing is None:
                                cmd = build_train_command(
                                    dataset_key=dataset_name,
                                    backbone=backbone,
                                    method=method,
                                    perspective=perspective,
                                    holdout=holdout,
                                    seed=seed,
                                    protocol="loto_source_to_target",
                                    init_type="zero_shot",
                                    fewshot_k=0,
                                    train_subset_size=0,
                                    train_subset_stratified=args.real_world_holdout_mode,
                                    ckpt_path="",
                                    save_head=True,
                                    output_dir=output_dir,
                                    hparams=hparams,
                                    environment_subset=pair_swap_environment_subset,
                                    result_csv_suffix=result_csv_suffix,
                                    real_world_split_mode=real_world_split_mode if args.real_world_holdout_mode else None,
                                    real_world_dataset_root=(
                                        args.real_world_dataset_root
                                        if args.real_world_holdout_mode
                                        else None
                                    ),
                                )
                                run_command(cmd, cwd=project_dir, dry_run=args.dry_run)
                        else:
                            cmd = build_train_command(
                                dataset_key=dataset_name,
                                backbone=backbone,
                                method=method,
                                perspective=perspective,
                                holdout=holdout,
                                seed=seed,
                                protocol="loto_source_to_target",
                                init_type="zero_shot",
                                fewshot_k=0,
                                train_subset_size=0,
                                train_subset_stratified=args.real_world_holdout_mode,
                                ckpt_path="",
                                save_head=True,
                                output_dir=output_dir,
                                hparams=hparams,
                                environment_subset=pair_swap_environment_subset,
                                result_csv_suffix=result_csv_suffix,
                                real_world_split_mode=real_world_split_mode if args.real_world_holdout_mode else None,
                                real_world_dataset_root=(
                                    args.real_world_dataset_root
                                    if args.real_world_holdout_mode
                                    else None
                                ),
                            )
                            run_command(cmd, cwd=project_dir, dry_run=args.dry_run)

                        if args.dry_run:
                            zero_head_path = ""
                        else:
                            rows = read_csv_rows(csv_path)
                            zero_row = find_matching_row(
                                rows,
                                backbone=backbone,
                                method=method,
                                perspective=perspective,
                                holdout=holdout,
                                seed=seed,
                                protocol="loto_source_to_target",
                                init_type="zero_shot",
                                fewshot_k=0,
                            )
                            if zero_row is None:
                                raise RuntimeError(
                                    "Missing zero-shot row after run for "
                                    f"{backbone}|{method}|{perspective}|{holdout}|seed={seed}"
                                )
                            zero_head_path = zero_row.get("Head Path", "").strip()
                            if not zero_head_path:
                                raise RuntimeError(
                                    "Zero-shot row found but Head Path is empty for "
                                    f"{backbone}|{method}|{perspective}|{holdout}|seed={seed}"
                                )

                        # 2) Few-shot transfer + scratch
                        for k in FEWSHOT_KS:
                            for init_type, init_ckpt in (
                                ("transfer", zero_head_path),
                                ("scratch", ""),
                            ):
                                if args.skip_existing and not args.dry_run:
                                    existing = find_matching_row(
                                        read_csv_rows(csv_path),
                                        backbone=backbone,
                                        method=method,
                                        perspective=perspective,
                                        holdout=holdout,
                                        seed=seed,
                                        protocol="target_only",
                                        init_type=init_type,
                                        fewshot_k=k,
                                    )
                                    if existing is not None:
                                        continue
                                cmd = build_train_command(
                                    dataset_key=dataset_name,
                                    backbone=backbone,
                                    method=method,
                                    perspective=perspective,
                                    holdout=holdout,
                                    seed=seed,
                                    protocol="target_only",
                                    init_type=init_type,
                                    fewshot_k=k,
                                    train_subset_size=k,
                                    train_subset_stratified=args.real_world_holdout_mode,
                                    ckpt_path=init_ckpt,
                                    save_head=False,
                                    output_dir=output_dir,
                                    hparams=hparams,
                                    environment_subset=pair_swap_environment_subset,
                                    result_csv_suffix=result_csv_suffix,
                                    real_world_split_mode=real_world_split_mode if args.real_world_holdout_mode else None,
                                    real_world_dataset_root=(
                                        args.real_world_dataset_root
                                        if args.real_world_holdout_mode
                                        else None
                                    ),
                                )
                                run_command(cmd, cwd=project_dir, dry_run=args.dry_run)

    if not args.skip_summary:
        report_cmd = [
            "python",
            "scripts/summarize_unreal_loto_fewshot_rebuttal.py",
            "--results-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir / "position_between_objects" / dataset_name / "reports"),
        ]
        run_command(report_cmd, cwd=project_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
