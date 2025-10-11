"""
Training script for equivariance regression probes with frozen backbones.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed


def _sanitize_name(name: str) -> str:
    sanitized = name.replace("/", "_").replace(" ", "_")
    sanitized = sanitized.replace("-", "_")
    return "".join(c for c in sanitized if c.isalnum() or c == "_").lower()


@dataclass
class FeatureRecord:
    feature: torch.Tensor
    target: torch.Tensor
    frame_number: int


def ddp_setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def gather_entries(local_entries: List[Tuple[Tuple[str, str, str], FeatureRecord]], world_size: int) -> List[List[Tuple[Tuple[str, str, str], FeatureRecord]]]:
    if world_size == 1:
        return [local_entries]
    gathered: List[List[Tuple[Tuple[str, str, str], FeatureRecord]]] = [None for _ in range(world_size)]  # type: ignore
    dist.all_gather_object(gathered, local_entries)
    return gathered


def collect_split_features(
    loader,
    model,
    rank: int,
    world_size: int,
) -> Dict[Tuple[str, str, str], List[FeatureRecord]]:
    groups: DefaultDict[Tuple[str, str, str], List[FeatureRecord]] = defaultdict(list)
    if loader is None:
        return groups

    iterator = tqdm(loader, disable=rank != 0)
    for batch in iterator:
        images = batch["image"].to(rank, non_blocking=True)
        targets = batch["target"]
        environments = batch["environment"]
        levels = batch["level"]
        objects = batch["object_class"]
        frame_numbers = batch["frame_number"].tolist()

        with torch.no_grad():
            feats = model(images)

        if not isinstance(feats, torch.Tensor):
            raise TypeError("Backbone forward expected to return a torch.Tensor for equivariance tasks.")

        feats_cpu = detach_to_cpu(feats)
        targets_cpu = detach_to_cpu(targets)

        local_entries: List[Tuple[Tuple[str, str, str], FeatureRecord]] = []
        for idx in range(feats_cpu.size(0)):
            key = (environments[idx], levels[idx], objects[idx])
            record = FeatureRecord(
                feature=feats_cpu[idx].clone(),
                target=targets_cpu[idx].clone(),
                frame_number=int(frame_numbers[idx]),
            )
            local_entries.append((key, record))

        gathered = gather_entries(local_entries, world_size)
        if rank == 0:
            for entry_list in gathered:
                for key, record in entry_list:
                    groups[key].append(record)

    return groups


def stack_records(records: List[FeatureRecord]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    if not records:
        return torch.empty(0), torch.empty(0), []
    records_sorted = sorted(records, key=lambda r: r.frame_number)
    features = torch.stack([r.feature for r in records_sorted])
    targets = torch.stack([r.target for r in records_sorted])
    frame_numbers = [r.frame_number for r in records_sorted]
    return features, targets, frame_numbers


def compute_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if preds.numel() == 0:
        return float("nan")
    mse = torch.mean((preds - targets) ** 2)
    return float(torch.sqrt(mse).item())


def evaluate_head(model, features: torch.Tensor, targets: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    if features.numel() == 0:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        preds = model(features.to(device))
        mse = torch.mean((preds - targets.to(device)) ** 2).item()
    rmse = math.sqrt(mse) if mse >= 0 else float("nan")
    return mse, rmse


def train_probe(
    head,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    cfg: DictConfig,
    object_identifier: str,
    log_prefix: str,
    log_to_wandb: bool,
) -> Dict[str, float]:
    if train_features.numel() == 0:
        return {
            "train_mse": float("nan"),
            "train_rmse": float("nan"),
            "val_mse": float("nan"),
            "val_rmse": float("nan"),
        }

    head = head.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": cfg.optimizer.probe_lr, "weight_decay": cfg.optimizer.weight_decay}]
    )

    steps_per_epoch = max(1, math.ceil(train_features.size(0) / max(1, cfg.batch_size)))
    total_steps = cfg.optimizer.n_epochs * steps_per_epoch
    warmup_steps = int(cfg.optimizer.warmup_epochs * steps_per_epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps)))
    loss_fn = torch.nn.MSELoss()

    indices = torch.arange(train_features.size(0))
    for epoch in range(cfg.optimizer.n_epochs):
        head.train()
        perm = indices[torch.randperm(indices.size(0))]
        epoch_loss = 0.0
        count = 0
        for batch_indices in perm.split(max(1, cfg.batch_size)):
            feats = train_features[batch_indices].to(device, non_blocking=True)
            targets = train_targets[batch_indices].to(device, non_blocking=True)
            preds = head(feats)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * batch_indices.size(0)
            count += batch_indices.size(0)

        train_mse_epoch = epoch_loss / max(1, count)
        if cfg.training.eval_every_epochs > 0 and epoch % cfg.training.eval_every_epochs == 0:
            val_mse_epoch, _ = evaluate_head(head, val_features, val_targets, device)

        if log_to_wandb:
            wandb.log(
                {
                    f"train_mse_{log_prefix}": train_mse_epoch,
                    f"val_mse_{log_prefix}": val_mse_epoch,
                }
            )


    head.eval()
    with torch.no_grad():
        train_preds = head(train_features.to(device))
        val_preds = head(val_features.to(device)) if val_features.numel() > 0 else torch.empty(0, device=device)

    train_preds_cpu = train_preds.detach().cpu()
    train_rmse = compute_rmse(train_preds_cpu, train_targets)
    train_mse = (
        float(torch.mean((train_preds_cpu - train_targets) ** 2).item())
        if train_preds_cpu.numel() > 0
        else float("nan")
    )

    val_mse, val_rmse = evaluate_head(head, val_features, val_targets, device)

    return {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "head": head,
    }


def run_equivariance(rank: int, world_size: int, cfg: DictConfig) -> None:
    set_random_seed(cfg.system.random_seed)
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        exp_path = Path(__file__).parent / f"equivariance_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    if (not getattr(cfg, "sweep", {}).get("enable", False)) and rank == 0 and cfg.wandb.use:
        sanitized_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        wandb.init(
            project="ssl-equivariance",
            config=sanitized_cfg,
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group=f"seed:{cfg.system.random_seed}",
        )

    train_loader = build_loader(cfg.dataset, "train", cfg.batch_size, world_size, seed=cfg.system.random_seed)
    val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size, world_size if world_size > 1 else 1, seed=cfg.system.random_seed)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, world_size if world_size > 1 else 1, seed=cfg.system.random_seed)

    model = instantiate(cfg.backbone)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    feat_dim_attr = model.module.feat_dim if isinstance(model, DDP) else model.feat_dim
    feat_dim = feat_dim_attr if isinstance(feat_dim_attr, int) else sum(feat_dim_attr)

    train_groups = collect_split_features(train_loader, model, rank, world_size)
    val_groups = collect_split_features(val_loader, model, rank, world_size)
    test_groups = collect_split_features(test_loader, model, rank, world_size)

    if world_size > 1:
        dist.barrier()

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    regression_dim = 2 if cfg.regression_type.lower() == "circle" else 1

    backbone_obj = model.module if isinstance(model, DDP) else model
    head_name = cfg.probe.get("_target_", "probe")
    backbone_name = getattr(backbone_obj, "checkpoint_name", getattr(backbone_obj, "model_name", "unknown"))

    results = []
    environments = sorted({key[0] for key in train_groups.keys()})
    levels = cfg.levels

    for environment in environments:
        for level in tqdm(levels, desc=f"Levels for {environment}"):
            object_keys = [key for key in train_groups.keys() if key[0] == environment and key[1] == level]
            for key in object_keys:
                _, _, object_name = key
                train_records = train_groups.get(key, [])
                val_records = val_groups.get(key, [])
                test_records = test_groups.get(key, [])

                if len(train_records) < cfg.training.min_samples_per_object:
                    logger.info(
                        "Skipping %s/%s/%s (insufficient samples: %d)",
                        environment,
                        level,
                        object_name,
                        len(train_records),
                    )
                    results.append(
                        {
                            "environment": environment,
                            "level": level,
                            "object": object_name,
                            "train_mse": float("nan"),
                            "val_mse": float("nan"),
                            "test_mse": float("nan"),
                            "train_rmse": float("nan"),
                            "val_rmse": float("nan"),
                            "test_rmse": float("nan"),
                            "num_train": len(train_records),
                            "num_val": len(val_records),
                            "num_test": len(test_records),
                        }
                    )
                    continue

                train_features, train_targets, _ = stack_records(train_records)
                val_features, val_targets, _ = stack_records(val_records)
                test_features, test_targets, _ = stack_records(test_records)

                head = instantiate(cfg.probe, feat_dim=feat_dim, output_dim=regression_dim)

                log_prefix = "_".join(
                    [
                        _sanitize_name(environment),
                        _sanitize_name(object_name),
                        _sanitize_name(level),
                    ]
                )

                metrics = train_probe(
                    head,
                    train_features,
                    train_targets,
                    val_features,
                    val_targets,
                    device,
                    cfg,
                    f"{environment}/{level}/{object_name}",
                    log_prefix,
                    cfg.wandb.use,
                )

                trained_head = metrics.pop("head")
                test_mse, test_rmse = evaluate_head(trained_head, test_features, test_targets, device)

                results.append(
                    {
                        "environment": environment,
                        "level": level,
                        "object": object_name,
                        "train_mse": metrics["train_mse"],
                        "val_mse": metrics["val_mse"],
                        "test_mse": test_mse,
                        "train_rmse": metrics["train_rmse"],
                        "val_rmse": metrics["val_rmse"],
                        "test_rmse": test_rmse,
                        "num_train": len(train_records),
                        "num_val": len(val_records),
                        "num_test": len(test_records),
                    }
                )

    # if cfg.wandb.use and results:
    #     env_metrics: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    #     level_metrics: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    #     for row in results:
    #         env = row["environment"]
    #         lvl = row["level"]
    #         for metric_key in ("train_mse", "val_mse", "test_mse", "train_rmse", "val_rmse", "test_rmse"):
    #             value = row.get(metric_key)
    #             if value is None or math.isnan(value):
    #                 continue
    #             env_metrics[env][metric_key].append(value)
    #             level_metrics[lvl][metric_key].append(value)

    #     for env, metric_dict in env_metrics.items():
    #         log_payload = {}
    #         prefix = _sanitize_name(env)
    #         for metric_key, values in metric_dict.items():
    #             if not values:
    #                 continue
    #             avg = float(torch.tensor(values, dtype=torch.float32).mean().item())
    #             log_payload[f"{metric_key}_environment_{prefix}"] = avg
    #         if log_payload:
    #             wandb.log(log_payload)

    #     for lvl, metric_dict in level_metrics.items():
    #         log_payload = {}
    #         prefix = _sanitize_name(lvl)
    #         for metric_key, values in metric_dict.items():
    #             if not values:
    #                 continue
    #             avg = float(torch.tensor(values, dtype=torch.float32).mean().item())
    #             log_payload[f"{metric_key}_level_{prefix}"] = avg
    #         if log_payload:
    #             wandb.log(log_payload)

    result_dir = Path(cfg.output_dir) / f"equivariance_{cfg.experiment_name}_{cfg.regression_type}"
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    object_csv = result_dir / "object_metrics.csv"
    new_file = not object_csv.exists()
    with object_csv.open("a") as f:
        if new_file:
            f.write(
                "Timestamp,Experiment,Regression,Environment,Level,Object,Train RMSE,Val RMSE,Test RMSE,Num Train,Num Val,Num Test,Backbone,Head\n"
            )
        for row in results:
            f.write(
                f"{timestamp},{cfg.experiment_name},{cfg.regression_type},{row['environment']},{row['level']},{row['object']},{row['train_rmse']},{row['val_rmse']},{row['test_rmse']},{row['num_train']},{row['num_val']},{row['num_test']},{backbone_name},{head_name}\n"
            )

    summary_csv = result_dir / "environment_summary.csv"
    new_summary = not summary_csv.exists()
    with summary_csv.open("a") as f:
        if new_summary:
            header_parts = ["Timestamp", "Experiment", "Regression", "Environment", "Backbone", "Head"]
            for level in levels:
                header_parts.append(f"Train RMSE {level}")
            for level in levels:
                header_parts.append(f"Val RMSE {level}")
            for level in levels:
                header_parts.append(f"Test RMSE {level}")
            f.write(",".join(header_parts) + "\n")

        for environment in environments:
            row_values = [timestamp, cfg.experiment_name, cfg.regression_type, environment, backbone_name, head_name]
            for level in levels:
                vals = [r["train_rmse"] for r in results if r["environment"] == environment and r["level"] == level]
                avg = float(torch.tensor(vals).nanmean().item()) if vals else float("nan")
                row_values.append(str(avg))
            for level in levels:
                vals = [r["val_rmse"] for r in results if r["environment"] == environment and r["level"] == level]
                avg = float(torch.tensor(vals).nanmean().item()) if vals else float("nan")
                row_values.append(str(avg))
            for level in levels:
                vals = [r["test_rmse"] for r in results if r["environment"] == environment and r["level"] == level]
                avg = float(torch.tensor(vals).nanmean().item()) if vals else float("nan")
                row_values.append(str(avg))
            f.write(",".join(row_values) + "\n")

    if cfg.wandb.use:
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


@hydra.main(config_name="equivariance_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig) -> None:
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(run_equivariance, args=(world_size, cfg), nprocs=world_size)
    else:
        run_equivariance(0, world_size, cfg)


if __name__ == "__main__":
    main()
