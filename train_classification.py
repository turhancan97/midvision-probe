"""
MIT License

Copyright (c) 2024 Mohamed El Banani
Copyright (c) 2025 Turhan Can Kargın

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb
import csv

from evals.datasets.builder import build_loader
from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed


def ddp_setup(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def topk_accuracies(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)):
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, 1, True, True)  # [B, maxk]
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [maxk, B]
        res = []
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / targets.size(0)).item())
        return res


def balanced_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        recalls = []
        for c in range(num_classes):
            mask = targets == c
            denom = mask.sum().item()
            if denom == 0:
                continue
            tp = (preds[mask] == c).sum().item()
            recalls.append(tp / denom)
        if len(recalls) == 0:
            return 0.0
        return float(sum(recalls) / len(recalls))


def train_one_epoch(model, head, loader, optimizer, loss_fn, rank, scheduler, detach_backbone=True):
    model.eval()  # backbone frozen
    head.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    n_samples = 0

    pbar = tqdm(loader) if rank == 0 else loader
    for batch in pbar:
        images = batch["image"].to(rank, non_blocking=True)
        labels = batch["label"].to(rank, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad() if detach_backbone else torch.enable_grad():
            feats = model(images)
        logits = head(feats)
        loss = loss_fn(logits, labels)
        loss.backward()
        scheduler.step()
        optimizer.step()
        top1, top5 = topk_accuracies(logits, labels, ks=(1, 5))
        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)
        running_top5 += top5 * images.size(0)
        n_samples += images.size(0)

        if rank == 0:
            pbar.set_description(f"loss: {loss.item():.4f} | top1: {top1:.3f} top5: {top5:.3f}")

    return running_loss / n_samples, running_top1 / n_samples, running_top5 / n_samples


@torch.no_grad()
def evaluate(model, head, loader, rank, num_classes):
    model.eval()
    head.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    running_bal = 0.0
    n_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(rank, non_blocking=True)
        labels = batch["label"].to(rank, non_blocking=True)

        feats = model(images)
        logits = head(feats)
        loss = loss_fn(logits, labels)
        top1, top5 = topk_accuracies(logits, labels, ks=(1, 5))
        bal = balanced_accuracy(logits, labels, num_classes)

        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)
        running_top5 += top5 * images.size(0)
        running_bal += bal * images.size(0)
        n_samples += images.size(0)

    return (
        running_loss / n_samples,
        running_top1 / n_samples,
        running_top5 / n_samples,
        running_bal / n_samples,
    )


def train_model(rank, world_size, cfg: DictConfig):
    set_random_seed(cfg.system.random_seed)
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== W&B =====
    if rank == 0 and cfg.wandb.use:
        sanitized_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        wandb.init(
            project="ssl-linear-probe-classification",
            config=sanitized_cfg,
            name=f"{cfg.experiment_name}_{cfg.dataset.name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group="seed: " + str(cfg.system.random_seed),
        )

    # ===== Data =====
    train_loader = build_loader(cfg.dataset, "train", cfg.batch_size, world_size)
    val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size, 1)

    # ===== Models =====
    model = instantiate(cfg.backbone)
    # freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # infer feat_dim from backbone
    feat_dim = model.feat_dim if isinstance(model.feat_dim, int) else sum(model.feat_dim)
    head = instantiate(cfg.probe, feat_dim=feat_dim)

    # DDP wrapping for head if multi-gpu; backbone stays eval frozen
    model = model.to(rank)
    head = head.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        head = DDP(head, device_ids=[rank])

    # ===== Optimizer/Scheduler =====
    optimizer = torch.optim.AdamW([{ "params": head.parameters(), "lr": cfg.optimizer.probe_lr, "weight_decay": cfg.optimizer.weight_decay }])
    total_steps = cfg.optimizer.n_epochs * max(1, len(train_loader))
    warmup_steps = int(cfg.optimizer.warmup_epochs * max(1, len(train_loader)))
    lr_lambda = lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ===== Logging setup =====
    if rank == 0:
        exp_path = Path(__file__).parent / f"linear_probe_classification_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Training Loop =====
    for epoch in range(cfg.optimizer.n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_top1, train_top5 = train_one_epoch(
            model, head, train_loader, optimizer, loss_fn, rank, scheduler,detach_backbone=True
        )

        val_loss, val_top1, val_top5, val_bal = evaluate(
            model, head, val_loader, rank, getattr(cfg.probe, "num_classes", 10)
        )

        if rank == 0:
            logger.info(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} top1 {train_top1:.4f} top5 {train_top5:.4f} | "
                f"val loss {val_loss:.4f} top1 {val_top1:.4f} top5 {val_top5:.4f} bal {val_bal:.4f}"
            )
            if cfg.wandb.use:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_top1": train_top1,
                        "train_top5": train_top5,
                        "val_loss": val_loss,
                        "val_top1": val_top1,
                        "val_top5": val_top5,
                        "val_balanced_acc": val_bal,
                        "probe_lr": optimizer.param_groups[0]["lr"],
                    }
                )

    # ===== Save CSV summary =====
    if rank == 0:
        model_name = model.module.checkpoint_name if isinstance(model, DDP) else model.checkpoint_name
        patch_size = model.module.patch_size if isinstance(model, DDP) else model.patch_size
        layer = model.module.layer if isinstance(model, DDP) else model.layer
        output = model.module.output if isinstance(model, DDP) else model.output

        # final eval for summary row
        val_loss, val_top1, val_top5, val_bal = evaluate(
            model, head, val_loader, 0, getattr(cfg.probe, "num_classes", 10)
        )

        headers = [
            "Timestamp",
            "Model Checkpoint",
            "Patch Size",
            "Layer",
            "Output",
            "Probe Name",
            "Random Seed",
            "Num Epochs",
            "Warmup Epochs",
            "Probe LR",
            "Model LR",
            "Batch Size",
            "Train Dataset",
            "Val Dataset",
            "Top1 Val",
            "Top5 Val",
            "Balanced Acc Val",
        ]

        row = [
            datetime.now().strftime("%d%m%Y-%H%M"),
            model_name,
            patch_size,
            str(layer),
            output,
            (head.module.name if isinstance(head, DDP) else head.name),
            cfg.system.random_seed,
            cfg.optimizer.n_epochs,
            cfg.optimizer.warmup_epochs,
            cfg.optimizer.probe_lr,
            0.0,
            cfg.batch_size,
            getattr(train_loader.dataset, "name", "imagenette"),
            getattr(val_loader.dataset, "name", "imagenette_val"),
            f"{val_top1*100:.2f}",
            f"{val_top5*100:.2f}",
            f"{val_bal*100:.2f}",
        ]

        result_dir = os.path.join(f"{cfg.output_dir}", "linear_probe_classification")
        os.makedirs(result_dir, exist_ok=True)
        csv_path = os.path.join(result_dir, "linear_probe_classification_results_imagenette_final.csv")
        is_new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(headers)
            writer.writerow(row)

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="linear_probe_classification_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()

