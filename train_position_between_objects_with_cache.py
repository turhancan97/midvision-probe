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

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import matplotlib
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed
from evals.datasets.unreal_position import LABEL_TO_INDEX

# use non-interactive backend for headless environments
matplotlib.use("Agg")


def ddp_setup(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def topk_accuracies(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 2)):
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


@dataclass
class FeatureSplit:
    features: torch.Tensor
    labels: torch.Tensor

    def to_dataset(self) -> Dataset:
        return TensorDataset(self.features, self.labels)


class FeatureCacheManager:
    def __init__(self, cfg: DictConfig, model, rank: int, world_size: int):
        self.cfg = cfg
        self.model = model
        self.rank = rank
        self.world_size = world_size
        if not getattr(cfg, "feature_cache_dir", ""):
            raise ValueError("cfg.feature_cache_dir must be set for cached training.")
        base_dir = Path(cfg.feature_cache_dir).expanduser()
        dataset_root = Path(cfg.dataset.root)
        # differentiate environments/objects using the last two components when available
        tail_parts = dataset_root.parts[-2:] if len(dataset_root.parts) >= 2 else dataset_root.parts
        dataset_name = getattr(cfg.dataset, "name", "dataset")
        model_identifier = self._model_identifier(model)
        self.cache_dir = base_dir / dataset_name
        for part in tail_parts:
            self.cache_dir = self.cache_dir / part
        self.cache_dir = self.cache_dir / model_identifier
        if getattr(cfg.dataset, "exclude_ambiguous", None) is not None:
            flag = "no_amb" if cfg.dataset.exclude_ambiguous else "with_amb"
            self.cache_dir = self.cache_dir / flag
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[str, FeatureSplit] = {}

    @staticmethod
    def _model_identifier(model) -> str:
        name = getattr(model, "checkpoint_name", "model")
        layer = getattr(model, "layer", "unknown")
        output = getattr(model, "output", "unknown")
        patch = getattr(model, "patch_size", "p?")
        return f"{name}_layer-{layer}_out-{output}_patch-{patch}"

    def _cache_path(self, split: str) -> Path:
        return self.cache_dir / f"{split}.pt"

    def _extract_split(self, split: str, batch_size: int):
        dataset = instantiate(self.cfg.dataset, split=split, seed=self.cfg.system.random_seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )
        features = []
        labels = []
        self.model.eval()
        device = torch.device(f"cuda:{self.rank}") if torch.cuda.is_available() else torch.device("cpu")
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {split} features", disable=self.rank != 0):
                images = batch["image"].to(device, non_blocking=True)
                feats = self.model(images)
                if isinstance(feats, (list, tuple)):
                    feats = torch.cat(feats, dim=-1)
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
                features.append(feats.cpu())
                labels.append(batch["label"].cpu())
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        cache_payload = {"features": features, "labels": labels}
        torch.save(cache_payload, self._cache_path(split))

    def get_split(self, split: str, batch_size: int) -> FeatureSplit:
        if split in self._loaded:
            return self._loaded[split]

        cache_file = self._cache_path(split)
        if self.rank == 0 and not cache_file.exists():
            self._extract_split(split, batch_size)
        if self.world_size > 1:
            torch.distributed.barrier()
        if not cache_file.exists():
            raise FileNotFoundError(f"Expected cached features at {cache_file}")
        data = torch.load(cache_file, map_location="cpu")
        split_obj = FeatureSplit(features=data["features"], labels=data["labels"])
        self._loaded[split] = split_obj
        return split_obj

    def get_trainval(self, batch_size: int) -> FeatureSplit:
        train_split = self.get_split("train", batch_size)
        val_split = self.get_split("valid", batch_size)
        features = torch.cat([train_split.features, val_split.features], dim=0)
        labels = torch.cat([train_split.labels, val_split.labels], dim=0)
        split_obj = FeatureSplit(features=features, labels=labels)
        self._loaded["trainval"] = split_obj
        return split_obj


def build_feature_loader(split: str, feature_split: FeatureSplit, batch_size: int, world_size: int) -> DataLoader:
    dataset = feature_split.to_dataset()
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))
    shuffle = split == "train" and sampler is None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def train_one_epoch(head, loader, optimizer, loss_fn, rank, scheduler, num_classes: int):
    head.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top2 = 0.0
    class_correct = torch.zeros(num_classes, dtype=torch.float64)
    class_total = torch.zeros(num_classes, dtype=torch.float64)
    n_samples = 0

    iterable = tqdm(loader, desc="train") if rank == 0 else loader
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    for feats, labels in iterable:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = head(feats)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        top1, top2 = topk_accuracies(logits, labels, ks=(1, 2))
        preds = logits.argmax(dim=1)
        for cls in range(num_classes):
            mask = labels == cls
            denom = mask.sum().item()
            if denom == 0:
                continue
            class_total[cls] += denom
            class_correct[cls] += (preds[mask] == cls).sum().item()

        running_loss += loss.item() * labels.size(0)
        running_top1 += top1 * labels.size(0)
        running_top2 += top2 * labels.size(0)
        n_samples += labels.size(0)

        if rank == 0 and isinstance(iterable, tqdm):
            iterable.set_description(f"loss: {loss.item():.4f} | top1: {top1:.3f} top2: {top2:.3f}")

    train_bal = 0.0
    valid_classes = class_total > 0
    if valid_classes.any():
        train_bal = float((class_correct[valid_classes] / class_total[valid_classes]).mean())

    return (
        running_loss / max(1, n_samples),
        running_top1 / max(1, n_samples),
        running_top2 / max(1, n_samples),
        train_bal,
    )


@torch.no_grad()
def evaluate(head, loader, rank, num_classes):
    head.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top2 = 0.0
    running_bal = 0.0
    n_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    for feats, labels in loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = head(feats)
        loss = loss_fn(logits, labels)
        top1, top2 = topk_accuracies(logits, labels, ks=(1, 2))
        bal = balanced_accuracy(logits, labels, num_classes)

        running_loss += loss.item() * labels.size(0)
        running_top1 += top1 * labels.size(0)
        running_top2 += top2 * labels.size(0)
        running_bal += bal * labels.size(0)
        n_samples += labels.size(0)

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        running_loss / n_samples,
        running_top1 / n_samples,
        running_top2 / n_samples,
        running_bal / n_samples,
    )


def plot_metrics(history, output_dir: Path, prefix: str, model_name: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_top1 = [h["train_top1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]
    train_bal = [h["train_bal"] for h in history]
    val_bal = [h["val_bal"] for h in history]

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss_{model_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_top1, label="train")
    plt.plot(epochs, val_top1, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_top1_{model_name}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_bal, label="train")
    plt.plot(epochs, val_bal, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_balanced_{model_name}.png", dpi=200)
    plt.close()


def save_confusion_matrix(head, loader, rank, class_order, class_names, output_path: Path):
    head.eval()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    preds = []
    labels = []
    with torch.no_grad():
        for feats, lbls in loader:
            feats = feats.to(device, non_blocking=True)
            logits = head(feats)
            preds.append(logits.argmax(dim=1).cpu())
            labels.append(lbls.cpu())
    if not labels:
        return
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    cm = confusion_matrix(y_true, y_pred, labels=class_order)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(len(class_order)), yticks=range(len(class_order)), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label', title='Test Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = 'white' if max_val and value > max_val / 2 else 'black'
            ax.text(j, i, f"{value}", ha='center', va='center', color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_trial_features(
    rank: int,
    probe_cfg: DictConfig,
    feat_dim: int,
    train_split: FeatureSplit,
    val_split: FeatureSplit,
    num_classes: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    warmup_epochs: float,
    patience: int,
    eval_every_epochs: int,
) -> Tuple[float, int, list]:
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    head = instantiate(probe_cfg, feat_dim=feat_dim)
    head = head.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": lr, "weight_decay": weight_decay}]
    )
    train_loader = build_feature_loader("train", train_split, batch_size, world_size=1)
    val_loader = build_feature_loader("valid", val_split, batch_size, world_size=1)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_lambda = lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(n_epochs):
        train_loss, train_top1, train_top2, train_bal = train_one_epoch(
            head, train_loader, optimizer, loss_fn, rank, scheduler, num_classes
        )
        log_epoch = (epoch % max(1, eval_every_epochs) == 0) or (epoch == n_epochs - 1)
        val_loss = val_top1 = val_top2 = val_bal = 0.0
        if log_epoch:
            val_loss, val_top1, val_top2, val_bal = evaluate(head, val_loader, rank, num_classes)
            if val_bal > best_val + 1e-8:
                best_val = val_bal
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top2": train_top2,
                "train_bal": train_bal,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top2": val_top2,
                "val_bal": val_bal,
            }
        )
        if patience > 0 and no_improve >= patience:
            break

    if best_epoch == -1 and history:
        best_epoch = history[-1]["epoch"]
        best_val = history[-1]["val_bal"]

    return best_val, best_epoch, history


def final_fit_and_test(
    rank: int,
    probe_cfg: DictConfig,
    feat_dim: int,
    trainval_split: FeatureSplit,
    val_split: FeatureSplit,
    test_split: FeatureSplit,
    num_classes: int,
    batch_size: int,
    best_epoch: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: float,
) -> Tuple[float, float, float, float]:
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    head = instantiate(probe_cfg, feat_dim=feat_dim)
    head = head.to(device)
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": lr, "weight_decay": weight_decay}]
    )
    train_loader = build_feature_loader("train", trainval_split, batch_size, world_size=1)
    val_loader = build_feature_loader("valid", val_split, batch_size, world_size=1)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, best_epoch + 1) * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_lambda = lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(max(1, best_epoch + 1)):
        train_one_epoch(head, train_loader, optimizer, loss_fn, rank, scheduler, num_classes)
        evaluate(head, val_loader, rank, num_classes)

    test_loader = build_feature_loader("test", test_split, batch_size, world_size=1)
    return evaluate(head, test_loader, rank, num_classes)


def run_sweep(
    cfg: DictConfig,
    backbone,
    cache_manager: FeatureCacheManager,
    train_split: FeatureSplit,
    val_split: FeatureSplit,
    test_split: FeatureSplit,
    feat_dim: int,
    rank: int,
):
    lrs = list(getattr(cfg.sweep, "learning_rates", [cfg.optimizer.probe_lr]))
    wds = list(getattr(cfg.sweep, "weight_decays", [cfg.optimizer.weight_decay]))
    patience = int(getattr(cfg.sweep, "patience", 3))
    eval_every = int(getattr(cfg.sweep, "eval_every_epochs", 1))
    n_epochs = int(cfg.optimizer.n_epochs)
    warmup_epochs = float(cfg.optimizer.warmup_epochs)
    num_classes = getattr(cfg.probe, "num_classes", 4)

    result_dir = Path(cfg.output_dir) / "position_between_objects"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "position_between_objects_sweep_unreal.csv"
    new_file = not csv_path.exists()

    sweep_rows = []
    best_overall = -1.0
    best_cfg: Optional[Dict[str, float]] = None
    for idx_lr, lr in enumerate(lrs):
        for idx_wd, wd in enumerate(wds):
            best_val, best_epoch, history = run_trial_features(
                rank,
                cfg.probe,
                feat_dim,
                train_split,
                val_split,
                num_classes,
                cfg.batch_size,
                lr,
                wd,
                n_epochs,
                warmup_epochs,
                patience,
                eval_every,
            )
            sweep_rows.append(
                [
                    datetime.now().strftime("%d%m%Y-%H%M"),
                    backbone.checkpoint_name,
                    backbone.patch_size,
                    str(backbone.layer),
                    backbone.output,
                    lr,
                    wd,
                    n_epochs,
                    best_val,
                    best_epoch,
                ]
            )
            if best_val > best_overall:
                best_overall = best_val
                best_cfg = {"lr": lr, "wd": wd, "best_epoch": best_epoch}

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "Timestamp",
                    "Model Checkpoint",
                    "Patch Size",
                    "Layer",
                    "Output",
                    "LR",
                    "Weight Decay",
                    "Epochs",
                    "Best Val Balanced Acc",
                    "Best Epoch",
                ]
            )
        for row in sweep_rows:
            writer.writerow(row)

    if getattr(cfg.sweep, "final_fit", True) and best_cfg is not None:
        trainval_split = cache_manager.get_trainval(cfg.batch_size)
        test_loss, test_top1, test_top2, test_bal = final_fit_and_test(
            rank,
            cfg.probe,
            feat_dim,
            trainval_split,
            val_split,
            test_split,
            num_classes,
            cfg.batch_size,
            best_cfg.get("best_epoch", 0),
            best_cfg["lr"],
            best_cfg["wd"],
            warmup_epochs,
        )
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%d%m%Y-%H%M"),
                    backbone.checkpoint_name,
                    backbone.patch_size,
                    str(backbone.layer),
                    backbone.output,
                    f"final_lr={best_cfg['lr']}",
                    f"final_wd={best_cfg['wd']}",
                    best_cfg.get("best_epoch", 0) + 1,
                    f"test_bal_acc={test_bal}",
                    "final",
                ]
            )

def train_model(rank, world_size, cfg: DictConfig):
    set_random_seed(cfg.system.random_seed)
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    if rank == 0:
        exp_path = Path(__file__).parent / f"position_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    backbone = instantiate(cfg.backbone)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(device)

    feat_dim = backbone.feat_dim if isinstance(backbone.feat_dim, int) else sum(backbone.feat_dim)

    cache_manager = FeatureCacheManager(cfg, backbone, rank, world_size)
    train_split = cache_manager.get_split("train", cfg.batch_size)
    val_split = cache_manager.get_split("valid", cfg.batch_size)
    test_split = cache_manager.get_split("test", cfg.batch_size)

    if getattr(cfg, "sweep", {}).get("enable", False):
        if world_size > 1 and rank != 0:
            destroy_process_group()
            return
        run_sweep(cfg, backbone, cache_manager, train_split, val_split, test_split, feat_dim, rank)
        if world_size > 1 and torch.distributed.is_initialized():
            destroy_process_group()
        return

    head = instantiate(cfg.probe, feat_dim=feat_dim)
    head = head.to(device)
    if world_size > 1:
        head = DDP(head, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": cfg.optimizer.probe_lr, "weight_decay": cfg.optimizer.weight_decay}]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = build_feature_loader("train", train_split, cfg.batch_size, world_size)
    val_loader = build_feature_loader("valid", val_split, cfg.batch_size, world_size=1)
    test_loader = build_feature_loader("test", test_split, cfg.batch_size, world_size=1)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.optimizer.n_epochs * steps_per_epoch
    warmup_steps = int(cfg.optimizer.warmup_epochs * steps_per_epoch)
    lr_lambda = lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    history = []
    num_classes = getattr(cfg.probe, "num_classes", 4)

    for epoch in range(cfg.optimizer.n_epochs):
        if world_size > 1 and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_top1, train_top2, train_bal = train_one_epoch(
            head, train_loader, optimizer, loss_fn, rank, scheduler, num_classes
        )
        val_loss, val_top1, val_top2, val_bal = evaluate(head, val_loader, rank, num_classes)

        if rank == 0:
            logger.info(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} top1 {train_top1:.4f} top2 {train_top2:.4f} bal {train_bal:.4f} | "
                f"val loss {val_loss:.4f} top1 {val_top1:.4f} top2 {val_top2:.4f} bal {val_bal:.4f}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top2": train_top2,
                "train_bal": train_bal,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top2": val_top2,
                "val_bal": val_bal,
            }
        )

    test_loss, test_top1, test_top2, test_bal = evaluate(head, test_loader, rank, num_classes)
    if rank == 0:
        logger.info(
            f"test loss {test_loss:.4f} top1 {test_top1:.4f} top2 {test_top2:.4f} bal {test_bal:.4f}"
        )

    if rank == 0:
        timestamp = datetime.now().strftime("%d%m%Y-%H%M")
        result_dir = Path(cfg.output_dir) / "position_between_objects"
        result_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = result_dir / "plots"
        plot_metrics(history, plot_dir, prefix=f"{cfg.experiment_name}_{timestamp}", model_name=cfg.experiment_model)

        class_order = [
            LABEL_TO_INDEX["Left"],
            LABEL_TO_INDEX["Right"],
            LABEL_TO_INDEX["Front"],
            LABEL_TO_INDEX["Back"],
        ]
        class_names = ["Left", "Right", "Front", "Back"]
        cm_path = plot_dir / f"{cfg.experiment_name}_{timestamp}_confusion_{cfg.experiment_model}.png"
        save_confusion_matrix(head, test_loader, rank, class_order, class_names, cm_path)

        csv_path = result_dir / "position_between_objects_results_unreal_final.csv"
        is_new = not csv_path.exists()
        model_name = backbone.checkpoint_name
        patch_size = backbone.patch_size
        layer = backbone.layer
        output = backbone.output

        # Evaluate once more on validation with rank 0 head for summary
        val_loss, val_top1, val_top2, val_bal = evaluate(head, val_loader, rank, num_classes)

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
            "Top2 Val",
            "Balanced Acc Val",
            "Top1 Test",
            "Top2 Test",
            "Balanced Acc Test",
        ]

        probe_name = head.module.name if isinstance(head, DDP) else head.name
        train_dataset_name = getattr(cfg.dataset, "name", "unreal_position")
        val_dataset_name = f"{train_dataset_name}_val"

        row = [
            timestamp,
            model_name,
            patch_size,
            str(layer),
            output,
            probe_name,
            cfg.system.random_seed,
            cfg.optimizer.n_epochs,
            cfg.optimizer.warmup_epochs,
            cfg.optimizer.probe_lr,
            0.0,
            cfg.batch_size,
            train_dataset_name,
            val_dataset_name,
            f"{val_top1*100:.2f}",
            f"{val_top2*100:.2f}",
            f"{val_bal*100:.2f}",
            f"{test_top1*100:.2f}",
            f"{test_top2*100:.2f}",
            f"{test_bal*100:.2f}",
        ]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(headers)
            writer.writerow(row)

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="position_between_objects_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
