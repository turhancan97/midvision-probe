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


def train_one_epoch(model, head, loader, optimizer, loss_fn, rank, scheduler, detach_backbone=True):
    model.eval()  # backbone frozen
    head.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top2 = 0.0
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
        optimizer.step()
        scheduler.step()
        top1, top2 = topk_accuracies(logits, labels, ks=(1, 2))
        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)
        running_top2 += top2 * images.size(0)
        n_samples += images.size(0)

        if rank == 0:
            pbar.set_description(f"loss: {loss.item():.4f} | top1: {top1:.3f} top2: {top2:.3f}")

    return running_loss / n_samples, running_top1 / n_samples, running_top2 / n_samples


@torch.no_grad()
def evaluate(model, head, loader, rank, num_classes):
    model.eval()
    head.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top2 = 0.0
    running_bal = 0.0
    n_samples = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(rank, non_blocking=True)
        labels = batch["label"].to(rank, non_blocking=True)

        feats = model(images)
        logits = head(feats)
        loss = loss_fn(logits, labels)
        top1, top2 = topk_accuracies(logits, labels, ks=(1, 2))
        bal = balanced_accuracy(logits, labels, num_classes)

        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)
        running_top2 += top2 * images.size(0)
        running_bal += bal * images.size(0)
        n_samples += images.size(0)

    return (
        running_loss / n_samples,
        running_top1 / n_samples,
        running_top2 / n_samples,
        running_bal / n_samples,
    )


def run_trial(
    rank,
    model,
    feat_dim,
    train_loader,
    val_loader,
    num_classes,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    warmup_epochs: float,
    use_wandb: bool,
    wandb_group: str,
    trial_name: str,
    patience: int = 3,
    eval_every_epochs: int = 1,
):
    head = instantiate({"_target_": "evals.models.probes.ClassificationHead", "feat_dim": feat_dim, "num_classes": num_classes, "use_layernorm": True})
    head = head.to(rank)
    optimizer = torch.optim.AdamW([{"params": head.parameters(), "lr": lr, "weight_decay": weight_decay}])
    total_steps = n_epochs * max(1, len(train_loader))
    warmup_steps = int(warmup_epochs * max(1, len(train_loader)))
    lr_lambda = lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = torch.nn.CrossEntropyLoss()

    run = None
    if use_wandb and rank == 0:
        run = wandb.init(project="ssl-position-between-objects", name=trial_name, group=wandb_group, config={"lr": lr, "weight_decay": weight_decay, "epochs": n_epochs})

    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    for epoch in range(n_epochs):
        train_loss, train_top1, train_top2 = train_one_epoch(model, head, train_loader, optimizer, loss_fn, rank, scheduler, detach_backbone=True)
        log_epoch = (epoch % max(1, eval_every_epochs) == 0) or (epoch == n_epochs - 1)
        val_loss, val_top1, val_top2, val_bal = (0.0, 0.0, 0.0, 0.0)
        if log_epoch:
            val_loss, val_top1, val_top2, val_bal = evaluate(model, head, val_loader, rank, num_classes)
            # Early stopping on balanced accuracy
            if val_bal > best_val + 1e-8:
                best_val = val_bal
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
        history.append((epoch, train_loss, train_top1, train_top2, val_loss, val_top1, val_top2, val_bal))
        if run is not None:
            wandb.log({
                "epoch": epoch,
                "trial/lr": lr,
                "trial/wd": weight_decay,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top2": train_top2,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top2": val_top2,
                "val_balanced_acc": val_bal,
                "probe_lr": optimizer.param_groups[0]["lr"],
            })
        if patience > 0 and no_improve >= patience:
            break

    if run is not None:
        run.finish()
    # return best val metric and history
    return best_val, best_epoch, history


def train_model(rank, world_size, cfg: DictConfig):
    set_random_seed(cfg.system.random_seed)
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== W&B =====
    if (not getattr(cfg, "sweep", {}).get("enable", False)) and rank == 0 and cfg.wandb.use:
        sanitized_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        wandb.init(
            project="ssl-position-between-objects",
            config=sanitized_cfg,
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group="seed: " + str(cfg.system.random_seed),
        )

    # ===== Data =====
    train_loader = build_loader(cfg.dataset, "train", cfg.batch_size, world_size, seed=cfg.system.random_seed)
    val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size, 1, seed=cfg.system.random_seed)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1, seed=cfg.system.random_seed)
    # print train, val, test dataset sizes
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # class_counts_train = [0, 0, 0, 0]
    # class_counts_val = [0, 0, 0, 0]
    # class_counts_test = [0, 0, 0, 0]
    # for batch in train_loader:
    #     labels = batch["label"]
    #     for label in labels:
    #         class_counts_train[label.item()] += 1
    # for batch in val_loader:
    #     labels = batch["label"]
    #     for label in labels:
    #         class_counts_val[label.item()] += 1
    # for batch in test_loader:
    #     labels = batch["label"]
    #     for label in labels:
    #         class_counts_test[label.item()] += 1
    # print('Train class counts:', class_counts_train)
    # print('Sum of train class counts:', sum(class_counts_train))
    # print('Val class counts:', class_counts_val)
    # print('Sum of val class counts:', sum(class_counts_val))
    # print('Test class counts:', class_counts_test)
    # print('Sum of test class counts:', sum(class_counts_test))
    # print('Sum of class counts:', sum(class_counts_train) + sum(class_counts_val) + sum(class_counts_test))

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
        exp_path = Path(__file__).parent / f"position_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Optional sweep mode =====
    if getattr(cfg, "sweep", {}).get("enable", False):
        if world_size > 1 and rank != 0:
            # avoid duplicate work across ranks
            return
        # Gather sweep params
        lrs = list(getattr(cfg.sweep, "learning_rates", [cfg.optimizer.probe_lr]))
        wds = list(getattr(cfg.sweep, "weight_decays", [cfg.optimizer.weight_decay]))
        patience = int(getattr(cfg.sweep, "patience", 3))
        eval_every = int(getattr(cfg.sweep, "eval_every_epochs", 1))
        n_epochs = int(cfg.optimizer.n_epochs)
        warmup_epochs = float(cfg.optimizer.warmup_epochs)

        model_name = model.checkpoint_name
        patch_size = model.patch_size
        layer = model.layer
        output = model.output

        sweep_rows = []
        best_overall = -1.0
        best_cfg = None
        group_name = f"sweep-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trial_idx = 0
        for lr in lrs:
            for wd in wds:
                name_suffix = f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trial_name = f"{name_suffix}_trial_{trial_idx:03d}_lr{lr}_wd{wd}"
                best_val, best_epoch, history = run_trial(
                    rank,
                    model,
                    feat_dim,
                    train_loader,
                    val_loader,
                    getattr(cfg.probe, "num_classes", 4),
                    lr,
                    wd,
                    n_epochs,
                    warmup_epochs,
                    cfg.wandb.use,
                    group_name,
                    trial_name,
                    patience=patience,
                    eval_every_epochs=eval_every,
                )
                sweep_rows.append([
                    datetime.now().strftime("%d%m%Y-%H%M"),
                    model_name,
                    patch_size,
                    str(layer),
                    output,
                    lr,
                    wd,
                    n_epochs,
                    best_val,
                    best_epoch,
                ])
                if best_val > best_overall:
                    best_overall = best_val
                    best_cfg = {"lr": lr, "wd": wd}
                trial_idx += 1

        # write sweep CSV
        result_dir = os.path.join(f"{cfg.output_dir}", "position_between_objects")
        os.makedirs(result_dir, exist_ok=True)
        csv_path = os.path.join(result_dir, "position_between_objects_sweep_unreal.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            import csv as _csv
            w = _csv.writer(f)
            if new_file:
                w.writerow([
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
                ])
            for r in sweep_rows:
                w.writerow(r)

        # final fit on train+val with best hyperparams and test eval
        if getattr(cfg.sweep, "final_fit", True) and best_cfg is not None:
            trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
            # train a fresh head using best cfg
            best_val, best_epoch, _ = run_trial(
                rank,
                model,
                feat_dim,
                trainval_loader,
                val_loader,  # still monitor val during final fit, but we will report test
                getattr(cfg.probe, "num_classes", 4),
                best_cfg["lr"],
                best_cfg["wd"],
                n_epochs,
                warmup_epochs,
                use_wandb=cfg.wandb.use,
                wandb_group=group_name,
                trial_name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_final_fit",
                patience=patience,
                eval_every_epochs=eval_every,
            )
            # evaluate on test
            # re-instantiate head with best cfg and train again to last best epoch? Already trained above
            # For a clean test, we can evaluate using the last trained head from run_trial
            # But run_trial discards head; do a short training to recreate then test
            # Simpler: create a new head and train for best_epoch epochs
            head_test = instantiate({"_target_": "evals.models.probes.ClassificationHead", "feat_dim": feat_dim, "num_classes": getattr(cfg.probe, "num_classes", 4), "use_layernorm": True}).to(rank)
            opt = torch.optim.AdamW([{"params": head_test.parameters(), "lr": best_cfg["lr"], "weight_decay": best_cfg["wd"]}])
            total_steps = (best_epoch + 1) * max(1, len(trainval_loader))
            warm_steps = int(warmup_epochs * max(1, len(trainval_loader)))
            sch = LambdaLR(opt, lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warm_steps)))
            loss_fn = torch.nn.CrossEntropyLoss()
            for ep in range(best_epoch + 1):
                train_one_epoch(model, head_test, trainval_loader, opt, loss_fn, rank, sch, detach_backbone=True)
            test_loss, test_top1, test_top2, test_bal = evaluate(model, head_test, test_loader, rank, getattr(cfg.probe, "num_classes", 4))

            with open(csv_path, "a", newline="") as f:
                import csv as _csv
                w = _csv.writer(f)
                w.writerow([
                    datetime.now().strftime("%d%m%Y-%H%M"),
                    model_name,
                    patch_size,
                    str(layer),
                    output,
                    f"final_lr={best_cfg['lr']}",
                    f"final_wd={best_cfg['wd']}",
                    best_epoch + 1,
                    f"test_bal_acc={test_bal}",
                    "final",
                ])
        return

    # ===== Training Loop =====
    for epoch in range(cfg.optimizer.n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_top1, train_top2 = train_one_epoch(
            model, head, train_loader, optimizer, loss_fn, rank, scheduler, detach_backbone=True
        )
        # scheduler.step()

        val_loss, val_top1, val_top2, val_bal = evaluate(
            model, head, val_loader, rank, getattr(cfg.probe, "num_classes", 4)
        )

        if rank == 0:
            logger.info(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} top1 {train_top1:.4f} top2 {train_top2:.4f} | "
                f"val loss {val_loss:.4f} top1 {val_top1:.4f} top2 {val_top2:.4f} bal {val_bal:.4f}"
            )
            if cfg.wandb.use:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_top1": train_top1,
                        "train_top2": train_top2,
                        "val_loss": val_loss,
                        "val_top1": val_top1,
                        "val_top2": val_top2,
                        "val_balanced_acc": val_bal,
                        "probe_lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # Optional: log eval images grid with predictions vs ground truth
            if (
                hasattr(cfg.wandb, "eval_images")
                and cfg.wandb.eval_images
                and len(val_loader) > 0
            ):
                try:
                    model.eval()
                    head.eval()
                    # deterministically fetch the first batch for consistency across epochs
                    val_iter = iter(val_loader)
                    batch = next(val_iter)
                    images = batch["image"].to(rank)
                    labels = batch["label"].to(rank)
                    with torch.no_grad():
                        feats = model(images)
                        logits = head(feats)
                        preds = logits.argmax(dim=1)
                    # take first 10
                    k = min(10, images.size(0))
                    panels = []
                    # human-readable mapping for Unreal task
                    id2name = {0: "Front", 1: "Back", 2: "Left", 3: "Right", 4: "Ambiguous"}
                    for i in range(k):
                        img = images[i].detach().cpu()
                        # de-normalize ImageNet
                        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                        img_vis = (img * std + mean).clamp(0, 1)
                        gt = int(labels[i].item())
                        pr = int(preds[i].item())
                        gt_name = id2name.get(gt, str(gt))
                        pr_name = id2name.get(pr, str(pr))
                        caption = f"GT: {gt_name} | PR: {pr_name}"
                        panels.append(wandb.Image(img_vis, caption=caption))
                    wandb.log({"eval_samples": panels, "epoch": epoch})
                except Exception as _e:
                    logger.warning(f"Failed to log eval images: {_e}")
    # test eval
    test_loss, test_top1, test_top2, test_bal = evaluate(model, head, test_loader, rank, getattr(cfg.probe, "num_classes", 4))
    if rank == 0:
        logger.info(f"test loss {test_loss:.4f} top1 {test_top1:.4f} top2 {test_top2:.4f} bal {test_bal:.4f}")
        if cfg.wandb.use:
            wandb.log({"test_loss": test_loss, "test_top1": test_top1, "test_top2": test_top2, "test_balanced_acc": test_bal})

    # ===== Save CSV summary =====
    if rank == 0:
        model_name = model.module.checkpoint_name if isinstance(model, DDP) else model.checkpoint_name
        patch_size = model.module.patch_size if isinstance(model, DDP) else model.patch_size
        layer = model.module.layer if isinstance(model, DDP) else model.layer
        output = model.module.output if isinstance(model, DDP) else model.output

        # final eval for summary row
        val_loss, val_top1, val_top2, val_bal = evaluate(
            model, head, val_loader, 0, getattr(cfg.probe, "num_classes", 4)
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
            "Top2 Val",
            "Balanced Acc Val",
            "Top1 Test",
            "Top2 Test",
            "Balanced Acc Test",
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
            getattr(train_loader.dataset, "name", "unreal_position"),
            getattr(val_loader.dataset, "name", "unreal_position_val"),
            f"{val_top1*100:.2f}",
            f"{val_top2*100:.2f}",
            f"{val_bal*100:.2f}",
            f"{test_top1*100:.2f}",
            f"{test_top2*100:.2f}",
            f"{test_bal*100:.2f}",
        ]

        result_dir = os.path.join(f"{cfg.output_dir}", "position_between_objects")
        os.makedirs(result_dir, exist_ok=True)
        csv_path = os.path.join(result_dir, "position_between_objects_results_unreal_final.csv")
        is_new = not os.path.exists(csv_path)
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
