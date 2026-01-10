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
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from evals.datasets.builder import build_loader
from evals.utils.optim import cosine_decay_linear_warmup
from evals.utils.seed import set_random_seed


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) with zero subgradient at x == 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real part is non negative.
    Quaternions are expected in real-first format (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions in real-first format (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_distance_deg(q1: torch.Tensor, q2: torch.Tensor, degree: bool = False) -> torch.Tensor:
    """
    Geodesic distance between two quaternions, returned in degrees.
    Quaternions are normalized inside.
    """
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q1 = standardize_quaternion(q1)
    q2 = standardize_quaternion(q2)
    dot_product = torch.abs(torch.sum(q1 * q2, dim=-1)).clamp(-1.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot_product)
    if degree:
        return angle_rad * 180.0 / torch.pi
    return angle_rad


def translation_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Euclidean translation error."""
    return torch.norm(t1 - t2, dim=-1)


def rot_magnitude_deg(quat: torch.Tensor) -> torch.Tensor:
    """Rotation magnitude (deg) of a quaternion w.r.t identity."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    quat = standardize_quaternion(quat)
    q0 = quat[..., 0].clamp(-1.0, 1.0)
    return 2.0 * torch.acos(q0) * 180.0 / torch.pi


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """Denormalize ImageNet-normalized tensor image to [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)[:, None, None]
    return (img * std + mean).clamp(0, 1)


def extract_features(backbone: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Runs the frozen backbone and flattens/concats outputs to [B, D]."""
    feats = backbone(images)
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(feats, dim=-1)
    if feats.dim() > 2:
        feats = feats.view(feats.size(0), -1)
    return feats


def build_pose_targets(Rt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (quat, translation) from a batch of 4x4 pose matrices."""
    gt_quat = matrix_to_quaternion(Rt[:, :3, :3])
    gt_trans = Rt[:, :3, 3]
    return gt_quat, gt_trans


def pose_metrics(pred_pose: torch.Tensor, gt_quat: torch.Tensor, gt_trans: torch.Tensor):
    pred_quat = pred_pose[:, :4]
    pred_trans = pred_pose[:, 4:]
    rot_err = quaternion_distance_deg(pred_quat, gt_quat)
    trans_err = translation_distance(pred_trans, gt_trans)
    return rot_err, trans_err


def train_one_epoch(backbone, head, loader, optimizer, scheduler, device, loss_fn):
    backbone.eval()  # frozen encoder
    head.train()
    running_loss = 0.0
    running_rot = 0.0
    running_trans = 0.0
    n_samples = 0

    pbar = tqdm(loader)
    for batch in pbar:
        images0 = batch["image_0"].to(device, non_blocking=True)
        images1 = batch["image_1"].to(device, non_blocking=True)
        Rt = batch["Rt_01"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            feat0 = extract_features(backbone, images0)
            feat1 = extract_features(backbone, images1)

        pair_feats = torch.cat([feat0, feat1], dim=-1)
        pred = head(pair_feats)

        gt_quat, gt_trans = build_pose_targets(Rt)
        gt_pose = torch.cat([gt_quat, gt_trans], dim=-1)

        loss = loss_fn(pred, gt_pose)
        loss.backward()
        optimizer.step()
        scheduler.step()

        rot_err, trans_err = pose_metrics(pred.detach(), gt_quat, gt_trans)

        batch_size = images0.size(0)
        running_loss += loss.item() * batch_size
        running_rot += rot_err.sum().item()
        running_trans += trans_err.sum().item()
        n_samples += batch_size

        pbar.set_description(
            f"loss {loss.item():.4f} | rot {rot_err.mean().item():.2f}deg | trans {trans_err.mean().item():.3f}"
        )

    return running_loss / n_samples, running_rot / n_samples, running_trans / n_samples


@torch.no_grad()
def evaluate(
    backbone,
    head,
    loader,
    device,
    loss_fn,
    *,
    log_images: bool = False,
    max_images: int = 0,
    epoch: int | None = None,
    run=None,
):
    backbone.eval()
    head.eval()
    running_loss = 0.0
    running_rot = 0.0
    running_trans = 0.0
    n_samples = 0
    image_panels = []

    for batch in loader:
        images0 = batch["image_0"].to(device, non_blocking=True)
        images1 = batch["image_1"].to(device, non_blocking=True)
        Rt = batch["Rt_01"].to(device, non_blocking=True)

        feat0 = extract_features(backbone, images0)
        feat1 = extract_features(backbone, images1)
        pair_feats = torch.cat([feat0, feat1], dim=-1)
        pred = head(pair_feats)

        gt_quat, gt_trans = build_pose_targets(Rt)
        gt_pose = torch.cat([gt_quat, gt_trans], dim=-1)

        loss = loss_fn(pred, gt_pose)
        rot_err, trans_err = pose_metrics(pred, gt_quat, gt_trans)

        batch_size = images0.size(0)
        running_loss += loss.item() * batch_size
        running_rot += rot_err.sum().item()
        running_trans += trans_err.sum().item()
        n_samples += batch_size

        if (
            log_images
            and run is not None
            and epoch is not None
            and len(image_panels) < max_images
        ):
            with torch.no_grad():
                pred_quat = pred[:, :4]
                pred_trans = pred[:, 4:]
                rot_pred = rot_magnitude_deg(pred_quat)
                rot_gt = rot_magnitude_deg(gt_quat)
                trans_pred = pred_trans.norm(dim=-1)
                trans_gt = gt_trans.norm(dim=-1)

            for i in range(min(batch_size, max_images - len(image_panels))):
                img0 = denormalize_image(images0[i].detach().cpu())
                img1 = denormalize_image(images1[i].detach().cpu())
                panel = torch.cat([img0, img1], dim=2)  # H x W*2 x 3
                caption = (
                    f"GT rot: {rot_gt[i].item():.2f} deg | GT trans: {trans_gt[i].item():.3f} | "
                    f"Pred rot: {rot_pred[i].item():.2f} deg | Pred trans: {trans_pred[i].item():.3f}"
                )
                image_panels.append(wandb.Image(panel.permute(1, 2, 0).numpy(), caption=caption))

    if log_images and run is not None and epoch is not None and image_panels:
        run.log({"val/image_pairs": image_panels, "epoch": epoch})

    return running_loss / n_samples, running_rot / n_samples, running_trans / n_samples


def train_model(cfg: DictConfig):
    set_random_seed(cfg.system.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.system.num_gpus > 1:
        logger.warning("Multi-GPU is not enabled for camera pose; using a single device.")

    sanitized_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    run = None
    if cfg.wandb.use:
        run = wandb.init(
            project="navi_camera_pose",
            config=sanitized_cfg,
            name=f"{cfg.experiment_name}_{cfg.experiment_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            group="seed: " + str(cfg.system.random_seed),
        )

    train_loader = build_loader(cfg.dataset, "train", cfg.batch_size, 1, pair_dataset=True)
    val_loader = build_loader(cfg.dataset, "valid", cfg.batch_size, 1, pair_dataset=True)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1, pair_dataset=True)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    backbone = instantiate(cfg.backbone)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(device)
    feat_dim = backbone.feat_dim if isinstance(backbone.feat_dim, int) else sum(backbone.feat_dim)
    pair_feat_dim = feat_dim * 2

    head = instantiate(cfg.probe, feat_dim=pair_feat_dim)
    head = head.to(device)

    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": cfg.optimizer.probe_lr, "weight_decay": cfg.optimizer.weight_decay}]
    )
    total_steps = cfg.optimizer.n_epochs * max(1, len(train_loader))
    warmup_steps = int(cfg.optimizer.warmup_epochs * max(1, len(train_loader)))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_decay_linear_warmup(step, total_steps, max(1, warmup_steps)),
    )
    loss_fn = torch.nn.MSELoss()

    exp_path = Path(__file__).parent / f"camera_pose_exps/{datetime.now().strftime('%d%m%Y-%H%M')}"
    exp_path.mkdir(parents=True, exist_ok=True)
    logger.add(exp_path / "training.log")
    logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    best_state = None
    best_val_loss = float("inf")
    best_val_rot = float("inf")
    best_val_trans = float("inf")
    best_epoch = -1

    for epoch in range(cfg.optimizer.n_epochs):
        log_pairs = bool(cfg.wandb.use and getattr(cfg.wandb, "eval_images", 0) > 0)
        max_images = int(getattr(cfg.wandb, "eval_images", 0))
        train_loss, train_rot, train_trans = train_one_epoch(
            backbone, head, train_loader, optimizer, scheduler, device, loss_fn
        )
        val_loss, val_rot, val_trans = evaluate(
            backbone,
            head,
            val_loader,
            device,
            loss_fn,
            log_images=log_pairs,
            max_images=max_images,
            epoch=epoch,
            run=run,
        )

        logger.info(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} rot {train_rot:.2f}deg trans {train_trans:.3f} | "
            f"val loss {val_loss:.4f} rot {val_rot:.2f}deg trans {val_trans:.3f}"
        )

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/rot_deg": train_rot,
                    "train/trans": train_trans,
                    "val/loss": val_loss,
                    "val/rot_deg": val_rot,
                    "val/trans": val_trans,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_val_rot = val_rot
            best_val_trans = val_trans
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())

    if best_state is not None:
        head.load_state_dict(best_state)

    test_loss, test_rot, test_trans = evaluate(backbone, head, test_loader, device, loss_fn)

    logger.info(
        f"best epoch {best_epoch} | val loss {best_val_loss:.4f} | "
        f"test loss {test_loss:.4f} rot {test_rot:.2f}deg trans {test_trans:.3f}"
    )
    if run is not None:
        wandb.log(
            {
                "best/epoch": best_epoch,
                "best/val_loss": best_val_loss,
                "test/loss": test_loss,
                "test/rot_deg": test_rot,
                "test/trans": test_trans,
            }
        )
        run.finish()

    model_name = getattr(backbone, "checkpoint_name", backbone.__class__.__name__)
    patch_size = getattr(backbone, "patch_size", "")
    layer = getattr(backbone, "layer", "")
    output = getattr(backbone, "output", "")
    probe_name = getattr(head, "name", head.__class__.__name__)

    result_dir = os.path.join(f"{cfg.output_dir}", "navi_camera_pose")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "navi_camera_pose_results.csv")
    is_new = not os.path.exists(csv_path)

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
        "Weight Decay",
        "Batch Size",
        "Train Dataset",
        "Val Dataset",
        "Best Epoch",
        "Val Loss",
        "Val Rot (deg)",
        "Val Trans",
        "Test Loss",
        "Test Rot (deg)",
        "Test Trans",
    ]

    row = [
        datetime.now().strftime("%d%m%Y-%H%M"),
        model_name,
        patch_size,
        str(layer),
        output,
        probe_name,
        cfg.system.random_seed,
        cfg.optimizer.n_epochs,
        cfg.optimizer.warmup_epochs,
        cfg.optimizer.probe_lr,
        cfg.optimizer.weight_decay,
        cfg.batch_size,
        getattr(train_loader.dataset, "name", "navi_train"),
        getattr(val_loader.dataset, "name", "navi_val"),
        best_epoch,
        f"{best_val_loss:.4f}",
        f"{best_val_rot:.4f}",
        f"{best_val_trans:.4f}",
        f"{test_loss:.4f}",
        f"{test_rot:.4f}",
        f"{test_trans:.4f}",
    ]

    import csv

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(headers)
        writer.writerow(row)


@hydra.main(config_name="navi_camera_pose_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    train_model(cfg)


if __name__ == "__main__":
    main()
