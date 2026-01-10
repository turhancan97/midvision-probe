from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image

from evals.datasets.unreal_position import _load_positions, _matching_json_for_image, LABEL_TO_INDEX


@dataclass
class AngleBinResult:
    start_deg: float
    end_deg: float
    center_deg: float
    count: int
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    recall: Optional[float]
    precision: Optional[float]
    f1: Optional[float]


def _iter_dataset_indices(dataset) -> Iterable[int]:
    split = dataset.split
    if split in ("valid", "val"):
        yield from dataset.idx_val
    elif split in ("test",):
        yield from dataset.idx_test
    elif split in ("train",):
        yield from dataset.idx_train
    elif split in ("trainval",):
        yield from dataset.idx_trainval
    else:
        raise ValueError(f"Unsupported split {split!r} for angle analysis.")


def _compute_divergence_for_sample(
    image_path: Path,
    reference_label: str,
    target_label: str,
    human_label: Optional[str],
) -> Optional[float]:
    json_path = _matching_json_for_image(image_path)
    if json_path is None:
        return None
    positions = _load_positions(
        json_path,
        reference_label=reference_label,
        target_label=target_label,
        human_label=human_label,
    )
    if not positions:
        return None

    camera = positions.get("camera")
    human = positions.get("human")
    reference = positions.get("reference")
    target = positions.get("target")
    if camera is None or human is None or reference is None or target is None:
        return None

    center = (reference + target) / 2.0
    cam_vec = center - camera
    human_vec = center - human

    cam_norm = np.linalg.norm(cam_vec)
    human_norm = np.linalg.norm(human_vec)
    if cam_norm < 1e-6 or human_norm < 1e-6:
        return None

    cam_unit = cam_vec / cam_norm
    human_unit = human_vec / human_norm

    dot = float(np.clip(np.dot(cam_unit, human_unit), -1.0, 1.0))
    det = float(cam_unit[0] * human_unit[1] - cam_unit[1] * human_unit[0])
    angle = math.degrees(math.atan2(det, dot))
    # map from (-180, 180] to [0, 360)
    angle = (angle + 360.0) % 360.0
    return angle


def collect_perspective_divergence_angles(dataset) -> Tuple[List[Optional[float]], List[int]]:
    angles: List[Optional[float]] = []
    real_indices: List[int] = []
    for real_idx in _iter_dataset_indices(dataset):
        image_path, _ = dataset.samples[real_idx]
        angle = _compute_divergence_for_sample(
            image_path,
            reference_label=dataset.reference_label,
            target_label=dataset.target_label,
            human_label=getattr(dataset, "human_label", None),
        )
        angles.append(angle)
        real_indices.append(real_idx)
    return angles, real_indices


def collect_predictions(head, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    head.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for feats, lbls in loader:
            feats = feats.to(device, non_blocking=True)
            logits = head(feats)
            preds.append(torch.argmax(logits, dim=1).cpu())
            labels.append(lbls.cpu())
    if not preds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pred_tensor = torch.cat(preds).numpy()
    label_tensor = torch.cat(labels).numpy()
    return pred_tensor, label_tensor


FOCUS_CLASSES: Dict[str, List[int]] = {
    "lateral": [LABEL_TO_INDEX["Left"], LABEL_TO_INDEX["Right"]],
    "sagittal": [LABEL_TO_INDEX["Front"], LABEL_TO_INDEX["Back"]],
}


def _apply_class_focus(
    preds: np.ndarray,
    labels: np.ndarray,
    angles: Sequence[Optional[float]],
    real_indices: Sequence[int],
    focus: str,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[float]], List[int]]:
    if not focus:
        focus = "all"
    focus = focus.lower()
    if focus == "all":
        return preds, labels, list(angles), list(real_indices)
    if focus not in FOCUS_CLASSES:
        raise ValueError(f"Unsupported class_focus '{focus}'. Expected one of: all, lateral, sagittal.")
    allowed = FOCUS_CLASSES[focus]
    mask = np.isin(labels, allowed)
    if mask.sum() == 0:
        return preds[:0], labels[:0], [], []
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]
    mask_list = mask.tolist()
    filtered_angles = [angle for angle, keep in zip(angles, mask_list) if keep]
    filtered_real_indices = [idx for idx, keep in zip(real_indices, mask_list) if keep]
    return filtered_preds, filtered_labels, filtered_angles, filtered_real_indices


def _compute_classification_metrics(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float, float]:
    if preds.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = float(np.mean(preds == labels))
    classes = np.union1d(preds, labels)

    recalls_for_bal = []
    recalls_macro = []
    precisions = []
    f1_scores = []

    for cls in classes:
        cls = int(cls)
        pred_mask = preds == cls
        label_mask = labels == cls
        tp = int(np.logical_and(pred_mask, label_mask).sum())
        pred_count = int(pred_mask.sum())
        label_count = int(label_mask.sum())

        recall = tp / label_count if label_count > 0 else 0.0
        precision = tp / pred_count if pred_count > 0 else 0.0
        if label_count > 0:
            recalls_for_bal.append(recall)
        recalls_macro.append(recall)
        precisions.append(precision)
        if precision + recall > 0:
            f1_scores.append((2 * precision * recall) / (precision + recall))
        else:
            f1_scores.append(0.0)

    balanced = float(np.mean(recalls_for_bal)) if recalls_for_bal else 0.0
    macro_recall = float(np.mean(recalls_macro)) if recalls_macro else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return accuracy, balanced, macro_recall, macro_precision, macro_f1


def bin_angle_accuracies(
    angles: Sequence[Optional[float]],
    preds: np.ndarray,
    labels: np.ndarray,
    bin_size_deg: float,
    min_count: int,
) -> Tuple[List[AngleBinResult], np.ndarray, np.ndarray, np.ndarray]:
    if preds.shape[0] != len(angles) or labels.shape[0] != len(angles):
        raise ValueError("Length mismatch between angles and predictions.")

    valid_angles = np.array([a if a is not None else np.nan for a in angles], dtype=float)
    valid_mask = ~np.isnan(valid_angles)
    num_bins = max(1, int(round(360.0 / bin_size_deg)))
    bin_edges = np.linspace(0.0, 360.0, num_bins + 1)
    bin_ids = np.digitize(valid_angles, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    results: List[AngleBinResult] = []
    for idx in range(num_bins):
        start = bin_edges[idx]
        end = bin_edges[idx + 1]
        center = 0.5 * (start + end)
        mask = (bin_ids == idx) & valid_mask
        count = int(mask.sum())
        accuracy: Optional[float]
        balanced: Optional[float]
        macro_recall: Optional[float]
        macro_precision: Optional[float]
        macro_f1: Optional[float]
        if count >= min_count and count > 0:
            accuracy, balanced, macro_recall, macro_precision, macro_f1 = _compute_classification_metrics(
                preds[mask], labels[mask]
            )
        else:
            accuracy = None
            balanced = None
            macro_recall = None
            macro_precision = None
            macro_f1 = None
        results.append(
            AngleBinResult(
                start_deg=start,
                end_deg=end,
                center_deg=center,
                count=count,
                accuracy=accuracy,
                balanced_accuracy=balanced,
                recall=macro_recall,
                precision=macro_precision,
                f1=macro_f1,
            )
        )
    return results, bin_edges, bin_ids, valid_mask


def plot_radar_accuracy(results: Sequence[AngleBinResult], output_path: Path):
    if not results:
        return

    centers = np.array([r.center_deg for r in results], dtype=float)
    accuracies = np.array([np.nan if r.accuracy is None else r.accuracy for r in results], dtype=float)
    counts = np.array([r.count for r in results], dtype=int)

    # Close the radar loop by appending the first point.
    theta = np.deg2rad(np.concatenate([centers, centers[:1]]))
    values = np.concatenate([accuracies, accuracies[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.plot(theta, values, color="tab:blue", linewidth=2)
    ax.fill(theta, values, color="tab:blue", alpha=0.2)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticks(np.deg2rad(centers))
    xtick_labels = [f"{int(round(c))}°" for c in centers]
    ax.set_xticklabels(xtick_labels)
    ax.set_title("Top-1 Accuracy vs. Perspective Divergence", va="bottom")

    # Annotate counts along the perimeter.
    for center_rad, acc, count in zip(np.deg2rad(centers), accuracies, counts):
        if np.isnan(acc):
            ax.scatter([center_rad], [ax.get_ylim()[1]], color="red", marker="x", s=60)
            ax.text(
                center_rad,
                ax.get_ylim()[1],
                f"{count}",
                color="red",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        else:
            ax.text(
                center_rad,
                acc + 0.05,
                f"{count}",
                color="tab:blue",
                fontsize=8,
                ha="center",
                va="bottom",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_bin_table(results: Sequence[AngleBinResult], output_path: Path):
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(
            "start_deg,end_deg,center_deg,count,top1_accuracy,balanced_accuracy,macro_recall,macro_precision,macro_f1\n"
        )
        for r in results:
            acc_str = "" if r.accuracy is None else f"{r.accuracy:.6f}"
            bal_str = "" if r.balanced_accuracy is None else f"{r.balanced_accuracy:.6f}"
            rec_str = "" if r.recall is None else f"{r.recall:.6f}"
            prec_str = "" if r.precision is None else f"{r.precision:.6f}"
            f1_str = "" if r.f1 is None else f"{r.f1:.6f}"
            f.write(
                f"{r.start_deg:.2f},{r.end_deg:.2f},{r.center_deg:.2f},{r.count},{acc_str},{bal_str},{rec_str},{prec_str},{f1_str}\n"
            )


def save_bin_sample_grid(
    results: Sequence[AngleBinResult],
    bin_ids: np.ndarray,
    valid_mask: np.ndarray,
    filtered_real_indices: Sequence[int],
    dataset,
    output_path: Path,
):
    if not results:
        return

    num_bins = len(results)
    cols = min(4, num_bins)
    rows = int(np.ceil(num_bins / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if isinstance(axes, np.ndarray):
        axes_iter = axes.flatten()
    else:
        axes_iter = [axes]

    for idx, ax in enumerate(axes_iter):
        if idx >= num_bins:
            ax.axis("off")
            continue
        res = results[idx]
        mask = (bin_ids == idx) & valid_mask
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            ax.axis("off")
            ax.set_title(
                f"Bin {idx + 1}\n{res.start_deg:.0f}°-{res.end_deg:.0f}°\n(no samples)",
                fontsize=9,
            )
            continue
        sample_idx = int(candidates[0])
        real_idx = filtered_real_indices[sample_idx]
        image_path, _ = dataset.samples[real_idx]
        try:
            with Image.open(image_path).convert("RGB") as img:
                image = np.array(img)
        except Exception:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        ax.imshow(image)
        ax.axis("off")
        acc_text = "n/a" if res.accuracy is None else f"{res.accuracy:.2f}"
        ax.set_title(
            f"Bin {idx + 1}\n{res.start_deg:.0f}°-{res.end_deg:.0f}°\ncount={res.count}, acc={acc_text}",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_perspective_divergence_analysis(
    head,
    feature_loader: DataLoader,
    raw_dataset,
    device: torch.device,
    output_plot: Path,
    output_table: Path,
    output_grid: Path,
    bin_size_deg: float,
    min_count: int,
    class_focus: str = "all",
):
    preds, labels = collect_predictions(head, feature_loader, device)
    if preds.size == 0:
        return

    angles, real_indices = collect_perspective_divergence_angles(raw_dataset)
    if len(angles) != preds.size:
        raise RuntimeError(
            f"Angle list length {len(angles)} does not match number of predictions {preds.size}."
        )

    preds, labels, angles, real_indices = _apply_class_focus(preds, labels, angles, real_indices, class_focus)
    if preds.size == 0:
        return

    results, _, bin_ids, valid_mask = bin_angle_accuracies(
        angles, preds, labels, bin_size_deg=bin_size_deg, min_count=min_count
    )
    plot_radar_accuracy(results, output_plot)
    save_bin_table(results, output_table)
    save_bin_sample_grid(results, bin_ids, valid_mask, real_indices, raw_dataset, output_grid)
