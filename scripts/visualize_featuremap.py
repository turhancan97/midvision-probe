"""
Feature Map Visualization Script (Hydra-based)

Visualizes feature maps from various vision models by projecting
high-dimensional features into RGB color space using PCA.

Usage:
    python scripts/visualize_featuremap.py image_folder=/path/to/images
    python scripts/visualize_featuremap.py image_folder=/path/to/images backbones=[spa_b16,dinov2_b14]
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def resolve_mean_std(cfg: DictConfig) -> Tuple[List[float], List[float]]:
    """Resolve image normalization mean and std from config."""
    image_mean = cfg.preprocessing.get("image_mean", "imagenet")
    if isinstance(image_mean, (list, tuple)):
        mean = [float(m) for m in image_mean]
    elif image_mean == "imagenet":
        mean = [0.485, 0.456, 0.406]
    elif image_mean == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
    else:
        mean = [0.485, 0.456, 0.406]  # default to imagenet

    image_std = cfg.preprocessing.get("image_std", "imagenet")
    if isinstance(image_std, (list, tuple)):
        std = [float(s) for s in image_std]
    elif image_std == "imagenet":
        std = [0.229, 0.224, 0.225]
    elif image_std == "clip":
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        std = [0.229, 0.224, 0.225]  # default to imagenet

    return mean, std


def get_robust_pca(
    features: torch.Tensor,
    m: float = 2.0,
    remove_first_component: bool = False,
):
    """
    Perform robust PCA on features with outlier handling.

    Args:
        features: (N, C) tensor of features
        m: Outlier threshold - controls how many std dev outside for outliers
        remove_first_component: Whether to remove first PCA component (background)

    Returns:
        reduction_mat: (C, 3) PCA reduction matrix
        rgb_min: (3,) minimum values for normalization
        rgb_max: (3,) maximum values for normalization
    """
    assert len(features.shape) == 2, "features should be (N, C)"

    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat

    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()

    # Outlier handling using median absolute deviation
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev

    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except Exception:
        rgb_min = torch.tensor([colors.min(), colors.min(), colors.min()])
        rgb_max = torch.tensor([colors.max(), colors.max(), colors.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    pca_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_pca_stats: bool = False,
    outlier_threshold: float = 2.0,
    remove_first_component: bool = False,
):
    """
    Convert feature map to PCA-based RGB visualization.

    Args:
        feature_map: (1, h, w, C) feature map tensor
        pca_stats: Optional pre-computed PCA stats (reduction_mat, color_min, color_max)
        return_pca_stats: Whether to return PCA stats
        outlier_threshold: Threshold for outlier filtering
        remove_first_component: Whether to remove first PCA component

    Returns:
        pca_color: (h, w, 3) RGB image as numpy array
        pca_stats: (optional) tuple of PCA statistics
    """
    if feature_map.shape[0] != 1:
        feature_map = feature_map[None]

    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1]),
            m=outlier_threshold,
            remove_first_component=remove_first_component,
        )
    else:
        reduct_mat, color_min, color_max = pca_stats

    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)

    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


def load_backbone_config(backbone_name: str) -> DictConfig:
    """Load a backbone config from configs/backbone/."""
    config_path = Path(__file__).parent.parent / "configs" / "backbone" / f"{backbone_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Backbone config not found: {config_path}")
    return OmegaConf.load(config_path)


def get_model_display_name(backbone_cfg: DictConfig, backbone_name: str) -> str:
    """Extract a display name for the model from the config."""
    # Try to get checkpoint_name if available, otherwise use backbone_name
    if hasattr(backbone_cfg, "checkpoint_name"):
        return backbone_cfg.checkpoint_name
    return backbone_name


def visualize_model(
    model,
    images: torch.Tensor,
    device: torch.device,
    model_name: str,
    output_dir: Path,
    pca_cfg: DictConfig,
):
    """
    Extract features and create PCA visualization for a single model.

    Args:
        model: The backbone model
        images: (N, C, H, W) batch of images
        device: torch device
        model_name: Name for output file
        output_dir: Directory to save output
        pca_cfg: PCA configuration
    """
    model.eval()
    model = model.to(device)

    # Process images one at a time to avoid OOM
    feature_maps = []
    with torch.no_grad():
        for img in tqdm(images, desc=f"Extracting {model_name} features", leave=False):
            fm = model(img.unsqueeze(0).to(device)).contiguous()
            feature_maps.append(fm.cpu())

    feature_map = torch.cat(feature_maps, dim=0)

    # Interpolate to target size
    interp_size = pca_cfg.get("interpolation_size", 224)
    feature_map = torch.nn.functional.interpolate(
        feature_map, size=(interp_size, interp_size), mode="bilinear"
    )

    # Stack vertically for visualization
    feature_map = rearrange(feature_map, "n c h w -> 1 (n h) w c")

    # Apply PCA
    feature_pca = get_pca_map(
        feature_map,
        outlier_threshold=pca_cfg.get("outlier_threshold", 2.0),
        remove_first_component=pca_cfg.get("remove_first_component", False),
    )
    feature_pca = (feature_pca * 255).astype(np.uint8).squeeze()

    # Save output
    output_path = output_dir / f"{model_name}_feature_map_vis.png"
    iio.imwrite(str(output_path), feature_pca)
    print(f"  Saved: {output_path}")

    # Free memory
    del feature_map, feature_pca, model
    torch.cuda.empty_cache()


@hydra.main(
    config_path="../configs",
    config_name="featuremap_visualization",
    version_base=None,
)
def main(cfg: DictConfig):
    # Validate required parameters
    if not cfg.image_folder:
        raise ValueError("image_folder must be specified. Use: image_folder=/path/to/images")

    image_folder = Path(cfg.image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(cfg.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load and preprocess images
    print(f"Loading images from: {image_folder}")
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    images = [
        Image.fromarray(iio.imread(os.path.join(image_folder, f))[..., :3])
        for f in image_files
    ]
    print(f"Found {len(images)} images")

    # Build transform
    mean, std = resolve_mean_std(cfg)
    img_size = cfg.preprocessing.get("img_size", 896)
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    images_tensor = torch.stack([transform(img) for img in images])
    print(f"Preprocessed images to size: {img_size}x{img_size}")

    # Process each backbone
    backbones = list(cfg.backbones)
    print(f"\nProcessing {len(backbones)} models: {backbones}\n")

    for backbone_name in backbones:
        print(f"Processing: {backbone_name}")
        try:
            # Load backbone config
            backbone_cfg = load_backbone_config(backbone_name)

            # Instantiate model
            model = instantiate(backbone_cfg)
            for p in model.parameters():
                p.requires_grad = False

            # Get display name for output
            model_display_name = get_model_display_name(backbone_cfg, backbone_name)

            # Visualize
            visualize_model(
                model=model,
                images=images_tensor,
                device=device,
                model_name=model_display_name,
                output_dir=output_dir,
                pca_cfg=cfg.pca,
            )

        except Exception as e:
            print(f"  Error processing {backbone_name}: {e}")
            continue

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
