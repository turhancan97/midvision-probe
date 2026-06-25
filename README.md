Probing the Mid-level Vision Capabilities of Self-supervised Learning Methods
=============================================================================

This repository contains official implementation of the code for the paper [Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474) which presents an analysis of the mid level perception of pretrained SSLs.


[Xuweiyi Chen](https://xuweiyichen.github.io/), [Markus Marks](https://damaggu.github.io/), [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)

If you find this code useful, please consider citing:  
```text
@article{chen2024probingmidlevelvisioncapabilities,
      title={Probing the Mid-level Vision Capabilities of Self-Supervised Learning}, 
      author={Xuweiyi Chen and Markus Marks and Zezhou Cheng},
      year={2024},
      eprint={2411.17474},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17474}, 
}
```
**:warning: Note:** This is a cleanup version. Further edits and refinements are in progress. This note will be removed once the content has been finalized.


Model Checkpoints
-----------------

| **Model Name**       | **Backbone**         | **Dataset**                           | **Source Link**                                                                                  |
|----------------------|----------------------|---------------------------------------|--------------------------------------------------------------------------------------------------|
| MoCo v3              | ViT-B/16            | ImageNet-1K                            | [MoCo v3 repository](https://github.com/facebookresearch/moco-v3)                                |
| iBOT                 | ViT-B/16            | ImageNet-1K                            | [iBOT repository](https://github.com/bytedance/ibot)                                             |
| DINO                 | ViT-B/16            | ImageNet-1K                            | [DINO repository](https://github.com/facebookresearch/dino)                                      |
| DINOv2               | ViT-B/14            | LVD-142M                               | [DINOv2 repository](https://github.com/facebookresearch/dinov2)                                  |
| DINOv2-Reg           | ViT-B/14 - L/14     | LVD-142M                               | [DINOv2-Reg repository](https://github.com/facebookresearch/dinov2)                              |
| DINOv3               | ViT-B/16            | LVD-1689M                              | [DINOv3 repository](https://github.com/facebookresearch/dinov3)                                  |
| DINOv3 (timm)        | ViT-S/B/L/H+/7B     | LVD-1689M                              | [timm models on Hugging Face](https://huggingface.co/timm)                                        |
| MAE                  | ViT-B/16            | ImageNet-1K                            | [MAE repository](https://github.com/facebookresearch/mae)                                        |
| MaskFeat             | ViT-B/16            | ImageNet-1K                            | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |
| SPA                  | ViT-B/16            | ScanNet, Hypersim, and more...         | [SPA repository](https://github.com/HaoyiZhu/SPA)                                                |
| CroCo                | ViT-B/16            | Habitat, ScanNet, and more...          | [CroCo repository](https://github.com/naver/croco)                                               |
| CROCOV2              | ViT-B/16            | ARKitScenes, MegaDepth, and more...    | [CROCOV2 repository](https://github.com/naver/croco)                                             |
| VGGT                 | ViT-L/14            | Co3D, MegaDepth, and more...           | [VGG-T repository](https://github.com/facebookresearch/vggt)                                     |
| DeiT-3               | ViT-B/16            | ImageNet-1K                            | [DeiT-3 repository](https://github.com/facebookresearch/deit3)                                   |
| CLIP                 | ViT-B/16            | Web Image-Text (WIT)                   | [CLIP repository](https://github.com/openai/CLIP)                                                |


Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```bash
conda create -n mid-probe python=3.9 --yes
conda activate mid-probe
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -c conda-forge nb_conda_kernels=2.3.1

pip install -r requirements.txt
python setup.py develop

pip install protobuf==3.20.3 
pre-commit install
```


Finally, please follow the dataset download and preprocessing instructions [here](./data_processing/README.md).


Evaluation Experiments
-----------

We provide code to train the depth probes and evaluate the correspondence. All experiments use
hydra configs which can be found [here](./configs). Below are example commands for running the
evaluations with the DINO ViT-B/16 backbone.

```python
python train_depth.py backbone=dino_b16 +backbone.return_multilayer=True dataset=nyu
python train_snorm.py backbone=dino_b16 +backbone.return_multilayer=True dataset=nyu
python train_generic_objectness.py backbone=dino_b16 dataset=voc12
python evaluate_model_percepture.py backbone=dino_b16 experiment_model=dino_b16 system.random_seed=8 system.num_gpus=1 batch_size=8 dataset=twoafcdataset output_dir=<OUTPUT_PATH> backbone.return_cls=True

python evaluate_navi_correspondence.py +backbone=dino_b16
python evaluate_scannet_correspondence.py +backbone=dino_b16
```

Obtain Visualization
-----------
```python
python train_depth.py backbone=beit_v2_vitb16 +backbone.return_multilayer=True experiment_model=depth_beitv2_vitb16 system.port=12345 system.random_seed=10 system.num_gpus=1 batch_size=8 is_eval=true ckpt_path=<PATH_TO_CKPT>
```

Evaluate the Mid-level Vision Capabilities
-----------

### Depth Estimation
- Train a depth estimator to estimate the depth of the image. The task aims at predicting pixel-wise depth from monocular images.

```bash
python train_depth.py backbone=dino_b16 experiment_model=depth_dino_b16
```

- NAVI and NYU Depth V2 datasets are used for training and testing the depth estimator.


### Surface Normal Estimation
- Train a surface normal estimator to estimate the surface normal of the image. The task aims at predicting pixel-wise surface normal from monocular images.

```bash
python train_snorm.py backbone=dino_b16 experiment_model=snorm_dino_b16
```

- NAVI and NYU Depth V2 datasets are used for training and testing the surface normal estimator.

### Generic Objectness
- Train a generic objectness estimator to estimate the objectness of the image. Generic object segmentation (or figure-ground segmentation) refers to the task of separating objects (figure) from the surrounding background without any semantics. This mid-level vision task is different from semantic segmentation (i.e., assigning each pixel to a semantic category) which is commonly used for evaluating the high-level visual capabilities of VFMs.

```bash
python train_generic_objectness.py backbone=dino_b16 experiment_model=generic_objectness_dino_b16
```

- VOC07 and VOC12 datasets are used for training and testing the generic objectness estimator.


### Geometric Correspondence
- Train a geometric correspondence estimator to estimate the geometric correspondence of the image. The task aims at predicting the geometric correspondence of the image.

```bash
python evaluate_navi_correspondence.py +backbone=dino_b16 model_name=dino_b16
python evaluate_spair_correspondence.py backbone=dino_b16 model_name=dino_b16
python render_scannet_correspondence.py backbone=dino_b16 model_name=dino_b16
```

- Visualize the navi correspondence by running the following command:
```bash
python render_navi_correspondence.py +backbone=dino_b16 model_name=dino_b16
```

- NAVI, ScanNet, and SPair-71k datasets are used for training and testing the geometric correspondence estimator.

### Mid-Level Image Similarity
- Train a mid-level image similarity estimator to estimate the similarity of the images. The task aims at measuring the similarity of two images with mid-level variations (e.g., viewpoint).

```bash
python evaluate_model_percepture.py backbone=dino_b16 experiment_model=dino_b16
```

- NIGHTS dataset is used for training and testing the mid-level image similarity estimator.


Acknowledgments
-----------------

We would also like to acknowledge the following repositories and users for releasing very valuable
code and datasets: 

- [GeoNet](https://github.com/xjqi/GeoNet) for releasing the extracted surface normals for full NYU.  
- [Probe3D](https://github.com/mbanani/probe3d) for releasing probing algorithms for 3D foundation models.
- [Comparing evaluation protocols for self-supervised pre-training with image classification](https://github.com/XuweiyiChen/probing-mid-level-vision/tree/ssl-previous) for releasing a collection of Self-Supervised Learning methods and their usages.

--------------------------------

New additions:

### Linear Classification
-----------
- Train a linear probe for image classification using CLS token features from the backbone. Supports three datasets:
  - **Imagenette**: 10-class subset of ImageNet
  - **FGVCAircraft**: Fine-grained aircraft recognition (100 classes)
  - **Flowers102**: Oxford Flowers 102 categories

```bash
# Imagenette (default, 10 classes)
python train_classification.py backbone=dinov2_b14 experiment_model=classification_dinov2_b14 dataset.name=imagenette probe.num_classes=10

# FGVC Aircraft (100 classes)
python train_classification.py backbone=dinov2_b14 experiment_model=classification_dinov2_b14 dataset.name=fgvcaircraft probe.num_classes=100

# Flowers102 (102 classes)
python train_classification.py backbone=dinov2_b14 experiment_model=classification_dinov2_b14 dataset.name=flowers102 probe.num_classes=102
```

- Metrics logged: Top-1, Top-5, Balanced Accuracy (val). Results CSV saved under `result/linear_probe_classification/`.
- Imagenette, FGVCAircraft, and Flowers102 datasets are used for training and testing the linear probe for image classification.
- W&B logging follows the same pattern as other experiments (enable via `wandb.use=True`).
- Configure dataset options in `configs/dataset/imagenette.yaml` (set `name` to `imagenette`, `fgvcaircraft`, or `flowers102`).

### Position Between Objects (Unreal)
-----------
- Train a linear probe to classify the relative position of Target B with respect to Reference A from the camera’s perspective. Classes: Front, Back, Left, Right (Ambiguous labels are excluded by default).

```bash
python train_position_between_objects.py backbone=dino_b16 experiment_model=position_between_objects_dino_b16
```

- Metrics logged: Top‑1, Top‑2, Balanced Accuracy (val). Results CSV saved under `result/position_between_objects/position_between_objects_results_unreal_final.csv`.
- Unreal dataset is used for training and testing the linear, AbMILP, and efficient probe for position between objects.
- W&B logging follows the same pattern as other experiments.

### Position Between Objects Rebuttal (Real-World Folder Dataset)
-----------
- The previous SpatialSense rebuttal path was retired and replaced by a folder-based real-world dataset:
  - Root: `/shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images`
  - Class folders: `front`, `back`, `left`, `right`
- New dataset config key: `dataset=real_world_position` (`configs/dataset/real_world_position.yaml`).
- Supported split modes:
  - `random`: stratified per-class shuffle, deterministic by seed
  - `time_series`: per-class lexicographic order with strict 80/10/10 (train/valid/test)
  - `clip_block_random`: split by contiguous frame blocks, then shuffle blocks
  - `clip_block_time_series`: split by contiguous frame blocks in chronological order

- Rebuttal launcher (same script path, now real-world behavior):
```bash
python launch_script/spatialsense_rebuttal.py \
  --output-dir result_real_world_rebuttal \
  --backbones dino_b16,dinov2_b14_reg,vggt_l16 \
  --heads EfficientProbing,ABMILP,GAP \
  --split-modes random,time_series,clip_block_random,clip_block_time_series
```

- Seed control options:
  - `--seeds`: comma-separated seed list for this run (default: `8,42,123`)
  - `--tune-seed`: optional seed used in tuning stage (default: first value from `--seeds`)
  - `--clip-block-size`: block size used by clip-block split modes (default: `20`)
  - `--clip-id-source`: clip block grouping source (`filename_numeric` or `parent_folder`)
  - If only one seed is provided, launcher runs both tune and final stage for that seed.
  - If multiple seeds are provided, final stage excludes `--tune-seed` unless `--rerun-seed8-final` is set.

- Example: run only one seed
```bash
python launch_script/spatialsense_rebuttal.py \
  --output-dir result_real_world_rebuttal \
  --backbones vggt_l16 \
  --heads EfficientProbing \
  --split-modes random \
  --seeds 8
```

- Example: custom seed subset with explicit tuning seed
```bash
python launch_script/spatialsense_rebuttal.py \
  --output-dir result_real_world_rebuttal \
  --backbones vggt_l16 \
  --heads EfficientProbing,ABMILP,GAP \
  --split-modes random,time_series \
  --seeds 42,123 \
  --tune-seed 42
```

- SLURM launcher:
```bash
sbatch zrun_launch_train_spatialsense_rebuttal.sh
```

- Summarization:
```bash
python scripts/summarize_spatialsense_rebuttal.py \
  --results-csv result_real_world_rebuttal/position_between_objects/real_world_position/position_between_objects_results_real_world_position.csv \
  --output-dir result_real_world_rebuttal/position_between_objects/real_world_position/reports \
  --dataset-root /shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images
```

- Outputs:
  - Run CSV: `result_real_world_rebuttal/position_between_objects/real_world_position/position_between_objects_results_real_world_position.csv`
  - Reports: `result_real_world_rebuttal/position_between_objects/real_world_position/reports/`
  - Rebuttal best params: `result_real_world_rebuttal/real_world_rebuttal_best_params.json`

### Position Between Objects Transfer Rebuttal (Unreal LOTO + Few-shot)
-----------
- Added a transfer protocol for rebuttal experiments on Unreal SpaRRTa:
  - `loto_source_to_target`: train/val on non-holdout environments, test on holdout environment
  - `target_only`: train/val/test on holdout environment (used for few-shot adaptation)
- Dataset config: `configs/dataset/unreal_position_transfer.yaml`
- Dataset implementation: `evals/datasets/unreal_position_transfer.py`
- Training script: `train_position_between_objects_with_cache.py` (supports optional probe init via `ckpt_path`).

- Rebuttal launcher (full matrix):
```bash
python launch_script/unreal_loto_fewshot_rebuttal.py --output-dir result_unreal_loto_rebuttal
```

- SLURM launcher:
```bash
sbatch zrun_launch_train_unreal_loto_fewshot_rebuttal.sh
```

- Summarization:
```bash
python scripts/summarize_unreal_loto_fewshot_rebuttal.py \
  --results-csv result_unreal_loto_rebuttal/position_between_objects/unreal_position_transfer/position_between_objects_results_unreal_position_transfer.csv \
  --output-dir result_unreal_loto_rebuttal/position_between_objects/unreal_position_transfer/reports
```

- Locked matrix defaults in launcher:
  - Backbones: `vggt_l16`, `dinov2_l14_reg`
  - Heads: `EfficientProbing`, `ClassificationHead` (GAP)
  - Perspectives: `camera`, `human`
  - Holdout folds: `bridge_2`, `city_2`, `desert_2`, `forest_2`, `winter_town_2`
  - Excluded envs: `desert_nonhuman`
  - Seeds: `8, 42, 123`
  - Few-shot K: `10, 50, 100, 500` (+ `K=0` zero-shot point)

- Outputs:
  - Run CSV: `result_unreal_loto_rebuttal/position_between_objects/unreal_position_transfer/position_between_objects_results_unreal_position_transfer.csv`
  - Reports: `result_unreal_loto_rebuttal/position_between_objects/unreal_position_transfer/reports/`
  - Per-run saved predictions (`Predictions Path`) and optional zero-shot probe checkpoints (`Head Path`) are logged in CSV rows.

- Pair-swap mode (target-fixed reference-swap transfer):
  - Enable with launcher flags:
    - `--pair-swap-mode`
    - `--pair-swap-source-env <ENV>`
    - `--pair-swap-target-env <ENV>`
  - Behavior when enabled:
    - Uses only `dataset.environments=[source_env,target_env]`
    - Sets holdout to `target_env` (so `--holdout-folds` is ignored)
    - Zero-shot (`loto_source_to_target`): train/val on `source_env`, test on `target_env`
    - Few-shot (`target_only`): train/val/test on `target_env` for both `transfer` and `scratch`
  - Example (`desert_2 -> desert_3`):
```bash
python launch_script/unreal_loto_fewshot_rebuttal.py \
  --output-dir result_unreal_loto_rebuttal \
  --pair-swap-mode \
  --pair-swap-source-env desert_2 \
  --pair-swap-target-env desert_3
```
  - Equivalent SLURM config in `zrun_launch_train_unreal_loto_fewshot_rebuttal.sh`:
    - `PAIR_SWAP_MODE=true`
    - `PAIR_SWAP_SOURCE_ENV="desert_2"`
    - `PAIR_SWAP_TARGET_ENV="desert_3"`

### Equivariance (Unreal)

- Train a linear probe to regress the position of the camera.

```bash
python train_equivariance.py backbone=dino_b16 experiment_model=equivariance_dino_b16
```

- Metrics logged: RMSE, MSE, MAE. Results CSV saved under `result/equivariance`.
- W&B logging follows the same pattern as other experiments.

### Camera Pose Regression (NAVI)
-----------
- Train an MLP probe to regress the relative camera pose (rotation + translation) between pairs of images from the NAVI dataset. The pose is represented as a 7D vector: 4D quaternion for rotation and 3D translation vector.

```bash
python evaluate_navi_camera_pose.py backbone=dino_b16 experiment_model=camera_pose_dino_b16
```

- Metrics logged: Rotation error (degrees), Translation error (Euclidean distance), MSE loss. Results CSV saved under `result/navi_camera_pose/navi_camera_pose_results.csv`.
- NAVI dataset is used for training and testing the camera pose regression estimator.
- W&B logging follows the same pattern as other experiments (enable via `wandb.use=True`).
- Configure dataset options in `configs/dataset/navi.yaml`
- Configure probe options in `configs/probe/camera_pose_regressor.yaml`
- Configure experiment options in `configs/navi_camera_pose_training.yaml`

### Feature Map Visualization
-----------
- Visualize feature maps from various vision backbones by projecting high-dimensional features into RGB color space using PCA. Useful for qualitatively comparing what different models "see" in images.

```bash
# Visualize with default backbones (spa_b16, croco_b16, crocov2_b16, dinov2_b14)
python scripts/visualize_featuremap.py image_folder=/path/to/images

# Visualize with specific backbones
python scripts/visualize_featuremap.py image_folder=/path/to/images backbones=[dino_b16,dinov2_b14,mae_b16]

# Custom image size and PCA settings
python scripts/visualize_featuremap.py image_folder=/path/to/images preprocessing.img_size=448 pca.remove_first_component=true
```

- Output: PNG images with PCA-projected feature maps saved to `visualize/<timestamp>/` directory.
- Configure default settings in `configs/featuremap_visualization.yaml`
- PCA options:
  - `pca.outlier_threshold`: Controls outlier filtering (default: 2.0)
  - `pca.remove_first_component`: Remove first PCA component to suppress background (default: false)
  - `pca.interpolation_size`: Size to interpolate feature maps before PCA (default: 224)

#### Important Notes:
- You can run the `launch_script/launch_position_object.py` to launch the experiments for all the backbones sequentially.
- Configure dataset root and options in `configs/dataset/unreal_position.yaml`
- Configure probe options in `configs/probe/classifier.yaml`
- Configure optimizer options in `configs/optimizer/custom_epoch.yaml`
- Configure experiment options in `configs/position_between_objects_training.yaml`
- This evaluation is now only support following backbones in directory of `configs/backbone`
  - `clip_b16_laion`
  - `deit3_b16`
  - `dino_b16`
  - `dinov2_b14`
  - `dinov2_b14_reg`
  - `dinov3_b16`
  - `dinov3_timm`
  - `croco_b16`
  - `crocov2_b16`
  - `mae_b16`
  - `maskfeat_vitb16`
  - `vggt_l16`
  - `spa_b16` 
  - `spa_l16`
- If you want to sweep the hyperparameters to search for optimal hyperparameters, you can enable the sweep in `configs/position_between_objects_training.yaml`
- return_cls means return the CLS token features from the backbone - [B, D]
- mean_pool means mean of patch tokens from the backbone - [B, (H/P x W/P), D].mean(dim=1) == [B, D]
- If both return_cls and mean_pool are True, it will take mean of all the tokens from the backbone - [B, 1 + (H/P x W/P), D].mean(dim=1) == [B, D]
- If both return_cls and mean_pool are False, it will not return any features from the backbone and raise an error.
- If you want to evaluate models with different image sizes than 224x224, you need to change the `size_image` in the `configs/backbone` folder only for CroCo, CROCOV2, SPA. Other models are supported automatically adapted to the image size.

### VGGT, SPA, CroCo and DINOv3 Notes
- You need to install the repositories of VGGT, SPA, CroCo and local DINOv3 (`dinov3_b16`) to use those backbones. Later, you need to define the location of the repositories in the `configs/backbone` folder.
- Please refer to following links for installation:
  - [VGGT](https://github.com/facebookresearch/vggt) - the weight are automatically downloaded from the huggingface repository.
  - [SPA](https://github.com/HaoyiZhu/SPA) - the weight are automatically downloaded from the huggingface repository.
  - [CroCo](https://github.com/naver/croco) - the weight are automatically downloaded to the `ckpt_dir` folder.
  - [DINOv3](https://github.com/facebookresearch/dinov3) - you need to ask for permission to download the weight from the repository and define the location of the weights in the `configs/backbone` folder.
- For timm-based DINOv3 (`dinov3_timm`), no local DINOv3 repository is required. Weights are loaded via timm from Hugging Face model IDs.
- Quick validation command for timm integration:
  - `python scripts/validate_dinov3_timm.py --model-name vit_base_patch16_dinov3.lvd1689m --device auto`
- You also need to install the dependencies of those repositories.
