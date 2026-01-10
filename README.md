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

| **Model Name**       | **Backbone**         | **Dataset**    | **Source Link**                                                                                   |
|-----------------------|----------------------|----------------|---------------------------------------------------------------------------------------------------|
| Jigsaw               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| RotNet               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| NPID                 | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| SeLa-v2              | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| NPID++               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| PIRL                 | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| ClusterFit           | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| DeepCluster-v2       | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| SwAV                 | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| SimCLR               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| MoCo v2              | ResNet-50           | ImageNet-1K    | [MoCo v2 repository](https://github.com/facebookresearch/moco)                                   |
| SimSiam              | ResNet-50           | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |
| BYOL                 | ResNet-50           | ImageNet-1K    | [Unofficial BYOL repo](https://github.com/yaox12/BYOL-PyTorch)                                   |
| Barlow Twins         | ResNet-50           | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |
| DenseCL              | ResNet-50           | ImageNet-1K    | [DenseCL repository](https://github.com/WXinlong/DenseCL)                                        |
| DINO                 | ResNet-50/ViT-B/16  | ImageNet-1K    | [DINO repository](https://github.com/facebookresearch/dino)                                      |
| MoCo v3              | ResNet-50/ViT-B/16  | ImageNet-1K    | [MoCo v3 repository](https://github.com/facebookresearch/moco-v3)                                |
| iBOT                 | ViT-B/16            | ImageNet-1K    | [iBOT repository](https://github.com/bytedance/ibot)                                             |
| MAE                  | ViT-B/16            | ImageNet-1K    | [MAE repository](https://github.com/facebookresearch/mae)                                       |
| MaskFeat             | ViT-B/16            | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |


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

Obtabin Visualization
-----------
```python
python train_depth.py backbone=beit_v2_vitb16 +backbone.return_multilayer=True experiment_model=depth_beitv2_vitb16 system.port=12345 system.random_seed=10 system.num_gpus=1 batch_size=8 is_eval=true ckpt_path=<PATH_TO_CKPT>
```


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
- W&B logging follows the same pattern as other experiments (enable via `wandb.use=True`).
- Configure dataset options in `configs/dataset/imagenette.yaml` (set `name` to `imagenette`, `fgvcaircraft`, or `flowers102`).

### Position Between Objects (Unreal)
-----------
- Train a linear probe to classify the relative position of Target B with respect to Reference A from the camera’s perspective. Classes: Front, Back, Left, Right (Ambiguous labels are excluded by default).

```bash
python train_position_between_objects.py backbone=dino_b16 experiment_model=position_between_objects_dino_b16
```

- Metrics logged: Top‑1, Top‑2, Balanced Accuracy (val). Results CSV saved under `result/position_between_objects/position_between_objects_results_unreal_final.csv`.
- W&B logging follows the same pattern as other experiments.

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
- W&B logging follows the same pattern as other experiments (enable via `wandb.use=True`).
- Configure dataset options in `configs/dataset/navi.yaml`
- Configure probe options in `configs/probe/camera_pose_regressor.yaml`
- Configure experiment options in `configs/navi_camera_pose_training.yaml`

#### Important Notes:
- You can run the `launch_script/launch_position_object.py` to launch the experiments for all the backbones sequentially.
- Configure dataset root and options in `configs/dataset/unreal_position.yaml`
- Configure probe options in `configs/probe/classifier.yaml`
- Configure optimizer options in `configs/optimizer/twenty_epoch.yaml`
- Configure experiment options in `configs/position_between_objects_training.yaml`
- This evaluation is now only support following backbones in directory of `configs/backbone`
  - `clip_b16_laion`
  - `deit3_b16`
  - `dino_b16`
  - `dinov2_b14`
  - `dinov2_b14_reg`
  - `dinov3_b16`
  - `croco_b16`
  - `crocov2_b16`
  - `mae_b16`
  - `maskfeat_vitb16`
  - `vggt_l16`
  - `spa_b16`
- If you want to sweep the hyperparameters to search for optimal hyperparameters, you can enable the sweep in `configs/position_between_objects_training.yaml`
- return_cls means return the CLS token features from the backbone - [B, D]
- mean_pool means mean of patch tokens from the backbone - [B, (H/P x W/P), D].mean(dim=1) == [B, D]
- If both return_cls and mean_pool are True, it will take mean of all the tokens from the backbone - [B, 1 + (H/P x W/P), D].mean(dim=1) == [B, D]
- If both return_cls and mean_pool are False, it will not return any features from the backbone and raise an error.

### VGGT, SPA and DINOv3 Notes
- You need to install the repositories of VGGT, SPA and DINOv3 to use these backbones. Later, you need to define the location of the repositories in the `configs/backbone` folder.
- Please refer to following links for installation:
  - [VGGT](https://github.com/facebookresearch/vggt) - the weight are automatically downloaded from the huggingface repository.
  - [SPA](https://github.com/HaoyiZhu/SPA) - the weight are automatically downloaded from the huggingface repository.
  - [DINOv3](https://github.com/facebookresearch/dinov3) - you need to ask for permission to download the weight from the repository and define the location of the weights in the `configs/backbone` folder.
- You also need to install the dependencies of those repositories.