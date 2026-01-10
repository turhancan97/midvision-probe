"""
Launch script for position between objects training.

Configuration is done via:
1. Environment variables (set in SLURM script): PERSPECTIVE, ENVIRONMENT, PROBE_TYPE
2. YAML config file (position_config.yaml): label mappings and hyperparameter defaults
3. Lists below: models and layers to evaluate
"""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import yaml


# =============================================================================
# EDITABLE LISTS - Modify these to run different models/layers
# =============================================================================

models = [
    # "dino_b16",
    # "dinov2_b14",
    # "dinov2_b14_reg",
    # "dinov2_l14_reg",
    # "dinov3_b16",
    "croco_b16",
    "crocov2_b16",
    # "mae_b16",
    # "maskfeat_vitb16",
    "spa_b16",
    # "spa_l16",
    # "vggt_l16",
    # "deit3_b16",
    # "clip_b16_laion",
]

layers = [
    # 1,
    # 3,
    # 5,
    # 7,
    # 9,
    # 11,
    # 13,
    # 15,
    # 17,
    # 19,
    # 21,
    # 23
    -1
]

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "position_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_environment_variables():
    """Read configuration from environment variables."""
    perspective = os.environ.get("PERSPECTIVE", "camera")
    environment = os.environ.get("ENVIRONMENT", "bridge")
    probe_type = os.environ.get("PROBE_TYPE", "EfficientProbing")
    
    return perspective, environment, probe_type


def get_labels(config, environment):
    """Get reference and target labels for the given environment."""
    env_config = config["environments"].get(environment)
    if env_config is None:
        raise ValueError(f"Invalid environment: {environment}. "
                        f"Valid options: {list(config['environments'].keys())}")
    return env_config["reference_label"], env_config["target_label"]


def get_hyperparameters(config, perspective, probe_type):
    """Get hyperparameters based on perspective and probe type."""
    perspective_config = config["hyperparameters"].get(perspective)
    if perspective_config is None:
        raise ValueError(f"Invalid perspective: {perspective}. "
                        f"Valid options: {list(config['hyperparameters'].keys())}")
    
    probe_config = perspective_config.get(probe_type)
    if probe_config is None:
        raise ValueError(f"Invalid probe_type: {probe_type}. "
                        f"Valid options: EfficientProbing, ABMILPHead, ClassificationHead")
    
    return {
        "ambiguity_degree": perspective_config["ambiguity_degree"],
        "return_cls": probe_config["return_cls"],
        "mean_pool": probe_config["mean_pool"],
        "efficient_probe": probe_config["efficient_probe"],
        "epochs": probe_config["epochs"],
        "warmup_epochs": probe_config["warmup_epochs"],
        "dropout_rate": config["probe"]["dropout_rate"],
    }


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

PROJECT_DIR = "/home/kargin/Projects/repositories/midvision-probe"

BASE_COMMAND = """python train_position_between_objects_with_cache.py \
    backbone={model} \
    experiment_model={model} \
    experiment_name=SSL_Position_Between_Objects_{environment}_{probe_type} \
    probe._target_=evals.models.probes.{probe_type} \
    probe.dropout_rate={dropout_rate} \
    backbone.return_cls={return_cls} \
    backbone.mean_pool={mean_pool} \
    backbone.efficient_probe={efficient_probe} \
    backbone.layer={layer} \
    optimizer.n_epochs={epochs} \
    optimizer.warmup_epochs={warmup_epochs} \
    environment={environment} \
    dataset.perspective={perspective} \
    dataset.reference_label={reference_label} \
    dataset.target_label={target_label}"""


def run_evaluation(model, layer, perspective, environment, reference_label, 
                   target_label, probe_type, params):
    """Run a single evaluation job."""
    try:
        command = BASE_COMMAND.format(
            model=model,
            layer=layer,
            perspective=perspective,
            environment=environment,
            reference_label=reference_label,
            target_label=target_label,
            probe_type=probe_type,
            **params
        )
        
        print(f"\n{'='*60}")
        print(f"Running: {model} | layer={layer}")
        print(f"{'='*60}")
        print(f"Command: {command}\n")
        
        subprocess.run(command, shell=True, check=True, cwd=PROJECT_DIR)
        print(f"✓ Completed: {model} | layer={layer}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {model} | layer={layer} | Error: {e}")


def main():
    # Load configuration
    config = load_config()
    perspective, environment, probe_type = get_environment_variables()
    
    # Get labels and hyperparameters
    reference_label, target_label = get_labels(config, environment)
    params = get_hyperparameters(config, perspective, probe_type)
    
    # Print configuration summary
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Perspective:     {perspective}")
    print(f"Environment:     {environment}")
    print(f"Probe Type:      {probe_type}")
    print(f"Reference Label: {reference_label}")
    print(f"Target Label:    {target_label}")
    print(f"Models:          {models}")
    print(f"Layers:          {layers}")
    print(f"Hyperparameters: {params}")
    print("="*60 + "\n")
    
    # Run evaluations (sequential execution with ThreadPoolExecutor max_workers=1)
    max_threads = 1
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(
                run_evaluation,
                model, layer, perspective, environment,
                reference_label, target_label, probe_type, params
            )
            for model in models
            for layer in layers
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An exception occurred: {exc}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60} minutes")
