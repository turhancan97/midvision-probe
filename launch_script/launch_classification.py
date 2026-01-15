import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of backbones to evaluate
models = [
    "dino_b16",
    "dinov2_b14",
    "dinov2_b14_reg",
    "dinov2_l14_reg",
    "dinov3_b16",
    "croco_b16",
    "crocov2_b16",
    "mae_b16",
    "maskfeat_vitb16",
    "spa_b16",
    "spa_l16",
    "vggt_l16",
    "deit3_b16",
    "clip_b16_laion",
]

# Path to the project directory
project_directory = (
    "/home/kargin/Projects/repositories/midvision-probe"
)

# Base command for running the evaluation script
base_command = "python train_classification.py \
    backbone={model} \
    experiment_model={model} \
    backbone.return_cls=false \
    backbone.mean_pool=true \
    backbone.efficient_probe=false \
    "


# Function to run an evaluation job for a specific model
def run_evaluation(model):
    try:
        # Prepare the command with the model name
        command = base_command.format(model=model)

        # Print the command to confirm it's correct
        print(f"Running command: {command}")

        # Execute the command in the specific directory
        subprocess.run(command, shell=True, check=True, cwd=project_directory)
        print(f"Completed evaluation: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to evaluate model {model}: {e}")


max_threads = 1

# Use ThreadPoolExecutor to run evaluations in parallel
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    # Submit evaluation tasks to the pool
    futures = [executor.submit(run_evaluation, model) for model in models]

    # Wait for the tasks to complete
    for future in as_completed(futures):
        try:
            future.result()  # This will raise an exception if the task failed
        except Exception as exc:
            print(f"An exception occurred: {exc}")
