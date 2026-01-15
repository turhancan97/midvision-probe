import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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

project_directory = "/home/kargin/Projects/repositories/midvision-probe"

# Base command for running the evaluation script
base_command = "python render_scannet_correspondence.py backbone={model} model_name={model}"


def run_evaluation(model):
    try:
        command = base_command.format(model=model)

        print(f"Running command: {command}")

        subprocess.run(command, shell=True, check=True, cwd=project_directory)
        print(f"Completed evaluation: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to evaluate model {model}: {e}")


max_threads = 2

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(run_evaluation, model) for model in models]

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"An exception occurred: {exc}")
