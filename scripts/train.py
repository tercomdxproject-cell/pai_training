# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Tatsuya Kamijo. All rights reserved.

"""
Train ACT (Action Chunking with Transformer) policy on the collected dataset.
"""

from pathlib import Path
import importlib.util
import wandb
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

def load_config_py(config_path: Path):
    spec = importlib.util.spec_from_file_location("crane_features", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def load_train_config(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# =====================
# Load parameters from YAML
# =====================
train_cfg = load_train_config(Path("configs/train_config.yaml"))
DATASET_ROOT = Path(train_cfg["dataset_root"])
REPO_ID = train_cfg["repo_id"]
OUTPUT_DIRECTORY = Path(train_cfg["output_directory"])
CONFIG_PATH = Path(train_cfg["config_path"])
DEVICE = torch.device(train_cfg["device"])
TRAINING_STEPS = train_cfg["training_steps"]
LOG_FREQ = train_cfg["log_freq"]
NUM_WORKERS = train_cfg["num_workers"]
BATCH_SIZE = train_cfg["batch_size"]
CHUNK_SIZE = train_cfg["chunk_size"]
N_ACTION_STEPS = train_cfg["n_action_steps"]
LEARNING_RATE = train_cfg["learning_rate"]
DELTA_TIMESTAMPS = train_cfg["delta_timestamps"]
# =====================

def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = load_config_py(CONFIG_PATH)
    features = config.features

    wandb.init(project="pai_crane_gazebo", config={
        "repo_id": REPO_ID,
        "dataset_root": str(DATASET_ROOT),
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": N_ACTION_STEPS,
        "device": str(DEVICE),
    })

    dataset_metadata = LeRobotDatasetMetadata(REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features: {input_features}")
    print(f"output_features: {output_features}")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS
    )
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(DEVICE)

    # Instantiate the dataset
    # 修正後:
    dataset = LeRobotDataset(
        REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=DELTA_TIMESTAMPS,
        video_backend="pyav",
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # Remove sequence length dimension for ACT inputs
            batch["observation.state"] = batch["observation.state"][:, -1, :]
            batch["observation.environment_state"] = batch["observation.environment_state"][:, -1, :]

            # 画像テンソルのシーケンス次元 (B, 1, C, H, W) -> (B, C, H, W) を整形し、RGBA(4ch)ならRGB(3ch)へスライス
            if "observation.images.cam_external" in batch:
                img = batch["observation.images.cam_external"][:, -1, :]
                if img.shape[1] == 4:
                    img = img[:, :3, :, :]
                batch["observation.images.cam_external"] = img

            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if step % LOG_FREQ == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
                wandb.log({"loss": loss.item(), "step": step})
            step += 1
            if step >= TRAINING_STEPS:
                done = True
                break

    policy.save_pretrained(OUTPUT_DIRECTORY)
    print(f"policy saved to {OUTPUT_DIRECTORY}")
    wandb.finish()

if __name__ == "__main__":
    main()