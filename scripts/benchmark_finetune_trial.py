"""Run a small real-data DACL10K fine-tuning trial."""

import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dacl10k.data.dataset import DACL10kDataset
from dacl10k.data.transforms import get_transforms
from dacl10k.engine.trainer import Trainer
from dacl10k.models.factory import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Actual DACL10K dataset location.
DATA_DIR = Path("/Users/Quyet/Documents/groupProject/dacl10k_v2_devphase")

# Keep checkpoints outside the Git repository.
OUTPUT_DIR = Path.home() / "Documents" / "groupProject" / "dacl10k_benchmark_trial" / "checkpoints"

NUM_TRAIN_SAMPLES = 16
NUM_VAL_SAMPLES = 8
BATCH_SIZE = 2
SEED = 42


def select_random_subset(dataset, num_samples, seed):
    """Select a reproducible random subset from a dataset."""
    if len(dataset) < num_samples:
        raise ValueError(
            f"Dataset has only {len(dataset)} samples, but {num_samples} were requested."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:num_samples].tolist()

    return Subset(dataset, indices)


def main():
    start_time = time.perf_counter()

    random.seed(SEED)
    torch.manual_seed(SEED)

    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Override settings for this trial only.
    config["data_dir"] = str(DATA_DIR)
    config["batch_size"] = BATCH_SIZE
    config["num_workers"] = 0
    config["epochs"] = 1
    config["mixed_precision"] = "no"
    config["use_wandb"] = False
    config["checkpoint_dir"] = str(OUTPUT_DIR)

    image_size_cfg = config.get("image_size", [512, 512])
    image_size = (
        int(image_size_cfg[0]),
        int(image_size_cfg[1]),
    )

    print("=" * 60)
    print("DACL10K REAL-DATA FINE-TUNING TRIAL")
    print("=" * 60)
    print(f"Dataset: {DATA_DIR}")
    print(f"Train samples: {NUM_TRAIN_SAMPLES}")
    print(f"Validation samples: {NUM_VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {image_size}")
    print(f"Architecture: {config['arch']}")
    print(f"Encoder: {config['encoder_name']}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATA_DIR}")

    # Build the real datasets.
    train_dataset = DACL10kDataset(
        root_dir=DATA_DIR,
        split="train",
        transforms=get_transforms(
            "train",
            image_size=image_size,
        ),
    )

    val_dataset = DACL10kDataset(
        root_dir=DATA_DIR,
        split="val",
        transforms=get_transforms(
            "val",
            image_size=image_size,
        ),
    )

    print(f"Full training dataset: {len(train_dataset)} images")
    print(f"Full validation dataset: {len(val_dataset)} images")

    # Select small, reproducible subsets.
    train_subset = select_random_subset(
        train_dataset,
        NUM_TRAIN_SAMPLES,
        SEED,
    )

    val_subset = select_random_subset(
        val_dataset,
        NUM_VAL_SAMPLES,
        SEED + 1,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Initialize the pretrained segmentation model.
    model = build_model(config)

    # Trainer handles device placement, optimization, validation,
    # metric calculation, and checkpoint saving.
    trainer = Trainer(
        cfg=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print(f"Training device: {trainer.accelerator.device}")
    print("Starting one-epoch trial...")

    metrics = trainer.fit()

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("TRIAL FINISHED")
    print("=" * 60)

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Checkpoint directory: {OUTPUT_DIR}")
    print(f"Last checkpoint exists: {(OUTPUT_DIR / 'last.pt').exists()}")
    print(f"Best checkpoint exists: {(OUTPUT_DIR / 'best.pt').exists()}")


if __name__ == "__main__":
    main()
