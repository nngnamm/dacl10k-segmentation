"""Visualize DACL10K ground truth and model predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from dacl10k.data.dataset import DACL10kDataset
from dacl10k.data.transforms import get_transforms
from dacl10k.models.factory import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path("/Users/Quyet/Documents/groupProject/dacl10k_v2_devphase")

CHECKPOINT_PATH = Path(
    "~/Documents/groupProject/dacl10k_benchmark_trial/checkpoints/best.pt"
).expanduser()

OUTPUT_PATH = PROJECT_ROOT / "reports" / "dacl10k_predictions.png"

NUM_IMAGES = 8
BATCH_SIZE = 2
SEED = 43
THRESHOLD = 0.5


def main():
    # Select the available device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Read the project configuration.
    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    image_size_cfg = config.get("image_size", [512, 512])
    image_size = (
        int(image_size_cfg[0]),
        int(image_size_cfg[1]),
    )

    # Load the validation dataset.
    dataset = DACL10kDataset(
        root_dir=DATA_DIR,
        split="val",
        transforms=get_transforms(
            "val",
            image_size=image_size,
        ),
    )

    if len(dataset) < NUM_IMAGES:
        raise ValueError("Not enough validation images.")

    # Reproduce the validation subset used in the diagnostic script.
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:NUM_IMAGES].tolist()

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Build the model and load its saved weights.
    model = build_model(config).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation indices: {indices}")

    # ImageNet normalization parameters used by the preprocessing.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].float()

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= THRESHOLD).float()

            for i in range(images.shape[0]):
                # Convert normalized image back to display range.
                image = images[i].cpu().permute(1, 2, 0).numpy()
                image = image * std + mean
                image = np.clip(image, 0, 1)

                target = targets[i].cpu().numpy()
                prediction = predictions[i].cpu().numpy()

                # Any positive class is displayed as a foreground pixel.
                target_any = target.max(axis=0) > 0.5
                prediction_any = prediction.max(axis=0) > 0.5

                sample_id = batch["id"][i]

                results.append(
                    (
                        sample_id,
                        image,
                        target_any,
                        prediction_any,
                    )
                )

    # Create one row per image.
    fig, axes = plt.subplots(
        len(results),
        3,
        figsize=(15, 4 * len(results)),
        squeeze=False,
    )

    for row, (sample_id, image, target, prediction) in enumerate(results):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{sample_id}\nOriginal image")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(
            np.ma.masked_where(~target, target),
            cmap="Greens",
            alpha=0.65,
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title("Ground truth overlay")

        axes[row, 2].imshow(image)
        axes[row, 2].imshow(
            np.ma.masked_where(~prediction, prediction),
            cmap="Reds",
            alpha=0.65,
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Prediction overlay")

        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(
        "DACL10K: Original vs Ground Truth vs Prediction",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved visualization to: {OUTPUT_PATH}")
    print("No training was performed.")


if __name__ == "__main__":
    main()
