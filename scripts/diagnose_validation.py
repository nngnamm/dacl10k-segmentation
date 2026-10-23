"""Diagnose per-class segmentation predictions from a saved checkpoint."""

from pathlib import Path

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

NUM_VAL_SAMPLES = 8
BATCH_SIZE = 2
SEED = 43
THRESHOLD = 0.5


def main():
    # 1. Select the available device.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 65)
    print("DACL10K VALIDATION DIAGNOSTICS")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATA_DIR}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {CHECKPOINT_PATH}")

    # 2. Read the project configuration.
    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    image_size_cfg = config.get("image_size", [512, 512])
    image_size = (
        int(image_size_cfg[0]),
        int(image_size_cfg[1]),
    )

    num_classes = int(config.get("num_classes", 19))

    # 3. Recreate the validation dataset.
    dataset = DACL10kDataset(
        root_dir=DATA_DIR,
        split="val",
        transforms=get_transforms(
            "val",
            image_size=image_size,
        ),
    )

    if len(dataset) < NUM_VAL_SAMPLES:
        raise ValueError(
            f"Expected at least {NUM_VAL_SAMPLES} validation images, but found {len(dataset)}."
        )

    # Reproduce the random subset used in the fine-tuning trial.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:NUM_VAL_SAMPLES].tolist()

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Full validation dataset: {len(dataset)} images")
    print(f"Diagnostic subset: {len(subset)} images")
    print(f"Validation indices: {indices}")

    # 4. Recreate the model and load saved weights.
    model = build_model(config).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(
        "Checkpoint best_miou:",
        checkpoint.get("best_miou", "unknown"),
    )

    # 5. Accumulate per-class statistics.
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    target_positive = torch.zeros(num_classes, dtype=torch.float64)
    predicted_positive = torch.zeros(num_classes, dtype=torch.float64)
    probability_sum = torch.zeros(num_classes, dtype=torch.float64)
    logit_sum = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device).float()

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            probability_sum += probabilities.sum(dim=(0, 2, 3)).cpu().double()

            logit_sum += logits.sum(dim=(0, 2, 3)).cpu().double()

            predictions = (probabilities >= THRESHOLD).float()

            targets_binary = (targets >= 0.5).float()

            dims = (0, 2, 3)

            batch_intersection = (predictions * targets_binary).sum(dim=dims).cpu().double()

            batch_union = (
                (predictions + targets_binary - predictions * targets_binary)
                .sum(dim=dims)
                .cpu()
                .double()
            )

            batch_target_positive = targets_binary.sum(dim=dims).cpu().double()

            batch_predicted_positive = predictions.sum(dim=dims).cpu().double()

            intersection += batch_intersection
            union += batch_union
            target_positive += batch_target_positive
            predicted_positive += batch_predicted_positive

    # 6. Compute IoU using the current metric's epsilon convention.
    eps = 1e-7
    per_class_iou = (intersection + eps) / (union + eps)

    # Compare with the convention that excludes classes with zero union.
    valid_classes = union > 0

    current_style_miou = per_class_iou.mean().item()

    if valid_classes.any():
        valid_union_miou = per_class_iou[valid_classes].mean().item()
    else:
        valid_union_miou = 0.0

    print("\nPER-CLASS RESULTS")
    print("-" * 100)
    print(
        f"{'Class':>7} "
        f"{'GT positive':>15} "
        f"{'Pred positive':>15} "
        f"{'Intersection':>15} "
        f"{'Union':>15} "
        f"{'IoU':>12}"
    )
    print("-" * 100)

    for class_idx in range(num_classes):
        print(
            f"{class_idx:>7} "
            f"{int(target_positive[class_idx].item()):>15} "
            f"{int(predicted_positive[class_idx].item()):>15} "
            f"{int(intersection[class_idx].item()):>15} "
            f"{int(union[class_idx].item()):>15} "
            f"{per_class_iou[class_idx].item():>12.6f}"
        )

    print("\nSUMMARY")
    print("-" * 40)
    print(f"Classes: {num_classes}")
    print(f"Classes with non-zero union: {int(valid_classes.sum())}")
    print(f"Classes with zero union: {int((~valid_classes).sum())}")
    print(f"Current-style mIoU: {current_style_miou:.6f}")
    print(f"Non-zero-union mIoU: {valid_union_miou:.6f}")
    print(
        "Classes with ground-truth positives:",
        int((target_positive > 0).sum()),
    )
    print(
        "Classes with predicted positives:",
        int((predicted_positive > 0).sum()),
    )
    print("\nPREDICTION DENSITY BY CLASS")
    print("-" * 60)
    print(f"{'Class':>7} {'GT positive %':>18} {'Pred positive %':>20}")
    print("-" * 60)

    total_pixels = NUM_VAL_SAMPLES * image_size[0] * image_size[1]

    for class_idx in range(num_classes):
        gt_pct = target_positive[class_idx].item() / total_pixels * 100
        pred_pct = predicted_positive[class_idx].item() / total_pixels * 100

        print(f"{class_idx:>7} {gt_pct:>17.4f}% {pred_pct:>19.4f}%")
    print("\nDiagnostic completed. No training was performed.")

    total_pixels = NUM_VAL_SAMPLES * image_size[0] * image_size[1]

    mean_probability = probability_sum / total_pixels
    mean_logit = logit_sum / total_pixels

    print("\nPROBABILITY DIAGNOSTICS")
    print("-" * 65)
    print(f"{'Class':>7} {'Mean probability':>20} {'Mean logit':>15}")
    print("-" * 65)

    for class_idx in range(num_classes):
        print(
            f"{class_idx:>7} "
            f"{mean_probability[class_idx].item():>20.6f} "
            f"{mean_logit[class_idx].item():>15.6f}"
        )

    print("\nDiagnostic completed. No training was performed.")


if __name__ == "__main__":
    main()
