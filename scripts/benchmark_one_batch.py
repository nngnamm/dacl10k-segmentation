import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dacl10k.data.dataset import DACL10kDataset  # noqa: E402
from dacl10k.data.transforms import get_transforms  # noqa: E402
from dacl10k.models.factory import build_model  # noqa: E402


def main():
    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------
    config_path = PROJECT_ROOT / "configs" / "default_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_dir = Path("/Users/Quyet/Documents/groupProject/dacl10k_v2_devphase")

    print("=" * 60)
    print("DACL10K ONE-BATCH SMOKE TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Select the available device
    # --------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # --------------------------------------------------
    # 3. Load one real batch
    # --------------------------------------------------
    dataset = DACL10kDataset(
        root_dir=data_dir,
        split="train",
        transforms=get_transforms(
            "train",
            image_size=(512, 512),
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=2,  # BatchNorm needs more than one value per channel
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(loader))

    images = batch["image"].to(device)
    masks = batch["mask"].to(device, dtype=torch.float32)

    print("Sample ID:", batch["id"])
    print("Input shape:", tuple(images.shape))
    print("Target shape:", tuple(masks.shape))
    print("Input dtype:", images.dtype)
    print("Target dtype:", masks.dtype)

    # --------------------------------------------------
    # 4. Build the real segmentation model
    # --------------------------------------------------
    model = build_model(config)
    model = model.to(device)
    model.train()

    print("Architecture:", config.get("arch"))
    print("Encoder:", config.get("encoder_name"))
    print("Encoder weights:", config.get("encoder_weights"))

    # --------------------------------------------------
    # 5. Forward pass
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)

    print("Output shape:", tuple(logits.shape))

    expected_shape = tuple(masks.shape)

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Output shape {tuple(logits.shape)} does not match target shape {expected_shape}"
        )

    # --------------------------------------------------
    # 6. Calculate loss
    # --------------------------------------------------
    loss = criterion(logits, masks)

    print("Loss:", loss.item())

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss is NaN or infinity.")

    # --------------------------------------------------
    # 7. Backward pass
    # --------------------------------------------------
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]

    if not gradients:
        raise RuntimeError("No gradients were generated.")

    all_gradients_finite = all(torch.isfinite(gradient).all().item() for gradient in gradients)

    if not all_gradients_finite:
        raise RuntimeError("Some gradients contain NaN or infinity.")

    print("Parameters with gradients:", len(gradients))
    print("All gradients finite:", all_gradients_finite)

    # --------------------------------------------------
    # 8. Perform one optimizer update
    # --------------------------------------------------
    first_parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)

    parameter_before = first_parameter.detach().clone()

    optimizer.step()

    parameter_change = (first_parameter.detach() - parameter_before).abs().max().item()

    print("Maximum parameter change:", parameter_change)

    if parameter_change == 0:
        raise RuntimeError(
            "The checked parameter did not change. Inspect the optimizer and gradients."
        )

    print("\nSUCCESS: One-batch smoke test passed.")
    print("No checkpoint was saved.")


if __name__ == "__main__":
    main()
