"""Fine-tune an EfficientNet U-Net on DACL10K.

Single GPU:  python scripts/train_efficientnet.py --model efficientnet-b0
Multi GPU :  accelerate launch --multi_gpu --num_processes 2 \
                 scripts/train_efficientnet.py --model efficientnet-b4
"""

import argparse
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from dacl10k.data.dataset import DACL10kDataset
from dacl10k.engine.trainer import Trainer
from dacl10k.losses.bce_dice_loss import BCEDiceLoss
from dacl10k.models.efficientnet_unet import build_efficientnet_unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_base.yaml")
    parser.add_argument("--model", default=None, help="efficientnet-b0|b3|b4|b5")
    parser.add_argument("--batch-size", type=int, default=None, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None, help="path to last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # Command-line overrides.
    for key, value in (
        ("model", args.model),
        ("batch_size", args.batch_size),
        ("grad_accum", args.grad_accum),
        ("epochs", args.epochs),
    ):
        if value is not None:
            cfg[key] = value
    cfg["checkpoint_dir"] = str(Path(cfg["checkpoint_root"]) / cfg["model"])

    mixed_precision = cfg["mixed_precision"] if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=int(cfg["grad_accum"]),
    )
    set_seed(int(cfg["seed"]))

    size = (cfg["image_size"], cfg["image_size"])
    train_ds = DACL10kDataset(cfg["data_root"], "train", image_size=size)
    val_ds = DACL10kDataset(cfg["data_root"], "validation", image_size=size)

    workers = int(cfg["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )

    # Only the main process downloads pretrained weights; the others reuse the cache.
    with accelerator.main_process_first():
        model = build_efficientnet_unet(cfg["model"], cfg["num_classes"], pretrained=True)

    trainer = Trainer(
        cfg, model, train_loader, val_loader,
        criterion=BCEDiceLoss(), accelerator=accelerator,
    )
    result = trainer.fit(resume_path=args.resume)
    accelerator.print("Final:", result)


if __name__ == "__main__":
    main()