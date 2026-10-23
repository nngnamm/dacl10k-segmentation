import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils import load_config, seed_everything, save_checkpoint
from src.dataset import DACL10kDataset, get_transforms
from src.models import build_model
from src.losses import MultiLabelComboLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DACL10kDataset(cfg["data"]["data_dir"], split="train", transforms=get_transforms(cfg["data"]["img_size"], is_train=True))
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = build_model(cfg)
    model.to(device)

    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    backbone_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr}
    ], weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    criterion = MultiLabelComboLoss(cfg["train"]["focal_weight"], cfg["train"]["dice_weight"])
    scaler = GradScaler()
    grad_accum = cfg["train"]["grad_accum_steps"]

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for step, (images, masks) in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                loss = criterion(logits, masks) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        scheduler.step()
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": total_loss / len(train_loader),
            "config": cfg
        }, cfg["train"]["save_dir"], f"{cfg['exp_name']}_last.pt")

if __name__ == "__main__":
    main()