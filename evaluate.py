import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from datasets import load_dataset
from tqdm import tqdm

from src.utils import load_config
from src.dataset import DACL10kDataset, DACL10K_CLASSES, get_transforms
from src.models import build_model

@torch.no_grad()
def sliding_window_inference(image, model, window_size, stride, num_classes):
    _, _, H, W = image.shape
    pad_h = max(0, window_size - H)
    pad_w = max(0, window_size - W)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h))
    _, _, H_pad, W_pad = image.shape

    prob_map = torch.zeros((1, num_classes, H_pad, W_pad), device=image.device)
    count_map = torch.zeros((1, 1, H_pad, W_pad), device=image.device)

    h_steps = list(range(0, H_pad - window_size + 1, stride))
    if (H_pad - window_size) % stride != 0:
        h_steps.append(H_pad - window_size)
    w_steps = list(range(0, W_pad - window_size + 1, stride))
    if (W_pad - window_size) % stride != 0:
        w_steps.append(W_pad - window_size)

    for h in h_steps:
        for w in w_steps:
            patch = image[:, :, h:h+window_size, w:w+window_size]
            with autocast():
                probs = torch.sigmoid(model(patch))
            prob_map[:, :, h:h+window_size, w:w+window_size] += probs
            count_map[:, :, h:h+window_size, w:w+window_size] += 1.0

    return (prob_map / count_map)[:, :, :H, :W]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]

    # Tải tập validation (Full-resolution image, không crop)
    hf_data = load_dataset(cfg["data"]["hf_repo"])
    val_split = "validation" if "validation" in hf_data else "val"
    val_ds = DACL10kDataset(hf_data[val_split], transforms=get_transforms(is_train=False))

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location=device)
    # Xử lý trường hợp lưu bằng DataParallel (có tiền tố 'module.')
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    for i in tqdm(range(len(val_ds)), desc="Evaluating Full Images"):
        image, mask = val_ds[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.numpy() # (19, H, W)

        probs = sliding_window_inference(
            image, model,
            window_size=cfg["eval"]["window_size"],
            stride=cfg["eval"]["stride"],
            num_classes=num_classes
        )
        preds = (probs.squeeze(0).cpu().numpy() >= cfg["eval"]["threshold"]).astype(np.uint8)
        targets = mask.astype(np.uint8)

        for c in range(num_classes):
            p, t = preds[c], targets[c]
            tp_c = np.logical_and(p == 1, t == 1).sum()
            fp_c = np.logical_and(p == 1, t == 0).sum()
            fn_c = np.logical_and(p == 0, t == 1).sum()

            tp[c] += tp_c
            fp[c] += fp_c
            fn[c] += fn_c
            inter[c] += tp_c
            union[c] += (tp_c + fp_c + fn_c)

    # Tính Per-class IoU, Dice, Precision, Recall theo đúng Mục 7 của Spec
    iou = inter / (union + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    print("\n" + "="*65)
    print(f"{'Class Name':<18} | {'IoU':<8} | {'Dice':<8} | {'Prec':<8} | {'Recall':<8}")
    print("-" * 65)
    for c, cls_name in enumerate(DACL10K_CLASSES):
        print(f"{cls_name:<18} | {iou[c]*100:6.2f}% | {dice[c]*100:6.2f}% | {precision[c]*100:6.2f}% | {recall[c]*100:6.2f}%")
    print("-" * 65)
    print(f"Mean IoU (mIoU)  : {np.mean(iou)*100:.2f}%")
    print(f"Mean Dice (mDice): {np.mean(dice)*100:.2f}%")
    print("="*65)

if __name__ == "__main__":
    main()