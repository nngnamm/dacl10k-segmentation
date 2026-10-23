import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, seed_everything
from src.dataset import DACL10kDataset, DACL10K_CLASSES, get_transforms
from src.models import build_model


@torch.no_grad()
def sliding_window_inference(image: torch.Tensor, model: torch.nn.Module, window_size: int, stride: int, num_classes: int) -> torch.Tensor:
    """Thực hiện suy luận bằng cửa sổ trượt (sliding window) cho ảnh kích thước lớn."""
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
                logits = model(patch)
                probs = torch.sigmoid(logits)
            prob_map[:, :, h:h+window_size, w:w+window_size] += probs
            count_map[:, :, h:h+window_size, w:w+window_size] += 1.0

    return (prob_map / count_map)[:, :, :H, :W]


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on dacl10k validation set")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Đường dẫn tới file checkpoint .pt/.pth")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]

    # 1. Nạp tập dữ liệu Validation
    val_ds = DACL10kDataset(
        root_dir=cfg["data"]["data_dir"],
        split="val",
        transforms=get_transforms(img_size=cfg["data"]["img_size"], is_train=False)
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

    # 2. Khởi tạo model và nạp checkpoint trọng số
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Xử lý linh hoạt trường hợp checkpoint lưu từ DataParallel hoặc State Dict trực tiếp
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # Khởi tạo các biến tích lũy thống kê cho từng class
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)

    window_size = cfg.get("eval", {}).get("window_size", cfg["data"]["img_size"])
    stride = cfg.get("eval", {}).get("stride", window_size // 2)
    threshold = cfg.get("eval", {}).get("threshold", 0.5)

    # 3. Vòng lặp đánh giá toàn bộ tập validation
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            targets = masks[0].cpu().numpy() # Lấy sample đầu tiên trong batch size = 1 -> [C, H, W]

            # Suy luận với sliding window để giữ nguyên độ phân giải gốc của ảnh
            probs = sliding_window_inference(
                images, model,
                window_size=window_size,
                stride=stride,
                num_classes=num_classes
            )
            
            preds = (probs.squeeze(0).cpu().numpy() >= threshold).astype(np.uint8)
            targets = (targets >= 0.5).astype(np.uint8)

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

    # 4. Tính toán các chỉ số IoU, Dice, Precision, Recall theo chuẩn Spec
    iou = inter / (union + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    # 5. In báo cáo kết quả chi tiết
    print("\n" + "=" * 70)
    print(f"{'Class Name':<18} | {'IoU':<8} | {'Dice':<8} | {'Prec':<8} | {'Recall':<8}")
    print("-" * 70)
    for c, cls_name in enumerate(DACL10K_CLASSES):
        print(f"{cls_name:<18} | {iou[c]*100:6.2f}% | {dice[c]*100:6.2f}% | {precision[c]*100:6.2f}% | {recall[c]*100:6.2f}%")
    print("-" * 70)
    print(f"Mean IoU (mIoU)  : {np.mean(iou)*100:.2f}%")
    print(f"Mean Dice (mDice): {np.mean(dice)*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()