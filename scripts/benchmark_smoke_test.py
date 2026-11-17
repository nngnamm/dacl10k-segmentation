import time

import segmentation_models_pytorch as smp
import torch

NUM_CLASSES = 19
IMAGE_SIZE = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Kiểm tra các lựa chọn trước khi tạo model.
print("PyTorch:", torch.__version__)
print("SMP:", smp.__version__)
print("Device:", DEVICE)

if not hasattr(smp, "DeepLabV3"):
    raise RuntimeError("DeepLabV3 is unavailable in this SMP installation.")

encoder_names = smp.encoders.get_encoder_names()
candidates = [name for name in encoder_names if "efficientnet" in name.lower()]
print("EfficientNet encoder candidates:", candidates)

# Chỉ dùng tên encoder đã xác nhận có trong danh sách.
encoder_name = next(
    (name for name in candidates if "b4" in name.lower()),
    None,
)

if encoder_name is None:
    raise RuntimeError("No supported EfficientNet-B4 encoder was found.")

model = smp.DeepLabV3(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None,
).to(DEVICE)

model.eval()
images = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

start = time.perf_counter()

with torch.no_grad():
    outputs = model(images)

elapsed = time.perf_counter() - start

print("Encoder:", encoder_name)
print("Input shape:", tuple(images.shape))
print("Output shape:", tuple(outputs.shape))
print("Forward time (seconds):", round(elapsed, 3))
print("Smoke test passed.")
