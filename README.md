# dacl10k-segmentation

**Enterprise-Grade Multi-Label Semantic Segmentation for Reinforced Concrete Bridge Inspection (`dacl10k`)**

[![CI](https://github.com/dacl10k/dacl10k-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/dacl10k/dacl10k-segmentation/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 1. Executive Summary & Domain Architecture

Civil infrastructure inspection of reinforced concrete bridges requires identifying both structural defects and key bridge components simultaneously. Unlike standard multi-class semantic segmentation datasets (e.g., Cityscapes, ADE20K) where each pixel belongs to a mutually exclusive category, **`dacl10k` (Damage Classification 10k)** is fundamentally a **multi-label semantic segmentation** problem across **19 canonical classes**:

| Category Group | Canonical Classes (`DACL10K_CLASSES`) |
| :--- | :--- |
| **Surface & Structural Defects (11)** | `Crack`, `ACrack` (Alligator Crack), `Wetspot`, `Efflorescence`, `Rust`, `Rockpocket`, `Hollowareas`, `Cavity`, `Spalling`, `Restformwork`, `ExposedRebars` |
| **Bridge Objects & Context (8)** | `Bearing`, `Drainage`, `ExpansionJoint`, `Graffiti`, `Weathering`, `Joint`, `PEquipment` (Protective Equipment), `WFlange` |

### Why Multi-Label (`[B, 19, H, W]`) and Sigmoid over Softmax?

In real-world bridge inspections, damage patterns physically overlap and co-occur at the exact same pixel coordinates:
- **`Spalling` + `ExposedRebars` + `Rust`**: Spalled concrete exposes oxidized reinforcing steel bars inside the spalled region.
- **`Wetspot` + `Efflorescence` + `Crack`**: Water seepage through structural cracks deposits crystalline salt efflorescence across the wet surface.
- **`Graffiti` / `Weathering`** superimposed over **`Joint`** or **`Drainage`** components.

Applying a standard `Softmax` activation across channels forces classes to compete ($\sum_{c=1}^{19} p_c = 1$), suppressing co-occurring defects. Instead, this system encodes ground-truth annotations as dense binary tensors of shape **`[B, 19, H, W]`** (`float32` in `{0.0, 1.0}`) and models each channel as an independent Bernoulli distribution via raw logits and element-wise **Sigmoid** activation:

$$\hat{p}_{b, c, h, w} = \sigma(z_{b, c, h, w}) = \frac{1}{1 + \exp(-z_{b, c, h, w})}$$

---

## 2. System Architecture & Repository Structure

The repository follows a modular `src/` package layout engineered for reproducibility, static analysis (`ruff`), unit testing (`pytest`), distributed multi-GPU training (`accelerate`), and edge deployment (`onnx` / `onnxruntime`).

```text
dacl10k-segmentation/
├── .github/workflows/
│   └── ci.yml                        # Automated Ruff lint/format + Pytest CI pipeline
├── configs/
│   └── default_config.yaml           # Declarative YAML training & architecture config
├── scripts/
│   ├── train.py                      # DDP-ready CLI training entrypoint (--config, --resume, --dummy)
│   └── export_and_benchmark.py       # ONNX export, CPU latency benchmark & failure mode CLI
├── src/dacl10k/
│   ├── data/
│   │   ├── constants.py              # Canonical 19-class ordering & index mappings
│   │   ├── rasterizer.py             # Vector polygon & interior hole rasterization (cv2.fillPoly)
│   │   ├── transforms.py             # Spatially-safe Albumentations pipelines
│   │   └── dataset.py                # PyTorch DACL10kDataset returning [3,H,W] & [19,H,W] tensors
│   ├── models/
│   │   └── factory.py                # Config-driven SMP factory (SegFormer, U-Net++, DeepLabV3+)
│   ├── losses/
│   │   ├── asymmetric_loss.py        # Asymmetric Loss (ASL) for extreme foreground/background imbalance
│   │   ├── dice_loss.py              # Channel-wise Soft Dice Loss for multi-label masks
│   │   └── factory.py                # Loss builder (BCE, BCEDiceLoss, AsymmetricLoss)
│   ├── engine/
│   │   ├── trainer.py                # Hugging Face Accelerate DDP trainer & fault-tolerant checkpointer
│   │   └── export.py                 # Dynamic-axis ONNX exporter & numerical parity verifier
│   └── utils/
│       ├── metrics.py                # Distributed-safe MultiLabelIoUMetric accumulator
│       ├── thresholds.py             # Per-class F1-optimal probability threshold tuner
│       ├── benchmark.py              # PyTorch vs. ONNX Runtime latency (Mean/P95/FPS) profiler
│       └── error_analysis.py         # Per-class TP/FP/FN/FNR/FPR failure mode analyzer
├── tests/
│   ├── test_sanity.py
│   ├── test_dataset.py
│   ├── test_models_and_trainer.py
│   ├── test_losses_and_thresholds.py
│   └── test_export_and_analysis.py
└── pyproject.toml                    # PEP 621 packaging, dependencies, Ruff & Pytest config
```

---

## 3. High-Performance Data Pipeline

### Exterior Polygon & Interior Hole Rasterization (`src/dacl10k/data/rasterizer.py`)
Raw `dacl10k` annotations are stored in Supervisely-style JSON files containing complex polygon geometries. Rather than pre-generating terabytes of uncompressed binary masks on disk, `rasterize_annotations` rasterizes polygons on the fly in C++ via OpenCV:
1. Allocates a contiguous `np.zeros((H, W, 19), dtype=np.uint8)` canvas.
2. Draws **exterior polygons** onto channel $c$ with value `1` using `cv2.fillPoly`.
3. Carves out **interior holes** (`bitmap` / `interior` ring coordinates) on the same channel with value `0` using a subsequent `cv2.fillPoly` pass, preserving doughnut-shaped spalling rings and hollow areas without corrupting overlapping classes on other channels.

### Spatially-Safe Augmentations (`src/dacl10k/data/transforms.py`)
Bridge images vary wildly in aspect ratio and resolution. During training (`split="train"`), applying `RandomCrop(512, 512)` directly to smaller images raises spatial dimension errors. Our pipeline guarantees safety:
- **`A.PadIfNeeded(min_height=H, min_width=W, border_mode=0)`** precedes `A.RandomCrop(height=H, width=W, p=1.0)`.
- Applies `HorizontalFlip`, `VerticalFlip`, `RandomBrightnessContrast`, ImageNet normalization, and `ToTensorV2()`, permuting masks cleanly into `[19, H, W]` `torch.float32` tensors.

---

## 4. Model Factory & Multi-Label Loss Engineering

### Hierarchical Transformer vs. CNN Architectures (`src/dacl10k/models/factory.py`)
`build_model(cfg)` wraps `segmentation_models_pytorch` (`smp`) with `in_channels=3`, `classes=19`, and `activation=None` (raw logits for mixed-precision numerical stability):
- **SegFormer (`arch="segformer"`, `encoder_name="mit_b2"`)**: Mix Vision Transformer backbone capturing long-range structural context (e.g., distinguishing continuous `Joint` or `Drainage` lines from irregular `Crack` propagation) paired with a lightweight all-MLP decoder.
- **U-Net++ (`arch="unetplusplus"`, `encoder_name="resnet34"`)**: Nested dense skip pathways preserving high-frequency spatial boundaries for thin hairline cracks (`Crack`, `ACrack`) and fine rebar meshes (`ExposedRebars`).

### Multi-Label Loss Formulation (`src/dacl10k/losses/`)
Thin defects like cracks occupy $<0.5\%$ of image pixels, creating severe foreground-background imbalance. We provide three configurable loss regimens via `build_loss(cfg)`:

1. **Binary Cross-Entropy with Logits (`loss: "bce"`)**:
   Standard pixel-wise cross-entropy utilizing log-sum-exp trick via `nn.BCEWithLogitsLoss()`.

2. **Combined BCE + Multi-Label Soft Dice (`loss: "bce_dice"`)**:
   Computes soft Dice per channel across batch and spatial dimensions $(B, H, W)$ before averaging across the 19 classes:
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{b,h,w} p_{b,c,h,w} y_{b,c,h,w} + \text{smooth}}{\sum_{b,h,w} p_{b,c,h,w} + \sum_{b,h,w} y_{b,c,h,w} + \text{smooth} + \epsilon}$$

3. **Asymmetric Loss (`loss: "asymmetric"` / `"asl"`)**:
   Decouples positive and negative focusing parameters ($\gamma_+ = 0.0, \gamma_- = 4.0$) and applies hard thresholding via probability shifting ($p_m = \max(p - m, 0)$ with clip margin $m = 0.05$) to down-weight easy background negatives without attenuating gradients from rare positive defect pixels:
   $$\mathcal{L}_{\text{ASL}} = -y (1 - p)^{\gamma_+} \log(\max(p, \epsilon)) - (1 - y) p_m^{\gamma_-} \log(\max(1 - p_m, \epsilon))$$

### Per-Class Threshold Tuning (`src/dacl10k/utils/thresholds.py`)
Using a uniform decision threshold of $\tau = 0.5$ across all 19 classes degrades performance on rare, low-confidence classes. `optimize_per_class_thresholds` performs a vectorized grid search over candidate thresholds $\tau \in [0.1, 0.9]$ on validation logits to maximize the per-class F1 score independently for each of the 19 classes.

---

## 5. Distributed Training on Kaggle 2x T4 GPUs

The `Trainer` (`src/dacl10k/engine/trainer.py`) uses **Hugging Face Accelerate** for zero-boilerplate Distributed Data Parallel (DDP) training, `fp16` mixed precision, synchronized validation metric gathering (`accelerator.gather_for_metrics`), and main-process checkpointing.

### Quick Smoke Test (CPU / Single GPU)
```bash
python scripts/train.py --config configs/default_config.yaml --dummy
```

### Multi-GPU DDP Launch (Kaggle 2x T4 GPUs)
```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    scripts/train.py \
    --config configs/default_config.yaml
```

### Fault-Tolerant Preemption Recovery (`--resume`)
Cloud GPU sessions (e.g., Kaggle 12-hour quota limits or spot instances) can be preempted. Every epoch saves `checkpoints/last.pt` and `checkpoints/best.pt` containing unwrapped `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, and `best_miou`. Resume seamlessly from the exact interrupted epoch:

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    scripts/train.py \
    --config configs/default_config.yaml \
    --resume checkpoints/last.pt
```

---

## 6. Edge Deployment, Benchmarking & Failure Mode Analysis

Field inspection drones and mobile bridge inspection units require deterministic, low-latency inference without heavy PyTorch training dependencies.

### Dynamic-Axis ONNX Export & Verification (`src/dacl10k/engine/export.py`)
`export_to_onnx` exports any trained model to ONNX (`opset_version=17`) with dynamic batch and spatial axes (`{0: "batch_size", 2: "height", 3: "width"}`), validates the graph via `onnx.checker.check_model`, and verifies numerical parity against `onnxruntime` (`max_abs_diff < 1e-3`).

### End-to-End Export, Latency Benchmark & Error Analysis CLI
```bash
python scripts/export_and_benchmark.py \
    --config configs/default_config.yaml \
    --checkpoint checkpoints/best.pt \
    --output outputs/model.onnx \
    --image-size 512 512
```

This command outputs:
1. **ONNX Parity & Model Size Report**: Verifies `< 1e-3` maximum absolute deviation between PyTorch and ONNX Runtime outputs.
2. **Latency & Throughput Comparison Table (`benchmark_pytorch_vs_onnx`)**: Compares Mean Latency (ms), P95 Tail Latency (ms), Throughput (FPS), and Speedup Ratio across warmup and timed runs.
3. **Failure Mode Diagnostic Report (`analyze_failure_modes`)**: Evaluates all 19 classes sorted by ascending IoU, exposing False Negative Rates (`FNR`) on thin structural defects (`Crack`, `ExposedRebars`) and False Positive Rates (`FPR`) on visually ambiguous surface textures (`Wetspot`, `Weathering`).

---

## 7. Quality Assurance & Testing

Run the full linter, formatter, and unit test suite locally:

```bash
ruff check .
ruff format --check .
pytest -v
```
