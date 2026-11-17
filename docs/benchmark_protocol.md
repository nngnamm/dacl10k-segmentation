# DACL10K Benchmark Protocol

## 1. Objective

The current experiment verifies the training pipeline and establishes a preliminary baseline for DeepLabV3+ with an EfficientNet-B4 encoder.

A future controlled benchmark may compare EfficientNet-B4 and EfficientNet-B5 under consistent experimental conditions.

## 2. Dataset

- Dataset: DACL10K.
- Available training images: 6,935.
- Available validation images: 975.
- Preliminary trial: 16 training images and 8 validation images.
- The preliminary trial must not be reported as a full-dataset benchmark.

## 3. Model and Input

- Architecture: DeepLabV3+.
- Encoder: EfficientNet-B4.
- Encoder weights: ImageNet pretrained.
- Input size: 512 × 512.
- Input channels: 3.
- Output classes: 19.
- Batch size: 2.

## 4. Training Configuration

- Optimizer: AdamW.
- Learning rate: 0.0001.
- Weight decay: 0.01.
- Loss function: BCEWithLogitsLoss.
- Configured training duration: 30 epochs.
- Completed preliminary trial: 1 epoch.
- Random seed: 42.
- Device: Apple Silicon MPS.
- Operating system: macOS.
- System memory: 16 GB.

Record the actual configuration used for each experiment. Do not assume that every standalone test script uses all settings in the default configuration.

## 5. Evaluation

- Record training loss and validation loss.
- Record validation mIoU and the metric implementation used.
- Prediction threshold: 0.5 in the current custom metric.
- Verify the required DACL10K evaluation protocol before making official benchmark comparisons.
- Investigate class-wise prediction behavior and image-mask alignment when results appear unreliable.

## 6. Resource Reporting

For each experiment, record:
- Model architecture and encoder.
- Dataset split and number of samples.
- Input size and batch size.
- Number of completed epochs.
- Training and validation metrics.
- Device and elapsed time.
- Checkpoint status.
- Known limitations.

Record memory usage only when it has actually been measured.

## 7. Limitations and Fair Comparison

The current results come from a one-epoch trial on a small subset. They do not establish final model quality or full-training runtime.

For a fair B4/B5 comparison, use the same dataset split, input size, evaluation protocol, and comparable training settings. Document hardware constraints and any deviations from the protocol.