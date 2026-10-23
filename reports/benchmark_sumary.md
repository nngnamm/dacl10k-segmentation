# DACL10K Personal Benchmark Report

## 1. Objective

The objective of this experiment is to verify the training pipeline and evaluate the preliminary fine-tuning behavior of DeepLabV3+ with an EfficientNet-B4 encoder on the DACL10K bridge damage segmentation dataset.

Two experiments were conducted:
- A one-batch smoke test to verify forward propagation, loss computation, gradient calculation, and parameter updates.
- A one-epoch fine-tuning trial on a small subset of the real dataset to verify training, validation, metric calculation, and checkpoint saving.

These experiments are preliminary engineering tests, not a completed full-dataset benchmark.

## 2. Environment

| Parameter | Value |
|---|---|
| Operating system | macOS |
| Hardware | Apple Silicon Mac |
| System memory | 16 GB RAM |
| Python version | 3.13.13 |
| Virtual environment | `.venv` |
| Training device | Apple Metal Performance Shaders (MPS) |
| Deep learning framework | PyTorch |
| Model library | Segmentation Models PyTorch |
| Dataset | DACL10K |
| Input image size | 512 × 512 |
| Number of classes | 19 |

The pipeline uses MPS for training on Apple Silicon.

## 3. Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | DeepLabV3+ |
| Encoder | EfficientNet-B4 |
| Encoder weights | ImageNet pretrained |
| Input channels | 3 |
| Output classes | 19 |
| Input image size | 512 × 512 |
| Batch size | 2 |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| Loss function | BCEWithLogitsLoss |
| Evaluation metric | Custom multi-label mIoU |
| Prediction threshold | 0.5 |
| Device | MPS |

The model produces 19 output channels, corresponding to the 19 segmentation classes. The current custom metric applies sigmoid to the logits and uses a threshold of 0.5 to obtain binary predictions.

## 4. Smoke Test Results

The one-batch smoke test completed successfully using two training images.

| Item | Result |
|---|---|
| Input shape | (2, 3, 512, 512) |
| Target shape | (2, 19, 512, 512) |
| Output shape | (2, 19, 512, 512) |
| Device | MPS |
| Loss | 0.682810 |
| Parameters with gradients | 447 |
| All gradients finite | True |
| Maximum parameter change | 0.00010252 |
| Result | Passed |

The test confirms that the model can process a batch, calculate the loss, produce finite gradients, and update its parameters. No checkpoint was saved during this test.

This result verifies technical execution only; it does not measure generalization performance.

## 5. Fine-tuning Results

A preliminary one-epoch fine-tuning trial was completed using a small subset of the real DACL10K dataset.

| Item | Result |
|---|---:|
| Training samples used | 16 |
| Validation samples used | 8 |
| Full training dataset available | 6,935 images |
| Full validation dataset available | 975 images |
| Epochs completed | 1 |
| Batch size | 2 |
| Device | MPS |
| Training loss | 0.659939 |
| Validation loss | 0.676032 |
| Validation mIoU | 0.017336 |
| Elapsed time | 12.80 seconds |
| Last checkpoint | Saved |
| Best checkpoint | Saved |
| Result | Completed |

The trial confirms that training and validation can run on real data and that checkpoints can be saved.

The validation mIoU of 0.017336 is low. Because the experiment used only 16 training images, 8 validation images, and one epoch, it is insufficient to establish the final quality of the model. The result should be treated as a preliminary baseline for further investigation.

## 6. Limitations

1. **Limited training data:** Only 16 training images were used instead of the full training dataset.
2. **Limited validation data:** Only 8 validation images were evaluated instead of the 975 available validation images.
3. **Insufficient training duration:** Only one epoch was completed.
4. **Low preliminary mIoU:** The measured validation mIoU was 0.017336. Further diagnostics are needed before drawing conclusions about model quality.
5. **Prediction distribution:** Preliminary diagnostics showed substantial differences in prediction density across classes, including overprediction and underprediction. The causes have not yet been established.
6. **Metric interpretation:** The reported mIoU comes from the project's custom metric implementation. Its protocol should be checked against the required benchmark evaluation protocol before official comparisons.
7. **Hardware and timing:** The reported elapsed time applies only to this small trial on MPS and is not a full-training runtime estimate.
8. **No controlled comparison:** No controlled comparison against another architecture or encoder has been completed.

## 7. Next Steps

1. Verify the dataset's image color convention and confirm that preprocessing matches the pretrained encoder's expected input.
2. Inspect image-mask alignment, class-channel mapping, and target values.
3. Investigate per-class prediction density and validation behavior.
4. Confirm the required DACL10K evaluation protocol and metric implementation.
5. Run a reproducible experiment with a larger training and validation subset.
6. If the preliminary checks are satisfactory, train for more epochs and evaluate on the full validation set.
7. Record the configuration, dataset split, device, elapsed time, losses, mIoU, and checkpoint status for each experiment.
8. Compare alternative architectures or encoders only after establishing a consistent evaluation protocol.
9. Keep datasets, virtual environments, and model checkpoints outside Git unless the project explicitly requires otherwise.

## Conclusion

The smoke test and one-epoch fine-tuning trial completed successfully. The pipeline can execute on Apple Silicon using MPS, and the fine-tuning trial saved checkpoints.

However, the current mIoU is low and the experiment used only a small subset of the dataset. The next priority is to investigate preprocessing, class predictions, and metric correctness, then conduct a larger and reproducible validation experiment before reporting final benchmark conclusions.