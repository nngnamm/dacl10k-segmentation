"""Multi-GPU (DDP) training engine built on Hugging Face Accelerate.

Fine-tuning features:
  * discriminative learning rates: small lr for the pretrained encoder,
    larger lr for the new decoder/head;
  * encoder "freeze" implemented as lr = 0 for `freeze_epochs` epochs
    (DDP-safe: no parameter changes requires_grad after wrapping);
  * gradient accumulation and gradient clipping.
"""

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dacl10k.utils.metrics import MultiLabelIoUMetric


def build_param_groups(model: nn.Module, lr_encoder: float, lr_decoder: float):
    """Split parameters into encoder / non-encoder groups with separate lrs.

    Returns (groups, has_encoder_group).
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return [{"params": list(model.parameters()), "lr": lr_decoder}], False

    encoder_ids = {id(p) for p in encoder.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    groups = [
        {"params": list(encoder.parameters()), "lr": lr_encoder},
        {"params": other_params, "lr": lr_decoder},
    ]
    return groups, True


def build_scheduler(optimizer, epochs: int, freeze_epochs: int, has_encoder_group: bool):
    """Per-epoch cosine decay (floor 1% of base lr); encoder lr is 0 while frozen."""

    def cosine(epoch: int) -> float:
        return max(0.5 * (1.0 + math.cos(math.pi * min(epoch, epochs) / epochs)), 1e-2)

    def encoder_factor(epoch: int) -> float:
        return 0.0 if epoch < freeze_epochs else cosine(epoch)

    lambdas = [encoder_factor, cosine] if has_encoder_group else [cosine]
    return LambdaLR(optimizer, lr_lambda=lambdas)


class Trainer:
    """Distributed-ready trainer for multi-label semantic segmentation."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """
        Args:
            cfg: configuration dict (see configs/efficientnet_base.yaml).
            model: segmentation model returning raw logits.
            train_loader, val_loader: DataLoaders.
            criterion: loss (defaults to nn.BCEWithLogitsLoss()).
            accelerator: optional Accelerator; create it with
                gradient_accumulation_steps=cfg["grad_accum"] if you pass one.
        """
        self.cfg = cfg
        if accelerator is None:
            mp = cfg.get("mixed_precision", "no") if torch.cuda.is_available() else "no"
            accelerator = Accelerator(
                mixed_precision=mp,
                gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
            )
        self.accelerator = accelerator
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        self.grad_clip = float(cfg.get("grad_clip", 0.0))

        lr = float(cfg.get("lr", 1e-4))
        lr_encoder = float(cfg.get("lr_encoder", lr))
        lr_decoder = float(cfg.get("lr_decoder", lr))
        weight_decay = float(cfg.get("weight_decay", 1e-2))
        epochs = max(int(cfg.get("epochs", 30)), 1)
        freeze_epochs = int(cfg.get("freeze_epochs", 0))

        groups, has_encoder = build_param_groups(model, lr_encoder, lr_decoder)
        optimizer = AdamW(groups, weight_decay=weight_decay)

        # The scheduler is NOT passed to accelerator.prepare(): a prepared
        # scheduler steps `num_processes` times per call, which would make the
        # cosine schedule run too fast on multiple GPUs.
        self.scheduler = build_scheduler(optimizer, epochs, freeze_epochs, has_encoder)

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(model, optimizer, train_loader, val_loader)

        num_classes = int(cfg.get("num_classes", 19))
        threshold = float(cfg.get("threshold", 0.5))
        self.metric = MultiLabelIoUMetric(num_classes=num_classes, threshold=threshold)

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (images, masks) from a dict or tuple/list batch."""
        if isinstance(batch, dict):
            return batch["image"], batch["mask"]
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch format: {type(batch)}")

    def _mean_across_processes(self, total: torch.Tensor, count: int) -> float:
        """Average a per-process loss sum over batches and over all GPUs."""
        mean_local = (total / max(count, 1)).reshape(1)
        return float(self.accelerator.gather(mean_local).mean().item())

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the mean loss over all GPUs."""
        self.model.train()
        total_loss = torch.zeros((), device=self.accelerator.device)
        num_batches = 0

        for batch in tqdm(
            self.train_loader,
            desc=f"train {epoch + 1}",
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        ):
            images, targets = self._unpack_batch(batch)
            targets = targets.to(dtype=torch.float32)

            # Inside accumulate(), the optimizer only steps every grad_accum batches.
            with self.accelerator.accumulate(self.model):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.grad_clip > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.detach()
            num_batches += 1

        return self._mean_across_processes(total_loss, num_batches)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation; return {"val_loss", "miou"}."""
        self.model.eval()
        self.metric.reset()
        total_loss = torch.zeros((), device=self.accelerator.device)
        num_batches = 0

        for batch in tqdm(
            self.val_loader,
            desc="val",
            leave=False,
            disable=not self.accelerator.is_local_main_process,
        ):
            images, targets = self._unpack_batch(batch)
            targets = targets.to(dtype=torch.float32)

            logits = self.model(images)
            total_loss += self.criterion(logits, targets).detach()
            num_batches += 1

            # Gather from all GPUs (drops padded duplicates) before updating the metric.
            g_logits, g_targets = self.accelerator.gather_for_metrics((logits, targets))
            self.metric.update(g_logits, g_targets)

        metrics = self.metric.compute()
        return {
            "val_loss": self._mean_across_processes(total_loss, num_batches),
            "miou": float(metrics["miou"]),
        }

    def save_checkpoint(self, path: str | Path, epoch: int, best_miou: float) -> None:
        """Save model, optimizer and scheduler state (main process only)."""
        if not self.accelerator.is_main_process:
            return
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, path: str | Path) -> tuple[int, float]:
        """Load a checkpoint; return (epoch, best_miou)."""
        checkpoint = torch.load(
            Path(path), map_location=self.accelerator.device, weights_only=False
        )
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_miou", 0.0))

    def fit(self, resume_path: str | Path | None = None) -> dict[str, float]:
        """Run the full train/validate loop; return the last epoch's metrics."""
        epochs = int(self.cfg.get("epochs", 30))
        checkpoint_dir = Path(self.cfg.get("checkpoint_dir", "checkpoints"))
        use_wandb = bool(self.cfg.get("use_wandb", False))

        start_epoch, best_miou = 0, -1.0
        if resume_path is not None:
            loaded_epoch, best_miou = self.load_checkpoint(resume_path)
            start_epoch = loaded_epoch + 1

        wandb_run = None
        if use_wandb and self.accelerator.is_main_process:
            import wandb

            wandb_run = wandb.init(
                project=self.cfg.get("wandb_project", "dacl10k-segmentation"),
                config=self.cfg,
                resume="allow",
            )

        last_metrics = {"train_loss": 0.0, "val_loss": 0.0, "miou": 0.0, "best_miou": best_miou}

        for epoch in range(start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()  # once per epoch, on every process

            val_loss, miou = val_metrics["val_loss"], val_metrics["miou"]
            is_best = miou > best_miou
            if is_best:
                best_miou = miou

            self.save_checkpoint(checkpoint_dir / "last.pt", epoch, best_miou)
            if is_best:
                self.save_checkpoint(checkpoint_dir / "best.pt", epoch, best_miou)
            self.accelerator.wait_for_everyone()

            lrs = [f"{g['lr']:.2e}" for g in self.optimizer.param_groups]
            self.accelerator.print(
                f"Epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | mIoU {miou:.4f} | best {best_miou:.4f} | lr {lrs}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_miou": miou, "best_miou": best_miou}
                )

            last_metrics = {
                "train_loss": train_loss, "val_loss": val_loss,
                "miou": miou, "best_miou": best_miou,
            }

        if wandb_run is not None:
            wandb_run.finish()
        return last_metrics