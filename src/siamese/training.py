"""Optimizer, LR schedule and the training loop with early stopping."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .losses import hard_triplet_loss
from .model import SimpleEmbeddingNet


class WarmupCosineScheduler:
    """Linear warmup for a few epochs, then cosine decay down to `min_lr`."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        """Store the schedule parameters; call `step()` once per epoch."""
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self) -> None:
        """Advance one epoch and update the LR of every parameter group."""
        self.current_epoch += 1
        if self.current_epoch < self.warmup_epochs:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_ratio", 1.0)

    def get_last_lr(self) -> List[float]:
        """Return the current learning rate of each parameter group."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


def build_optimizer(model: SimpleEmbeddingNet, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with a 0.1x learning rate on the backbone and full rate on the head."""
    return torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()),
             "lr": cfg.learning_rate * 0.1, "lr_ratio": 0.1},
            {"params": list(model.proj.parameters()),
             "lr": cfg.learning_rate, "lr_ratio": 1.0},
        ],
        weight_decay=cfg.weight_decay,
    )


def train_epoch(model, dataloader, optimizer, device, margin=0.3) -> float:
    """Run one training epoch and return the mean triplet loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = hard_triplet_loss(model(images), labels, margin=margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(model, dataloader, device, margin=0.3) -> float:
    """Run one validation pass and return the mean triplet loss."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        loss = hard_triplet_loss(model(images), labels, margin=margin)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path: Path) -> None:
    """Persist model + optimizer state and the losses at this epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # Cast to plain floats: numpy scalars are rejected by the safe
            # (weights_only=True) unpickler when the checkpoint is loaded back.
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        path,
    )


def load_checkpoint(model, path: Path, device) -> dict:
    """Load weights from a checkpoint into `model` and return the checkpoint dict."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def fit(model, loaders: Dict, cfg: Config, device) -> Dict[str, list]:
    """Train with validation and early stopping, saving the best checkpoint.

    Returns the train/val loss history. If a checkpoint already exists it is
    loaded and training is skipped.
    """
    if cfg.checkpoint_path.exists():
        checkpoint = load_checkpoint(model, cfg.checkpoint_path, device)
        print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch'] + 1} "
              f"(val_loss={checkpoint.get('val_loss', 'N/A')})")
        return {"train_losses": [], "val_losses": []}

    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.warmup_epochs, cfg.n_epochs, cfg.learning_rate, cfg.min_lr
    )

    best_val = float("inf")
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(cfg.n_epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, device, cfg.margin)
        val_loss = validate(model, loaders["val"], device, cfg.margin)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        line = f"Epoch {epoch + 1:2d}/{cfg.n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr:.2e}"

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, cfg.checkpoint_path)
            print(line + "  [saved]")
        else:
            patience_counter += 1
            print(line + f"  (no improvement {patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training complete. Best val loss: {best_val:.4f}")
    return {"train_losses": train_losses, "val_losses": val_losses}
