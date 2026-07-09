"""Experiment configuration.

All hyperparameters and paths live here in a single dataclass so a run can be
fully described (and reproduced) from one YAML file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass
class Config:
    """Holds every knob for a training/evaluation run."""

    # --- Paths ---
    data_dir: str = "data/images"          # root with one subfolder per class
    checkpoints_dir: str = "checkpoints"   # where weights and artifacts are written

    # --- Model ---
    embed_dim: int = 512                   # size of the output embedding
    backbone: str = "resnet34"             # ImageNet-pretrained feature extractor
    freeze_bn: bool = True                 # keep BatchNorm frozen while fine-tuning

    # --- Image preprocessing ---
    img_size: int = 224
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Augmentation ---
    rotation_degrees: float = 2.0
    translate_pixels: int = 10
    augmentation_factor: int = 4           # multiplies the number of batches per epoch

    # --- Training ---
    batch_size: int = 32                   # used for evaluation/inference only
    n_classes_per_batch: int = 8           # P in the P x K balanced batches
    n_samples_per_class: int = 4           # K in the P x K balanced batches
    learning_rate: float = 1e-4
    n_epochs: int = 30
    margin: float = 0.3                    # triplet loss margin
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    patience: int = 5                      # early-stopping patience
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42

    # --- Split ratios ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Gallery ---
    keep_fraction: float = 0.975           # outlier filtering: fraction kept per class
    proto_mode: str = "mean"               # "mean" (centroid) or "medoid"
    pca_components: Optional[int] = 128    # None disables PCA

    @property
    def checkpoint_path(self) -> Path:
        """Path to the best fine-tuned weights."""
        return Path(self.checkpoints_dir) / "backbone.pth"

    @property
    def gallery_path(self) -> Path:
        """Path to the saved class prototypes."""
        return Path(self.checkpoints_dir) / "gallery.pkl"

    @property
    def pca_path(self) -> Path:
        """Path to the fitted PCA transformer."""
        return Path(self.checkpoints_dir) / "pca.pkl"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config from a YAML file (missing keys fall back to defaults)."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Dump the current Config to a YAML file."""
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))
