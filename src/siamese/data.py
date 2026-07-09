"""Datasets, transforms, samplers and the stratified split."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from .config import Config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Sample = Tuple[str, int]


class LetterboxResize:
    """Resize so the longest side equals `size` (aspect ratio kept), then pad to a square.

    Unlike Resize((size, size)) this never distorts the image: it scales the
    picture to fit inside a size x size canvas and fills the leftover margin with
    a constant color, so nothing is cropped or stretched.
    """

    def __init__(self, size: int, fill: int = 0):
        """Store the target side length and the padding color (0 = black)."""
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """Scale `img` to fit the square, then center-pad it to size x size."""
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


def get_transforms(cfg: Config, mode: str = "train") -> transforms.Compose:
    """Build the image transform pipeline for 'train' (with augmentation) or 'val'."""
    normalize = transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std)

    if mode == "train":
        translate_fraction = cfg.translate_pixels / cfg.img_size
        return transforms.Compose([
            LetterboxResize(cfg.img_size),
            transforms.RandomAffine(
                degrees=cfg.rotation_degrees,
                translate=(translate_fraction, translate_fraction),
                fill=0,
            ),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        LetterboxResize(cfg.img_size),
        transforms.ToTensor(),
        normalize,
    ])


class SplitContrastiveDataset(Dataset):
    """Wraps a pre-computed list of (path, label) samples and applies a transform."""

    def __init__(self, samples, class_to_idx, idx_to_class, transform=None, img_size=224):
        """Store the samples and record per-class counts."""
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.img_size = img_size

        self.class_counts = defaultdict(int)
        for _, label in samples:
            self.class_counts[idx_to_class[label]] += 1

    def __len__(self) -> int:
        """Number of samples in the split."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load one image (falling back to a black image on read errors) and its label."""
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


class BalancedBatchSampler(Sampler):
    """Yields P-classes x K-samples batches so every batch can form valid triplets."""

    def __init__(self, labels, n_classes: int = 8, n_samples: int = 4, n_batches: int = 100):
        """Group sample indices by class and store the batch geometry."""
        self.labels = np.array(labels)
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_batches = n_batches

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.n_classes_total = len(self.classes)

    def __iter__(self):
        """Produce `n_batches` batches, each with n_classes x n_samples indices."""
        for cls in self.classes:
            random.shuffle(self.class_to_indices[cls])

        class_pointers = {cls: 0 for cls in self.classes}

        for _ in range(self.n_batches):
            batch_indices: List[int] = []
            selected = random.sample(self.classes, min(self.n_classes, self.n_classes_total))
            for cls in selected:
                indices = self.class_to_indices[cls]
                ptr = class_pointers[cls]
                for _ in range(self.n_samples):
                    batch_indices.append(indices[ptr % len(indices)])
                    ptr += 1
                class_pointers[cls] = ptr
            yield batch_indices

    def __len__(self) -> int:
        """Number of batches produced per epoch."""
        return self.n_batches


def create_stratified_split(cfg: Config, verbose: bool = True):
    """Scan an ImageFolder-style directory and split it into stratified train/val/test."""
    root_path = Path(cfg.data_dir)

    all_samples: List[str] = []
    all_labels: List[int] = []
    class_to_idx, idx_to_class = {}, {}

    for cls_idx, cls_dir in enumerate(sorted(root_path.iterdir())):
        if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
            continue
        class_to_idx[cls_dir.name] = cls_idx
        idx_to_class[cls_idx] = cls_dir.name
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                all_samples.append(str(img_path))
                all_labels.append(cls_idx)

    # train vs (val + test)
    train_x, temp_x, train_y, temp_y = train_test_split(
        all_samples, all_labels, train_size=cfg.train_ratio,
        stratify=all_labels, random_state=cfg.seed,
    )
    # val vs test
    val_share = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, train_size=val_share, stratify=temp_y, random_state=cfg.seed,
    )

    train_data = list(zip(train_x, train_y))
    val_data = list(zip(val_x, val_y))
    test_data = list(zip(test_x, test_y))

    if verbose:
        n = len(all_samples)
        print(f"Dataset: {n} samples, {len(class_to_idx)} classes")
        print(f"  Train: {len(train_data)} ({100 * len(train_data) / n:.0f}%)")
        print(f"  Val:   {len(val_data)} ({100 * len(val_data) / n:.0f}%)")
        print(f"  Test:  {len(test_data)} ({100 * len(test_data) / n:.0f}%)")

    return train_data, val_data, test_data, class_to_idx, idx_to_class


def build_dataloaders(cfg: Config, verbose: bool = True):
    """Create the train/val/test datasets, samplers and dataloaders from a Config."""
    train_split, val_split, test_split, class_to_idx, idx_to_class = create_stratified_split(
        cfg, verbose=verbose
    )

    train_ds = SplitContrastiveDataset(
        train_split, class_to_idx, idx_to_class, get_transforms(cfg, "train"), cfg.img_size
    )
    val_ds = SplitContrastiveDataset(
        val_split, class_to_idx, idx_to_class, get_transforms(cfg, "val"), cfg.img_size
    )
    test_ds = SplitContrastiveDataset(
        test_split, class_to_idx, idx_to_class, get_transforms(cfg, "val"), cfg.img_size
    )

    per_batch = cfg.n_classes_per_batch * cfg.n_samples_per_class
    train_sampler = BalancedBatchSampler(
        [lbl for _, lbl in train_ds.samples],
        n_classes=cfg.n_classes_per_batch, n_samples=cfg.n_samples_per_class,
        n_batches=(len(train_ds) // per_batch) * cfg.augmentation_factor,
    )
    val_sampler = BalancedBatchSampler(
        [lbl for _, lbl in val_ds.samples],
        n_classes=cfg.n_classes_per_batch, n_samples=cfg.n_samples_per_class,
        n_batches=max(1, len(val_ds) // per_batch),
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    if verbose:
        print(f"Dataloaders: {len(train_sampler)} train batches, {len(val_sampler)} val batches")

    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "class_to_idx": class_to_idx, "idx_to_class": idx_to_class,
    }
