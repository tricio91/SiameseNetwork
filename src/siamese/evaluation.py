"""Gallery evaluation and the plots used to inspect a trained model."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from .gallery import VALID_EXTENSIONS, classify_image


def evaluate_gallery(templates_dir: str, gallery: dict, model, transform, device) -> dict:
    """Classify every image under `templates_dir` and report gallery accuracy."""
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    sims: List[float] = []

    for cls_name in sorted(os.listdir(templates_dir)):
        cls_dir = os.path.join(templates_dir, cls_name)
        if not os.path.isdir(cls_dir) or cls_name not in gallery["class_names"]:
            continue
        for f in os.listdir(cls_dir):
            if not any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                continue
            try:
                img = Image.open(os.path.join(cls_dir, f)).convert("RGB")
                pred, sim, _ = classify_image(img, gallery, model, transform, device, threshold=0.0)
                y_true.append(cls_name)
                y_pred.append(pred)
                sims.append(sim)
            except Exception:
                continue

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
    print(f"Gallery accuracy: {acc:.2%} ({len(y_true)} samples)")
    return {"y_true": y_true, "y_pred": y_pred, "similarities": sims, "accuracy": acc}


def classification_text_report(results: dict, class_names: List[str]) -> str:
    """Return sklearn's per-class precision/recall/F1 report as a string."""
    return classification_report(
        results["y_true"], results["y_pred"], labels=class_names, zero_division=0
    )


def save_classification_csv(results: dict, class_names: List[str], path) -> Path:
    """Write per-class precision/recall/F1/support plus the global rows to a CSV.

    Columns: class, precision, recall, f1_score, support. The per-class rows are
    followed by the 'accuracy', 'macro avg' and 'weighted avg' summary rows, so
    the file holds both the per-class and the overall metrics.
    """
    report = classification_report(
        results["y_true"], results["y_pred"], labels=class_names,
        zero_division=0, output_dict=True,
    )

    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names:
            row = report.get(name)
            if row is None:  # class absent from both y_true and y_pred
                continue
            writer.writerow([
                name, f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}", int(row["support"]),
            ])
        # Global / summary rows.
        for name in ("accuracy", "macro avg", "weighted avg"):
            entry = report.get(name)
            if entry is None:
                continue
            if name == "accuracy":  # sklearn stores accuracy as a bare float
                writer.writerow(["accuracy", "", "", f"{entry:.4f}", int(report["macro avg"]["support"])])
            else:
                writer.writerow([
                    name, f"{entry['precision']:.4f}", f"{entry['recall']:.4f}",
                    f"{entry['f1-score']:.4f}", int(entry["support"]),
                ])
    return path


def plot_loss_curves(train_losses, val_losses, ax=None):
    """Plot train/validation loss curves and mark the best epoch."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, "b-", label="Train")
    ax.plot(val_losses, "r-", label="Validation")
    if val_losses:
        ax.axvline(int(np.argmin(val_losses)), color="g", ls="--", alpha=0.5, label="Best")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_confusion_matrix(y_true, y_pred, figsize=(12, 10)):
    """Plot the row-normalized confusion matrix; returns (matrix, labels)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # With many classes the per-cell numbers and tick labels are unreadable
    # (and slow to render), so drop them past a threshold.
    many = len(labels) > 25
    plt.figure(figsize=figsize)
    sns.heatmap(cm_norm, annot=not many, fmt=".2f", cmap="Blues",
                xticklabels=not many, yticklabels=not many, vmin=0, vmax=1)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized confusion matrix ({len(labels)} classes)")
    if not many:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return cm, labels


def plot_similarity_distribution(results: dict):
    """Overlay the similarity histograms of correct vs. incorrect predictions."""
    import matplotlib.pyplot as plt

    correct = [s for t, p, s in zip(results["y_true"], results["y_pred"], results["similarities"]) if t == p]
    incorrect = [s for t, p, s in zip(results["y_true"], results["y_pred"], results["similarities"]) if t != p]

    plt.figure(figsize=(10, 4))
    if correct:
        plt.hist(correct, bins=30, alpha=0.7, label=f"Correct ({len(correct)})")
    if incorrect:
        plt.hist(incorrect, bins=30, alpha=0.7, label=f"Incorrect ({len(incorrect)})")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.title("Similarity distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
