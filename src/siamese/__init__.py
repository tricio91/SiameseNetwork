"""Siamese metric-learning image classification.

A ResNet backbone trained with hard-mined triplet loss produces embeddings that
are classified by nearest-prototype lookup against a gallery.
"""
from .config import Config
from .data import (
    BalancedBatchSampler,
    SplitContrastiveDataset,
    build_dataloaders,
    create_stratified_split,
    get_transforms,
)
from .evaluation import (
    classification_text_report,
    evaluate_gallery,
    plot_confusion_matrix,
    plot_loss_curves,
    plot_similarity_distribution,
    save_classification_csv,
)
from .gallery import build_gallery, classify_image, fit_pca_on_embeddings, load_gallery, save_gallery
from .losses import hard_triplet_loss
from .metrics import compute_retrieval_metrics, evaluate_embeddings
from .model import SimpleEmbeddingNet, build_model
from .training import fit, train_epoch, validate
from .utils import get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "Config",
    "SimpleEmbeddingNet",
    "build_model",
    "hard_triplet_loss",
    "get_transforms",
    "SplitContrastiveDataset",
    "BalancedBatchSampler",
    "create_stratified_split",
    "build_dataloaders",
    "fit",
    "train_epoch",
    "validate",
    "evaluate_embeddings",
    "compute_retrieval_metrics",
    "evaluate_gallery",
    "classification_text_report",
    "plot_confusion_matrix",
    "plot_similarity_distribution",
    "plot_loss_curves",
    "save_classification_csv",
    "build_gallery",
    "fit_pca_on_embeddings",
    "classify_image",
    "save_gallery",
    "load_gallery",
    "get_device",
    "set_seed",
]
