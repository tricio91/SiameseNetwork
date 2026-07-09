"""Gallery construction and gallery-based classification.

The gallery is a set of class prototypes (one embedding per class). A query is
classified by cosine similarity to the nearest prototype, with an optional
threshold for open-set rejection.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@torch.no_grad()
def extract_embedding(img_pil: Image.Image, model: nn.Module, transform, device) -> np.ndarray:
    """Return the L2-normalized embedding of a single PIL image."""
    model.eval()
    tensor = transform(img_pil).unsqueeze(0).to(device)
    return model(tensor).cpu().numpy().squeeze()


def filter_outliers(embeddings: np.ndarray, keep_fraction: float = 0.975):
    """Keep the fraction of samples closest to the class centroid (cosine distance)."""
    if keep_fraction >= 1.0 or len(embeddings) < 3:
        return embeddings, np.arange(len(embeddings))

    n_keep = max(2, int(round(len(embeddings) * keep_fraction)))
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    distances = 1.0 - (embeddings / norms) @ centroid

    keep = np.argsort(distances)[:n_keep]
    return embeddings[keep], keep


def apply_pca(embeddings: np.ndarray, pca: PCA, normalize: bool = True) -> np.ndarray:
    """Project embeddings with a fitted PCA and optionally re-normalize them."""
    transformed = pca.transform(embeddings)
    if normalize:
        transformed = transformed / (np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-8)
    return transformed


def _iter_class_images(cls_dir: str):
    """Yield full paths of valid image files inside a class directory."""
    for f in os.listdir(cls_dir):
        if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            yield os.path.join(cls_dir, f)


def fit_pca_on_embeddings(
    templates_dir: str, model: nn.Module, transform, device,
    n_components: int = 128, verbose: bool = True,
) -> PCA:
    """Fit a PCA on every embedding found under `templates_dir`."""
    model.eval()
    all_embeddings: List[np.ndarray] = []

    for cls_name in sorted(os.listdir(templates_dir)):
        cls_dir = os.path.join(templates_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for img_path in _iter_class_images(cls_dir):
            try:
                img = Image.open(img_path).convert("RGB")
                all_embeddings.append(extract_embedding(img, model, transform, device))
            except Exception:
                continue

    all_embeddings = np.stack(all_embeddings)
    n_comp = min(n_components, all_embeddings.shape[1], len(all_embeddings))
    pca = PCA(n_components=n_comp)
    pca.fit(all_embeddings)

    if verbose:
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA: {all_embeddings.shape[1]} -> {n_comp} dims, {var:.1f}% variance explained")
    return pca


def build_gallery(
    templates_dir: str, model: nn.Module, transform, device,
    keep_fraction: float = 0.975, proto_mode: str = "mean",
    pca: Optional[PCA] = None, verbose: bool = True,
) -> dict:
    """Build one prototype per class, with optional PCA and outlier filtering.

    Returns a dict with prototypes, per-class embeddings, class names, the PCA
    object and the config used.
    """
    model.eval()
    class_embeddings: Dict[str, np.ndarray] = {}
    class_prototypes: Dict[str, np.ndarray] = {}
    class_names: List[str] = []

    for cls_name in sorted(os.listdir(templates_dir)):
        cls_dir = os.path.join(templates_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        embeddings = []
        for img_path in _iter_class_images(cls_dir):
            try:
                img = Image.open(img_path).convert("RGB")
                embeddings.append(extract_embedding(img, model, transform, device))
            except Exception:
                continue
        if not embeddings:
            if verbose:
                print(f"Warning: no images for '{cls_name}'")
            continue

        embeddings = np.stack(embeddings)
        if pca is not None:
            embeddings = apply_pca(embeddings, pca, normalize=True)
        embeddings, _ = filter_outliers(embeddings, keep_fraction)

        if proto_mode == "medoid":
            centroid = embeddings.mean(axis=0)
            prototype = embeddings[np.argmin(np.linalg.norm(embeddings - centroid, axis=1))]
        else:
            prototype = embeddings.mean(axis=0)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-8)

        class_embeddings[cls_name] = embeddings
        class_prototypes[cls_name] = prototype
        class_names.append(cls_name)

    if verbose:
        print(f"Gallery built: {len(class_names)} classes")

    return {
        "prototypes": class_prototypes,
        "embeddings": class_embeddings,
        "class_names": class_names,
        "pca": pca,
        "config": {"keep_fraction": keep_fraction, "proto_mode": proto_mode},
    }


def classify_image(
    img_pil: Image.Image, gallery: dict, model: nn.Module, transform, device,
    threshold: float = 0.5,
) -> Tuple[str, float, Dict[str, float]]:
    """Classify a PIL image against the gallery; below `threshold` returns 'unknown'.

    Returns (predicted_class, best_similarity, all_similarities).
    """
    embedding = extract_embedding(img_pil, model, transform, device)

    pca = gallery.get("pca")
    if pca is not None:
        embedding = apply_pca(embedding.reshape(1, -1), pca, normalize=True).squeeze()
    else:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

    similarities = {
        cls: float(np.dot(embedding, proto)) for cls, proto in gallery["prototypes"].items()
    }
    best_class = max(similarities, key=similarities.get)
    best_sim = similarities[best_class]

    predicted = "unknown" if best_sim < threshold else best_class
    return predicted, best_sim, similarities


def classify_batch(
    images: List[Image.Image], gallery: dict, model: nn.Module, transform, device,
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    """Classify a list of PIL images, returning (class, similarity) for each."""
    return [classify_image(img, gallery, model, transform, device, threshold)[:2] for img in images]


def save_gallery(gallery: dict, gallery_path: Path, pca_path: Optional[Path] = None) -> None:
    """Serialize prototypes (and, if present, the PCA transformer) to disk."""
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {
        "prototypes": {k: v.tolist() for k, v in gallery["prototypes"].items()},
        "class_names": gallery["class_names"],
        "config": gallery["config"],
    }
    with open(gallery_path, "wb") as f:
        pickle.dump(to_save, f)
    if pca_path is not None and gallery.get("pca") is not None:
        with open(pca_path, "wb") as f:
            pickle.dump(gallery["pca"], f)


def load_gallery(gallery_path: Path, pca_path: Optional[Path] = None) -> dict:
    """Load prototypes (and optional PCA) from disk into a gallery dict."""
    with open(gallery_path, "rb") as f:
        data = pickle.load(f)
    gallery = {
        "prototypes": {k: np.array(v) for k, v in data["prototypes"].items()},
        "class_names": data["class_names"],
        "config": data.get("config", {}),
        "pca": None,
    }
    if pca_path is not None and Path(pca_path).exists():
        with open(pca_path, "rb") as f:
            gallery["pca"] = pickle.load(f)
    return gallery
