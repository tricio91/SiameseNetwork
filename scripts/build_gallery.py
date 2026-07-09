"""Fit PCA, build the class-prototype gallery and save both to disk.

Usage:
    python scripts/build_gallery.py --config configs/cub200.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running this file directly, without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from siamese import Config, build_model, get_device  # noqa: E402
from siamese.data import get_transforms  # noqa: E402
from siamese.gallery import build_gallery, fit_pca_on_embeddings, save_gallery  # noqa: E402
from siamese.training import load_checkpoint  # noqa: E402


def main() -> None:
    """Load the trained model, fit PCA, build the gallery and persist the artifacts."""
    parser = argparse.ArgumentParser(description="Build the prototype gallery.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (optional).")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    device = get_device()

    model = build_model(cfg, device, pretrained=False)
    load_checkpoint(model, cfg.checkpoint_path, device)
    model.eval()

    transform = get_transforms(cfg, "val")

    pca = None
    if cfg.pca_components:
        pca = fit_pca_on_embeddings(
            cfg.data_dir, model, transform, device, n_components=cfg.pca_components
        )

    gallery = build_gallery(
        cfg.data_dir, model, transform, device,
        keep_fraction=cfg.keep_fraction, proto_mode=cfg.proto_mode, pca=pca,
    )
    save_gallery(gallery, cfg.gallery_path, cfg.pca_path)
    print(f"Saved gallery to {cfg.gallery_path}")


if __name__ == "__main__":
    main()
