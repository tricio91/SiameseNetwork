"""Classify a single image against a saved gallery and print the top-5 matches.

Usage:
    python scripts/predict.py --image path/to/bird.jpg --config configs/cub200.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running this file directly, without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image  # noqa: E402

from siamese import Config, build_model, get_device  # noqa: E402
from siamese.data import get_transforms  # noqa: E402
from siamese.gallery import classify_image, load_gallery  # noqa: E402
from siamese.training import load_checkpoint  # noqa: E402


def main() -> None:
    """Load model + gallery, classify the given image and print the ranked matches."""
    parser = argparse.ArgumentParser(description="Classify one image with the gallery.")
    parser.add_argument("--image", required=True, help="Path to the image to classify.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (optional).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Rejection threshold.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    device = get_device()

    model = build_model(cfg, device, pretrained=False)
    load_checkpoint(model, cfg.checkpoint_path, device)
    model.eval()

    transform = get_transforms(cfg, "val")
    gallery = load_gallery(cfg.gallery_path, cfg.pca_path)

    img = Image.open(args.image).convert("RGB")
    pred, sim, all_sims = classify_image(
        img, gallery, model, transform, device, threshold=args.threshold
    )

    print(f"Prediction: {pred} (similarity {sim:.3f})\n")
    print("Top-5:")
    for cls, s in sorted(all_sims.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cls:<30} {s:.3f}")


if __name__ == "__main__":
    main()
