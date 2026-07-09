"""Train the embedding network and report retrieval metrics on the test split.

Usage:
    python scripts/train.py --config configs/cub200.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from siamese import (  # noqa: E402  (import after the sys.path bootstrap above)
    Config,
    build_dataloaders,
    build_model,
    compute_retrieval_metrics,
    evaluate_embeddings,
    fit,
    get_device,
    set_seed,
)


def main() -> None:
    """Parse the config, build the data + model, train, then evaluate on the test set."""
    parser = argparse.ArgumentParser(description="Train the Siamese embedding network.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (optional).")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    loaders = build_dataloaders(cfg)
    model = build_model(cfg, device, pretrained=True)

    fit(model, loaders, cfg, device)


if __name__ == "__main__":
    main()

   