"""Evaluate the classifier over a whole image folder and report performance.

Classifies every image under a folder (one subfolder per class) against the
saved gallery, prints per-class precision/recall/F1 and accuracy, and produces
the confusion matrix + similarity-distribution plots. The plots are saved as
PNG files and, unless --no-show is passed, also popped up in windows.

Usage:
    python scripts/evaluate.py --config configs/cub200.yaml
    python scripts/evaluate.py --config configs/cub200.yaml --data-dir data/CUB_200_2011/images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running this file directly, without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from siamese import (  # noqa: E402
    Config,
    build_model,
    classification_text_report,
    evaluate_gallery,
    get_device,
    load_gallery,
    plot_confusion_matrix,
    plot_similarity_distribution,
    save_classification_csv,
)
from siamese.data import get_transforms  # noqa: E402
from siamese.training import load_checkpoint  # noqa: E402


def main() -> None:
    """Load model + gallery, evaluate a folder, then save and show the metric plots."""
    parser = argparse.ArgumentParser(description="Evaluate the classifier over an image folder.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (optional).")
    parser.add_argument(
        "--data-dir", default=None,
        help="Folder to evaluate (one subfolder per class). Defaults to the config's data_dir.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save the plots. Defaults to <checkpoints_dir>/evaluation.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plots without popping up windows.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    device = get_device()
    print(f"Device: {device}")

    data_dir = args.data_dir or cfg.data_dir
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.checkpoints_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load the trained model and the prototype gallery ---
    model = build_model(cfg, device, pretrained=False)
    load_checkpoint(model, cfg.checkpoint_path, device)
    model.eval()
    transform = get_transforms(cfg, "val")
    gallery = load_gallery(cfg.gallery_path, cfg.pca_path)

    # --- Classify every image in the folder ---
    print(f"Evaluating images under: {data_dir}")
    results = evaluate_gallery(data_dir, gallery, model, transform, device)
    if not results["y_true"]:
        print("No images were classified. Check that --data-dir has one subfolder per class "
              "and that those class names match the gallery.")
        return

    # --- Text report: per-class precision / recall / F1 ---
    report = classification_text_report(results, gallery["class_names"])
    print("\n" + report)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(f"Accuracy: {results['accuracy']:.4f}\n\n{report}")
    print(f"Saved report to {report_path}")

    # Same metrics as a CSV (per-class rows + accuracy/macro/weighted globals),
    # easier to sort or open in a spreadsheet.
    csv_path = save_classification_csv(results, gallery["class_names"], out_dir / "classification_report.csv")
    print(f"Saved report to {csv_path}")

    # --- Plots: build them, save to disk, then show as pop-ups ---
    plot_confusion_matrix(results["y_true"], results["y_pred"])
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    print(f"Saved {cm_path}")

    plot_similarity_distribution(results)
    sim_path = out_dir / "similarity_distribution.png"
    plt.savefig(sim_path, dpi=150, bbox_inches="tight")
    print(f"Saved {sim_path}")

    if args.no_show:
        plt.close("all")
    else:
        print("\nShowing plots — close the windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
