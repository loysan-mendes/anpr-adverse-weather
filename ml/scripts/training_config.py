"""Training configuration helpers that can be checked without loading ML runtimes."""

import argparse
import importlib


def restoration_parser(manifest_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["haze", "rain", "blur"])
    parser.add_argument("--manifest", default=str(manifest_path))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ssim_weight", type=float, default=0.3)
    parser.add_argument("--edge_weight", type=float, default=1.0,
                        help="Sobel weight; use --ssim_weight 0 --edge_weight 0 for L1 only")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out_dir", help="Separate candidate output directory; defaults to models/restoration/<condition>")
    return parser


def checkpoint_score(condition_accuracy, severity_accuracy):
    """Harmonic mean penalizes checkpoints that sacrifice either quality head."""
    return 2 * condition_accuracy * severity_accuracy / max(condition_accuracy + severity_accuracy, 1e-12)


def ssim_function(weight):
    if weight < 0:
        raise ValueError("SSIM weight must be nonnegative")
    if weight == 0:
        return None
    try:
        module = importlib.import_module("torchmetrics.functional.image")
        return module.structural_similarity_index_measure
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("SSIM requires torchmetrics. Install requirements or pass --ssim_weight 0.") from exc
