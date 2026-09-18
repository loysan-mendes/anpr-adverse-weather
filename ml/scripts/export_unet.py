"""
6b: One-time TorchScript export for the UNet restoration models.

TorchScript compiles the UNet graph to a serialised, optimised IR that the
PyTorch runtime can execute without Python overhead. On CPU this gives a
20-40% inference speedup with zero accuracy cost -- no retraining needed.

Exports model_scripted.pt alongside the existing best_model.pt for each
condition. decision_engine.py will load the scripted version when present
and fall back to eager mode when not.

Usage (run once after training all three restoration models):
    python ml/scripts/export_unet.py
    python ml/scripts/export_unet.py --conditions haze  # single condition
"""

import argparse
from pathlib import Path

import torch

from unet_model import UNet

ML_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = ML_DIR / "models" / "restoration"
CONDITIONS = ["haze", "rain", "blur"]


def export_condition(condition: str, device: torch.device) -> None:
    ckpt_path = MODEL_ROOT / condition / "best_model.pt"
    out_path = MODEL_ROOT / condition / "model_scripted.pt"

    if not ckpt_path.exists():
        print(f"  [{condition}] SKIP -- {ckpt_path} not found (train first)")
        return

    print(f"  [{condition}] Loading checkpoint ... ", end="", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    is_residual = any("outc_raw" in k for k in ckpt["model_state"].keys())
    model = UNet(base=32, residual=is_residual)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print("done")

    print(f"  [{condition}] TorchScript tracing ... ", end="", flush=True)
    # torch.jit.script works on UNet because it has no data-dependent
    # control flow -- the graph is the same shape for every input.
    # Use script (not trace) so the model is compiled, not frozen to one
    # specific input shape.
    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        print(f"script failed ({e}), falling back to trace")
        # Trace with a representative dummy input (img_size stored in ckpt).
        img_size = ckpt.get("img_size", 512)
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        scripted = torch.jit.trace(model, dummy)

    scripted.save(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"done  ->  {out_path}  ({size_mb:.1f} MB)")

    # Quick sanity check: outputs should match eager mode to float32 tolerance.
    print(f"  [{condition}] Sanity checking ... ", end="", flush=True)
    img_size = ckpt.get("img_size", 512)
    dummy = torch.rand(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        eager_out = model(dummy)
        script_out = scripted(dummy)
    max_diff = (eager_out - script_out).abs().max().item()
    if max_diff > 1e-4:
        print(f"WARNING: max diff = {max_diff:.2e} (larger than expected)")
    else:
        print(f"OK (max diff = {max_diff:.2e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITIONS,
        help=f"which conditions to export (default: all -- {CONDITIONS})",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting UNet models to TorchScript on device: {device}\n")

    for cond in args.conditions:
        if cond not in CONDITIONS:
            print(f"  [{cond}] SKIP -- unknown condition (choose from {CONDITIONS})")
            continue
        export_condition(cond, device)

    print("\nDone. decision_engine.py will automatically use model_scripted.pt "
          "when present, falling back to best_model.pt otherwise.\n"
          "Tip: re-run this script after retraining to refresh the scripted weights.")


if __name__ == "__main__":
    main()
