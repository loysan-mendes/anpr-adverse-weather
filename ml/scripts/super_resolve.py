"""
Phase 6 - Step 1: Real-ESRGAN 4x super-resolution for plate crops.

Inference only -- uses the official RealESRGAN_x4plus.pth pretrained
weights (auto-downloaded on first run, ~64MB, needs internet). We are NOT
training/fine-tuning this: a super-resolution GAN needs tens of thousands
of paired images to train well, far beyond our 47-photo dataset. The
pretrained model generalizes well to small, low-res crops like plates
without any fine-tuning.

Prereq: run fix_basicsr_compat.py once first.

Usage (standalone test on any image):
    python ml/scripts/super_resolve.py --image path/to/crop.jpg
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

ML_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ML_DIR / "models" / "realesrgan" / "weights"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

_upsampler = None  # lazy-loaded singleton so repeated calls don't reload the model


def load_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        print("Downloading RealESRGAN_x4plus.pth (first run only)...")
        model_path = load_file_from_url(
            url=WEIGHTS_URL, model_dir=str(WEIGHTS_DIR), progress=True, file_name=None
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    use_half = torch.cuda.is_available()  # fp16 only makes sense on GPU

    _upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        # tile=0 (process the whole image in one pass) works fine for small
        # plate crops, but on a 6GB GPU, an unusually large crop (e.g. a
        # wide box from a high-res source photo) can OOM after 4x upscaling
        # in a single forward pass. Tiling processes the image in chunks
        # instead -- slightly slower, but safe regardless of input size.
        tile=200,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
    )
    print(f"Real-ESRGAN loaded (device: {'cuda' if use_half else 'cpu'})")
    return _upsampler


def upscale_image(img_bgr, outscale=4):
    """img_bgr: numpy array (H, W, 3) BGR uint8, as returned by cv2.imread.
    Returns: numpy array (H*outscale, W*outscale, 3) BGR uint8."""
    upsampler = load_upsampler()
    output, _ = upsampler.enhance(img_bgr, outscale=outscale)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--outscale", type=int, default=4)
    parser.add_argument("--out", default=None, help="output path (default: <image>_upscaled.jpg)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    print(f"Input size: {img.shape[1]}x{img.shape[0]}")
    output = upscale_image(img, outscale=args.outscale)
    print(f"Output size: {output.shape[1]}x{output.shape[0]}")

    out_path = args.out or str(Path(args.image).with_stem(Path(args.image).stem + "_upscaled"))
    cv2.imwrite(out_path, output)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()