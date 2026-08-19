"""
Phase 2: Synthetic weather augmentation.

Takes the 47 clean, labeled images from data/processed/manifest.json and
generates degraded variants (haze / rain / blur / low-light), each at
multiple severities. Bounding boxes stay valid on every variant since we
only touch pixel values -- we do resize images (they're huge, up to
4608px) to a sane training resolution, scaling boxes to match.

Output:
    ml/data/processed/augmented/<condition>/<severity>/<original_filename>
    ml/data/processed/augmented_manifest.json   <- unified manifest for
                                                     every clear + degraded
                                                     image, with a
                                                     `condition` and
                                                     `severity` label on
                                                     each entry (this is
                                                     what the Quality
                                                     Analyzer, Phase 3,
                                                     trains on)
    ml/data/processed/augmentation_preview.png  <- sanity-check grid

Usage:
    python ml/scripts/weather_augment.py
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "manifest.json"
OUT_DIR = ML_DIR / "data" / "processed" / "augmented"
MAX_SIDE = 1280  # resize longest side to this before augmenting

SEVERITIES = ["mild", "moderate", "severe"]
SEVERITY_SCALE = {"mild": 0.4, "moderate": 0.7, "severe": 1.0}

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------- resize ---
def resize_with_boxes(img, boxes, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img, boxes, 1.0  # don't upscale
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_boxes = []
    for b in boxes:
        new_boxes.append({
            **b,
            "xmin": b["xmin"] * scale, "ymin": b["ymin"] * scale,
            "xmax": b["xmax"] * scale, "ymax": b["ymax"] * scale,
        })
    return resized, new_boxes, scale


# --------------------------------------------------------- degradations ---
def add_haze(img, severity):
    """Atmospheric scattering model: I = J*t + A*(1-t)."""
    strength = SEVERITY_SCALE[severity]
    h, w = img.shape[:2]
    img_f = img.astype(np.float32) / 255.0

    # depth proxy: distance from a random "vanishing point" (denser haze far away)
    yy, xx = np.mgrid[0:h, 0:w]
    vp_x, vp_y = w * random.uniform(0.3, 0.7), h * random.uniform(0.0, 0.3)
    dist = np.sqrt((xx - vp_x) ** 2 + (yy - vp_y) ** 2)
    dist = dist / dist.max()

    beta = 0.8 + strength * 1.5
    transmission = np.exp(-beta * dist)
    transmission = np.clip(transmission, 1 - 0.75 * strength, 1.0)[..., None]

    atmosphere = 0.85 + 0.1 * random.random()
    hazy = img_f * transmission + atmosphere * (1 - transmission)
    return np.clip(hazy * 255, 0, 255).astype(np.uint8)


def add_rain(img, severity):
    strength = SEVERITY_SCALE[severity]
    h, w = img.shape[:2]
    rain_layer = np.zeros((h, w), dtype=np.float32)

    n_drops = int((h * w) / 2200 * strength * 4)
    length_range = (12, 22) if strength < 0.7 else (18, 32)
    angle = random.uniform(-15, 15)  # slight wind tilt, degrees from vertical

    for _ in range(n_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(*length_range)
        dx = int(length * np.sin(np.radians(angle)))
        dy = int(length * np.cos(np.radians(angle)))
        cv2.line(rain_layer, (x, y), (x + dx, y + dy), 1.0, thickness=1)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_layer_3ch = np.stack([rain_layer] * 3, axis=-1)

    img_f = img.astype(np.float32) / 255.0
    # slight overall desaturation/fog under rain, then streaks on top
    img_f = img_f * (1 - 0.15 * strength) + 0.6 * (0.15 * strength)
    rainy = np.clip(img_f + rain_layer_3ch * 0.55, 0, 1)
    return (rainy * 255).astype(np.uint8)


def add_blur(img, severity):
    strength = SEVERITY_SCALE[severity]
    kind = random.choice(["motion", "gaussian"])
    if kind == "gaussian":
        k = int(3 + strength * 12)
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    else:
        k = int(5 + strength * 20)
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0
        kernel = kernel / k
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        return cv2.filter2D(img, -1, kernel)


def add_lowlight(img, severity):
    strength = SEVERITY_SCALE[severity]
    img_f = img.astype(np.float32) / 255.0

    gamma = 1.0 + strength * 2.5
    darkened = np.power(img_f, gamma)

    # sensor noise gets worse as light drops
    noise_sigma = 0.01 + strength * 0.04
    noise = np.random.normal(0, noise_sigma, img_f.shape)
    noisy = np.clip(darkened + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


DEGRADATIONS = {
    "haze": add_haze,
    "rain": add_rain,
    "blur": add_blur,
    "lowlight": add_lowlight,
}


# -------------------------------------------------------------- pipeline ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--severities", nargs="+", default=SEVERITIES)
    args = parser.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)

    out_dir = Path(args.out_dir)
    clear_dir = out_dir / "clear"
    clear_dir.mkdir(parents=True, exist_ok=True)

    augmented_manifest = []
    preview_samples = {}  # condition -> one example image for the sanity grid

    for entry in entries:
        img_path = ML_DIR / entry["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        img_resized, boxes_resized, scale = resize_with_boxes(img, entry["boxes"])
        h, w = img_resized.shape[:2]

        # save the clean/resized version too -- Quality Analyzer's "clear" class
        clear_name = Path(entry["image_path"]).stem + ".jpg"
        clear_path = clear_dir / clear_name
        cv2.imwrite(str(clear_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        augmented_manifest.append({
            "image_path": str(clear_path.relative_to(ML_DIR)),
            "width": w, "height": h,
            "boxes": boxes_resized,
            "condition": "clear",
            "severity": "none",
            "orig_source": entry["source"],
        })
        preview_samples.setdefault("clear", img_resized)

        for condition, fn in DEGRADATIONS.items():
            for severity in args.severities:
                degraded = fn(img_resized, severity)
                cond_dir = out_dir / condition / severity
                cond_dir.mkdir(parents=True, exist_ok=True)
                out_name = Path(entry["image_path"]).stem + ".jpg"
                out_path = cond_dir / out_name
                cv2.imwrite(str(out_path), degraded, [cv2.IMWRITE_JPEG_QUALITY, 95])

                augmented_manifest.append({
                    "image_path": str(out_path.relative_to(ML_DIR)),
                    "width": w, "height": h,
                    "boxes": boxes_resized,
                    "condition": condition,
                    "severity": severity,
                    "orig_source": entry["source"],
                })
                key = f"{condition}_{severity}"
                if key not in preview_samples:
                    preview_samples[key] = degraded

    manifest_out = ML_DIR / "data" / "processed" / "augmented_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(augmented_manifest, f, indent=2)

    print(f"Wrote {len(augmented_manifest)} entries -> {manifest_out}")
    print(f"({len(entries)} source images x (1 clear + "
          f"{len(DEGRADATIONS)} conditions x {len(args.severities)} severities))")

    # ---- sanity-check preview grid ----
    keys = list(preview_samples.keys())
    n = len(keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, key in zip(axes, keys):
        img_bgr = preview_samples[key]
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(key, fontsize=10)
        ax.axis("off")
    for ax in axes[len(keys):]:
        ax.axis("off")
    plt.tight_layout()
    preview_path = ML_DIR / "data" / "processed" / "augmentation_preview.png"
    plt.savefig(preview_path, dpi=120)
    print(f"Saved preview grid -> {preview_path}")


if __name__ == "__main__":
    main()