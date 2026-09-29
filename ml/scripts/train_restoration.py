"""
Phase 4: Train a restoration model (Dehaze, Derain, or Deblur) using
synthetic (degraded, clean) pairs from Phase 2's augmented_manifest.json.

Every degraded image has a known-clean counterpart (the source photo
before we synthetically degraded it) -- that pairing is exactly what
image restoration needs, and we already have it for free.

Run once per condition:
    python ml/scripts/train_restoration.py --condition haze
    python ml/scripts/train_restoration.py --condition rain
    python ml/scripts/train_restoration.py --condition blur

(lowlight has no dedicated restoration branch in the UNet training --
 the Quality Analyzer flags it and decision_engine.py handles it with
 a fast CLAHE enhancer instead of a UNet. 3d change.)

3a: IMG_SIZE raised from 256 -> 512 to close the mismatch between training
    resolution and inference reality (images can be 1280px wide).
    Requires >= 8 GB VRAM at batch_size=8; drop to batch_size=4 on smaller GPUs.

3b: RestorationLoss now blends L1 + SSIM + Sobel-gradient. SSIM correlates
    better with perceived sharpness and contrast than per-pixel L1 alone.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from unet_model import UNet
from image_ops import aligned_patch
from training_config import restoration_parser, ssim_function
from dataset_utils import load_manifest, split_entries, image_path, source_id, write_provenance

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "augmented_manifest.json"
MODEL_ROOT = ML_DIR / "models" / "restoration"
IMG_SIZE = 512  # 3a: was 256; raised to reduce the train/inference resolution mismatch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_pairs(manifest_path: Path, condition: str, val_frac=0.15, test_frac=0.15):
    splits = split_entries(load_manifest(manifest_path), val_frac, test_frac)
    result = []
    for split in ("train", "val"):
        groups = defaultdict(dict)
        for entry in splits[split]:
            group = groups[source_id(entry)]
            if entry["condition"] == "clear":
                group["clear"] = entry["image_path"]
            elif entry["condition"] == condition:
                group.setdefault("degraded", []).append(entry)
        pairs = []
        for key, group in groups.items():
            if group.get("degraded") and "clear" not in group:
                raise ValueError(f"Missing clean counterpart: {key}")
            pairs.extend({"degraded": entry["image_path"], "clean": group["clear"], "boxes": entry["boxes"]}
                         for entry in group.get("degraded", []))
        if not pairs:
            raise ValueError(f"No {condition} pairs in {split}")
        result.append(pairs)
    return tuple(result)


class PairedDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, rel_path):
        img = cv2.imread(str(image_path({"image_path": rel_path}, ML_DIR)))
        if img is None:
            raise FileNotFoundError(rel_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        deg = self._load(pair["degraded"])
        clean = self._load(pair["clean"])
        deg, clean = aligned_patch(deg, clean, self.img_size, pair["boxes"], random if self.augment else None)
        # Teach the restorer to leave already-clear character strokes alone.
        if self.augment and random.random() < 0.1:
            deg = clean.copy()

        deg_t = torch.from_numpy(deg.astype(np.float32) / 255.0).permute(2, 0, 1)
        clean_t = torch.from_numpy(clean.astype(np.float32) / 255.0).permute(2, 0, 1)
        return deg_t, clean_t


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * np.log10(1.0 / mse)


class SobelGradLoss(nn.Module):
    """Penalizes differences in edge/gradient content between pred and
    target, on top of plain pixel loss. Plain L1 alone tends to make a
    deblur network hedge toward a smoothed 'average' output (safest bet
    to minimize per-pixel error across many blur directions/magnitudes),
    which is exactly the 'still blurry' failure mode we saw. Forcing the
    gradients to match pushes the network to commit to actual sharp
    edges instead of hedging."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t()
        # depthwise conv: one (1,3,3) kernel per input channel, groups=3
        self.register_buffer("weight_x", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("weight_y", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        gx_p = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        gy_p = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        gx_t = F.conv2d(target, self.weight_x, padding=1, groups=3)
        gy_t = F.conv2d(target, self.weight_y, padding=1, groups=3)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class RestorationLoss(nn.Module):
    """3b: L1 pixel loss + SSIM perceptual loss + weighted Sobel-gradient sharpness loss.

    SSIM (Structural Similarity) measures luminance, contrast, and structure
    simultaneously -- it correlates better with how humans perceive sharpness
    and contrast restoration than per-pixel L1 alone. Blended at 0.3 weight
    so it guides without dominating the pixel-accurate L1 signal.

    SSIM is required when its weight is positive; use --ssim_weight 0 to disable explicitly.
    """

    def __init__(self, edge_weight=1.0, ssim_weight=0.3):
        super().__init__()
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.grad = SobelGradLoss()
        if edge_weight < 0 or ssim_weight < 0:
            raise ValueError("Loss weights must be nonnegative")
        self._ssim_fn = ssim_function(ssim_weight)

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        if self.ssim_weight > 0 and self._ssim_fn is not None:
            # 3b: SSIM loss: 1 - SSIM so minimising it maximises similarity.
            ssim_loss = 1.0 - self._ssim_fn(pred, target, data_range=1.0)
            loss = loss + self.ssim_weight * ssim_loss
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.grad(pred, target)
        return loss


def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train() if train else model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for deg, clean in loader:
            deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                pred = model(deg)
            loss = criterion(pred.float(), clean)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = deg.size(0)
            total_loss += loss.item() * bs
            total_psnr += psnr(pred.detach(), clean) * bs
            n += bs

    return total_loss / n, total_psnr / n


def save_preview_grid(model, val_pairs, device, out_path, n=4):
    model.eval()
    ds = PairedDataset(val_pairs[:n], augment=False)
    fig, axes = plt.subplots(min(n, len(ds)), 3, figsize=(9, 3 * min(n, len(ds))))
    if len(ds) == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            deg, clean = ds[i]
            pred = model(deg.unsqueeze(0).to(device))[0].cpu()
            for ax, img, title in zip(
                axes[i], [deg, pred, clean], ["degraded (input)", "restored (output)", "clean (target)"]
            ):
                ax.imshow(img.permute(1, 2, 0).numpy())
                ax.set_title(title, fontsize=9)
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = restoration_parser(MANIFEST_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Loss: L1 + {args.ssim_weight}*SSIM + {args.edge_weight}*Sobel-gradient")

    train_pairs, val_pairs = build_pairs(Path(args.manifest), args.condition, args.val_frac, args.test_frac)
    train_ds = PairedDataset(train_pairs, augment=True)
    val_ds = PairedDataset(val_pairs, augment=False)
    loader_options = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = UNet(base=32, residual=args.residual).to(device)
    criterion = RestorationLoss(edge_weight=args.edge_weight, ssim_weight=args.ssim_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_dir = Path(args.out_dir) if args.out_dir else MODEL_ROOT / args.condition
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}
    best_val_psnr = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_psnr = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=scaler)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(tr_psnr)
        history["val_psnr"].append(val_psnr)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} psnr {tr_psnr:.2f} | "
              f"val loss {val_loss:.4f} psnr {val_psnr:.2f}")

        if val_psnr > best_val_psnr + 1e-4:
            best_val_psnr = val_psnr
            stale_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "condition": args.condition,
                "architecture": {"base": 32, "residual": args.residual},
                "loss_config": {"l1": 1.0, "ssim": args.ssim_weight, "sobel": args.edge_weight},
                "img_size": IMG_SIZE,
                "training_sampling": "native_plate_focused_patches_with_10_percent_identity",
                "amp": scaler.is_enabled(),
                "epoch": epoch,
                "val_psnr": float(val_psnr),
            }, model_dir / "best_model.pt")
            splits = split_entries(load_manifest(args.manifest), args.val_frac, args.test_frac)
            write_provenance(model_dir / "best_model.pt", splits["train"] + splits["val"], ML_DIR)
            print(f"  -> saved new best checkpoint (val_psnr={val_psnr:.2f} dB)")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation improvement")
                break

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{args.condition}: L1 Loss"); axes[0].legend()
    axes[1].plot(history["train_psnr"], label="train"); axes[1].plot(history["val_psnr"], label="val")
    axes[1].set_title(f"{args.condition}: PSNR (dB)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png", dpi=120)

    # reload best checkpoint for the preview grid
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    save_preview_grid(model, val_pairs, device, model_dir / "restoration_preview.png")

    print(f"\nBest val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoint -> {model_dir / 'best_model.pt'}")
    print(f"Preview grid -> {model_dir / 'restoration_preview.png'}")


if __name__ == "__main__":
    main()
