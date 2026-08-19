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

(lowlight has no dedicated restoration branch in the architecture --
 the Quality Analyzer still flags it, but only haze/rain/blur route
 through a restoration model before detection.)
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

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "augmented_manifest.json"
MODEL_ROOT = ML_DIR / "models" / "restoration"
IMG_SIZE = 256

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def group_key(image_path: str) -> str:
    return Path(image_path).stem


def build_pairs(manifest_path: Path, condition: str, val_frac=0.15):
    with open(manifest_path) as f:
        entries = json.load(f)

    groups = defaultdict(dict)
    for e in entries:
        stem = group_key(e["image_path"])
        if e["condition"] == "clear":
            groups[stem]["clear"] = e["image_path"]
        elif e["condition"] == condition:
            groups[stem].setdefault("degraded", []).append(e["image_path"])

    stems = [s for s, d in groups.items() if "clear" in d and d.get("degraded")]
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * val_frac))
    val_stems = set(stems[:n_val])

    train_pairs, val_pairs = [], []
    for stem in stems:
        clear_path = groups[stem]["clear"]
        for deg_path in groups[stem]["degraded"]:
            (val_pairs if stem in val_stems else train_pairs).append((deg_path, clear_path))

    print(f"[{condition}] {len(stems)} source photos with this condition -> "
          f"{len(stems) - n_val} train photos ({len(train_pairs)} pairs), "
          f"{n_val} val photos ({len(val_pairs)} pairs)")
    return train_pairs, val_pairs


class PairedDataset(Dataset):
    def __init__(self, pairs, img_size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _load(self, rel_path):
        img = cv2.imread(str(ML_DIR / rel_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg = self._load(deg_path)
        clean = self._load(clean_path)

        if self.augment and random.random() < 0.5:
            deg = np.ascontiguousarray(deg[:, ::-1, :])
            clean = np.ascontiguousarray(clean[:, ::-1, :])

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
    """L1 pixel loss + weighted Sobel-gradient sharpness loss."""

    def __init__(self, edge_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
        self.l1 = nn.L1Loss()
        self.grad = SobelGradLoss()

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        if self.edge_weight > 0:
            loss = loss + self.edge_weight * self.grad(pred, target)
        return loss


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for deg, clean in loader:
            deg, clean = deg.to(device), clean.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(deg)
            loss = criterion(pred, clean)
            if train:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["haze", "rain", "blur"])
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--edge_weight", type=float, default=1.0,
                         help="Weight for the Sobel-gradient sharpness loss on top of L1. "
                              "0 = old behavior (L1 only). Higher values push harder for sharp "
                              "edges -- use a higher value (e.g. 2.0-3.0) for --condition blur, "
                              "which is the one most prone to hedging toward a smoothed output.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loss: L1 + {args.edge_weight} x Sobel-gradient (sharpness) loss")

    train_pairs, val_pairs = build_pairs(Path(args.manifest), args.condition, args.val_frac)
    train_ds = PairedDataset(train_pairs, augment=True)
    val_ds = PairedDataset(val_pairs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(base=32).to(device)
    criterion = RestorationLoss(edge_weight=args.edge_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_dir = MODEL_ROOT / args.condition
    model_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}
    best_val_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_psnr = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_psnr"].append(tr_psnr)
        history["val_psnr"].append(val_psnr)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} psnr {tr_psnr:.2f} | "
              f"val loss {val_loss:.4f} psnr {val_psnr:.2f}")

        if val_psnr >= best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                "model_state": model.state_dict(),
                "condition": args.condition,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_psnr": val_psnr,
            }, model_dir / "best_model.pt")
            print(f"  -> saved new best checkpoint (val_psnr={val_psnr:.2f} dB)")

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