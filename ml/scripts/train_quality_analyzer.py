"""
Phase 3: Quality Analyzer -- EfficientNet-B0 classifier that predicts:
    1. condition:  clear | haze | rain | blur | lowlight
    2. severity:   none | mild | moderate | severe

This is the "AI Decision Engine" gatekeeper from the architecture: given an
incoming plate image, it decides WHICH restoration model(s) to route through
(Dehaze / Derain / Deblur) and how aggressively, before detection.

Trains on ml/data/processed/augmented_manifest.json (built in Phase 2).

IMPORTANT CAVEAT: we only have 47 unique source photographs. Augmentation
multiplies that to ~611 images, but they're still 47 underlying scenes. We
split train/val BY SOURCE PHOTO (not by individual augmented image) so a
photo's clear/hazy/rainy/blurry variants never appear on both sides of the
split -- this gives an honest (if noisy, given the small photo count) signal
on whether the model generalizes to unseen scenes, not just memorizes them.
Treat this as a working proof-of-concept; revisit with more diverse source
images (e.g. a public real-world hazy/rainy dataset) once available.

Usage:
    python ml/scripts/train_quality_analyzer.py --epochs 15
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt

ML_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ML_DIR / "data" / "processed" / "augmented_manifest.json"
MODEL_DIR = ML_DIR / "models" / "quality_analyzer"

CONDITIONS = ["clear", "haze", "rain", "blur", "lowlight"]
SEVERITIES = ["none", "mild", "moderate", "severe"]
COND2IDX = {c: i for i, c in enumerate(CONDITIONS)}
SEV2IDX = {s: i for i, s in enumerate(SEVERITIES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------- dataset ---
def group_key(image_path: str) -> str:
    """Every condition/severity variant of the same source photo shares
    the same output filename stem (see weather_augment.py) -- use that as
    the grouping key so we split by source photo, not by augmented image."""
    return Path(image_path).stem


def load_manifest_and_split(manifest_path: Path, val_frac=0.15):
    with open(manifest_path) as f:
        entries = json.load(f)

    groups = defaultdict(list)
    for e in entries:
        groups[group_key(e["image_path"])].append(e)

    keys = list(groups.keys())
    random.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])

    train_entries, val_entries = [], []
    for k, items in groups.items():
        (val_entries if k in val_keys else train_entries).extend(items)

    print(f"{len(groups)} unique source photos -> "
          f"{len(keys) - n_val} train photos ({len(train_entries)} images), "
          f"{n_val} val photos ({len(val_entries)} images)")
    return train_entries, val_entries


class QualityDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img_path = ML_DIR / e["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        cond_idx = COND2IDX[e["condition"]]
        sev_idx = SEV2IDX[e["severity"]]
        return img, cond_idx, sev_idx


def make_transforms():
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def class_weights(entries, key, classes):
    counts = np.zeros(len(classes))
    idx_of = {c: i for i, c in enumerate(classes)}
    for e in entries:
        counts[idx_of[e[key]]] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------- model ---
class QualityNet(nn.Module):
    def __init__(self, num_conditions, num_severities, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features  # 1280
        self.dropout = nn.Dropout(0.3)
        self.condition_head = nn.Linear(in_features, num_conditions)
        self.severity_head = nn.Linear(in_features, num_severities)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.condition_head(x), self.severity_head(x)


# --------------------------------------------------------------- training ---
def run_epoch(model, loader, cond_criterion, sev_criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, cond_correct, sev_correct, n = 0.0, 0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, cond_labels, sev_labels in loader:
            imgs = imgs.to(device)
            cond_labels = cond_labels.to(device)
            sev_labels = sev_labels.to(device)

            if train:
                optimizer.zero_grad()

            cond_logits, sev_logits = model(imgs)
            loss = cond_criterion(cond_logits, cond_labels) + 0.5 * sev_criterion(sev_logits, sev_labels)

            if train:
                loss.backward()
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            cond_correct += (cond_logits.argmax(1) == cond_labels).sum().item()
            sev_correct += (sev_logits.argmax(1) == sev_labels).sum().item()
            n += bs

    return total_loss / n, cond_correct / n, sev_correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_entries, val_entries = load_manifest_and_split(Path(args.manifest), args.val_frac)
    train_tf, val_tf = make_transforms()
    train_ds = QualityDataset(train_entries, train_tf)
    val_ds = QualityDataset(val_entries, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = QualityNet(len(CONDITIONS), len(SEVERITIES)).to(device)

    cond_w = class_weights(train_entries, "condition", CONDITIONS).to(device)
    sev_w = class_weights(train_entries, "severity", SEVERITIES).to(device)
    cond_criterion = nn.CrossEntropyLoss(weight=cond_w)
    sev_criterion = nn.CrossEntropyLoss(weight=sev_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_cond_acc": [], "val_cond_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cond_acc, tr_sev_acc = run_epoch(
            model, train_loader, cond_criterion, sev_criterion, optimizer, device, train=True)
        val_loss, val_cond_acc, val_sev_acc = run_epoch(
            model, val_loader, cond_criterion, sev_criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_cond_acc"].append(tr_cond_acc)
        history["val_cond_acc"].append(val_cond_acc)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} cond_acc {tr_cond_acc:.3f} sev_acc {tr_sev_acc:.3f} | "
              f"val loss {val_loss:.3f} cond_acc {val_cond_acc:.3f} sev_acc {val_sev_acc:.3f}")

        if val_cond_acc >= best_val_acc:
            best_val_acc = val_cond_acc
            torch.save({
                "model_state": model.state_dict(),
                "conditions": CONDITIONS,
                "severities": SEVERITIES,
                "img_size": 224,
                "imagenet_mean": IMAGENET_MEAN,
                "imagenet_std": IMAGENET_STD,
                "epoch": epoch,
                "val_cond_acc": val_cond_acc,
            }, MODEL_DIR / "best_model.pt")
            print(f"  -> saved new best checkpoint (val_cond_acc={val_cond_acc:.3f})")

    with open(MODEL_DIR / "label_map.json", "w") as f:
        json.dump({"conditions": CONDITIONS, "severities": SEVERITIES}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_cond_acc"], label="train")
    axes[1].plot(history["val_cond_acc"], label="val")
    axes[1].set_title("Condition Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_curves.png", dpi=120)

    print(f"\nBest val condition accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved -> {MODEL_DIR / 'best_model.pt'}")
    print(f"Training curves -> {MODEL_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()