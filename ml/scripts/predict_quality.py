"""
Phase 3: Run the trained Quality Analyzer on a single image.

Usage:
    python ml/scripts/predict_quality.py --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from train_quality_analyzer import QualityNet, IMAGENET_MEAN, IMAGENET_STD

ML_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ML_DIR / "models" / "quality_analyzer" / "best_model.pt"


def load_model(checkpoint_path=CHECKPOINT_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: safe here since this checkpoint is one we trained
    # ourselves (it also stores plain lists/ints alongside tensors, which
    # weights_only=True would reject).
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = QualityNet(len(ckpt["conditions"]), len(ckpt["severities"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt, device


def predict(image_path, model=None, ckpt=None, device=None):
    if model is None:
        model, ckpt, device = load_model()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["imagenet_mean"], ckpt["imagenet_std"]),
    ])
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cond_logits, sev_logits = model(x)
        cond_probs = F.softmax(cond_logits, dim=1)[0].cpu().numpy()
        sev_probs = F.softmax(sev_logits, dim=1)[0].cpu().numpy()

    cond_idx = cond_probs.argmax()
    sev_idx = sev_probs.argmax()

    result = {
        "condition": ckpt["conditions"][cond_idx],
        "condition_confidence": float(cond_probs[cond_idx]),
        "condition_probs": {c: float(p) for c, p in zip(ckpt["conditions"], cond_probs)},
        "severity": ckpt["severities"][sev_idx],
        "severity_confidence": float(sev_probs[sev_idx]),
        "severity_probs": {s: float(p) for s, p in zip(ckpt["severities"], sev_probs)},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    model, ckpt, device = load_model(args.checkpoint)
    result = predict(args.image, model, ckpt, device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()