# ANPR in Adverse Weather Conditions

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Automatic Number Plate Recognition (ANPR) pipeline that handles adverse weather conditions (**haze, rain, blur, low-light**) using an intelligent multi-stage AI decision engine.

---

## Architecture

```
                       Input Image
                            │
                            ▼
                ┌───────────────────────┐
                │  Phase 3: Quality     │  ← EfficientNet-B0 classifier
                │  Analyzer             │     (clear / haze / rain / blur / lowlight)
                └───────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            │ If degradation detected       │ If clear / confident
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  Restoration Pipeline   │     │                         │
│  - CLAHE (Lowlight)     │     │                         │
│  - UNet Dehaze          │     │                         │
│  - UNet Derain          │     │                         │
│  - UNet Deblur          │     │                         │
└───────────┬─────────────┘     │                         │
            │                   │                         │
            └───────────┬───────┘                         │
                        ▼                                 │
            ┌─────────────────────────┐                   │
            │  Phase 5: YOLOv11       │  ← Fine-tuned yolo11n detector
            │  Plate Detection        │     (auto confidence scaling)
            └───────────┬─────────────┘
                        │ per detected plate crop
                        ▼
            ┌─────────────────────────┐
            │  Phase 6: Super-Res     │  ← Real-ESRGAN 4×
            │  (Conditional/Dual-Eval)│     (<150px upscale, 150-200px dual test)
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Phase 7: PaddleOCR     │  ← PP-OCR via onnxruntime backend
            │  Text Candidate Extraction│
            └───────────┬─────────────┘
                        │
                        ▼
            ┌─────────────────────────┐
            │  Phase 8: Validation    │  ← Indian plate format scoring
            │  & Character Correction │     + confusion repair (O↔0, I↔1)
            └─────────────────────────┘
```

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/loysan-mendes/anpr-adverse-weather.git
cd anpr-adverse-weather/ml

# Create and activate virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: For GPU support, install PyTorch with CUDA and PaddlePaddle before running `pip install -r requirements.txt`. See instructions in `ml/requirements.txt`.

---

### 2. Obtain Model Weights

#### Option A: One-Command Download (Recommended)
Download all pre-trained weights directly from the [GitHub Release](https://github.com/loysan-mendes/anpr-adverse-weather/releases):

```powershell
# From ml/ directory
python scripts/download_weights.py
```
This automatically downloads `models-v1.0.0.zip` and extracts the trained checkpoints into `ml/models/`:
* `ml/models/quality_analyzer/best_model.pt` (Quality Analyzer)
* `ml/models/restoration/blur/best_model.pt` (Deblur UNet)
* `ml/models/restoration/haze/best_model.pt` (Dehaze UNet)
* `ml/models/restoration/rain/best_model.pt` (Derain UNet)
* `ml/models/yolo_runs/plate_detector/weights/best.pt` (YOLOv11 Plate Detector)

*(Real-ESRGAN weights download automatically on first inference).*

#### Option B: Train from Scratch
To train the entire pipeline on your own data:

```powershell
# 1. Dataset download & weather augmentations
python scripts/download_dataset.py
python scripts/weather_augment.py
python scripts/build_manifest.py

# 2. Train Quality Analyzer (EfficientNet-B0)
python scripts/train_quality_analyzer.py --epochs 15

# 3. Train Restoration UNets (Residual mode)
python scripts/train_restoration.py --condition haze --epochs 30 --residual
python scripts/train_restoration.py --condition rain --epochs 30 --residual
python scripts/train_restoration.py --condition blur --epochs 30 --edge_weight 2.0 --residual

# Optional: Export TorchScript for 20-40% faster CPU inference
python scripts/export_unet.py

# 4. Train YOLOv11 Plate Detector
python scripts/prepare_yolo_dataset.py
python scripts/train_yolo.py --epochs 100
```

---

### 3. Run Inference

```powershell
# End-to-end pipeline on any photo
python scripts/decision_engine.py --image path/to/vehicle.jpg

# Test individual pipeline stages
python scripts/predict_quality.py --image path/to/vehicle.jpg
python scripts/ocr_plate.py --image path/to/plate_crop.jpg
python scripts/plate_validator.py --candidates "DL3CD1210"
python scripts/super_resolve.py --image path/to/plate_crop.jpg
```

### Example Pipeline Output

```json
{
  "image": "adverse_weather_car.jpg",
  "quality": {
    "condition": "blur",
    "condition_confidence": 0.606,
    "severity": "none",
    "severity_confidence": 0.439
  },
  "restoration_applied": "blur",
  "restoration_reason": "condition_confidence_fallback",
  "num_plates_detected": 1,
  "plates": [
    {
      "box": [144, 426, 446, 681],
      "detection_confidence": 0.909,
      "crop_width_px": 302,
      "upscaled": false,
      "plate_text": "AP29D4337",
      "validation_score": 0.92,
      "raw_ocr_regions": [
        {"text": "AP29D4337", "confidence": 0.94}
      ]
    }
  ]
}
```

---

## Project Structure

```
anpr-adverse-weather/
├── ml/
│   ├── scripts/
│   │   ├── decision_engine.py          # Full pipeline decision engine (Phase 9)
│   │   ├── download_weights.py         # Helper to fetch pre-trained weights from GitHub Releases
│   │   ├── train_quality_analyzer.py   # Phase 3: EfficientNet-B0 classifier
│   │   ├── unet_model.py               # Phase 4: UNet & Residual UNet architecture
│   │   ├── train_restoration.py        # Phase 4: Dehaze/Derain/Deblur training
│   │   ├── export_unet.py              # TorchScript compilation for fast CPU inference
│   │   ├── train_yolo.py               # Phase 5: YOLOv11 fine-tuning
│   │   ├── super_resolve.py            # Phase 6: Real-ESRGAN inference
│   │   ├── ocr_plate.py                # Phase 7: PaddleOCR wrapper
│   │   ├── plate_validator.py          # Phase 8: Indian plate format validator
│   │   ├── predict_quality.py          # Quality analyzer inference
│   │   ├── download_dataset.py         # Phase 1: dataset acquisition
│   │   ├── weather_augment.py          # Phase 2: synthetic degradation generator
│   │   ├── build_manifest.py           # Phase 2: manifest builder
│   │   ├── prepare_yolo_dataset.py     # Phase 5: YOLO dataset preparation
│   │   ├── evaluate_ocr.py             # Phase 7: OCR evaluation
│   │   ├── detect_and_upscale.py       # Phase 6: detection + upscale pipeline
│   │   ├── eda.py                      # Phase 1: exploratory data analysis
│   │   └── fix_basicsr_compat.py       # One-time compatibility patch for basicsr
│   ├── models/
│   │   ├── quality_analyzer/           # EfficientNet-B0 checkpoints
│   │   ├── restoration/                # UNet checkpoints per condition (blur/haze/rain)
│   │   ├── yolo_runs/                  # YOLO training checkpoints
│   │   └── realesrgan/                 # Real-ESRGAN weights (cached automatically)
│   ├── data/
│   │   ├── raw/                        # Raw dataset (downloaded locally via script)
│   │   ├── processed/                  # Manifests & EDA reports
│   │   └── yolo_dataset/              # YOLO splits (regenerated locally)
│   └── requirements.txt
├── .gitattributes                      # Git LFS config
└── .gitignore
```

---

## Key Design Decisions & Optimizations

| Feature / Decision | Rationale |
|:---|:---|
| **EfficientNet-B0 Quality Classifier** | Pretrained ImageNet representations with 5.3M parameters; strong generalisation without overfitting small datasets. |
| **Decoupled Dual Heads** | Separate condition and severity heads allow flexible decision thresholds. |
| **Confidence Fallback (≥55%)** | When severe degradation causes the severity head to predict "none", a confident condition triggers restoration automatically. |
| **CLAHE Lowlight Enhancement** | Equalises only the L channel in LAB color space to boost illumination without colour shift artefacts or neural network overhead. |
| **Residual UNet Architecture** | Learns `Output = Input + Residual`, accelerating training convergence and preserving fine plate boundaries. |
| **Dual-Scale OCR Validation (150–200px)** | In the borderline resolution zone, tests both raw and 4× upscaled crops, choosing the candidate with the highest syntactic validation score. |
| **PaddleOCR via ONNX Runtime** | Avoids native oneDNN incompatibilities on PP-OCR and ensures reliable CPU/GPU execution. |
| **Indian Plate Syntactic Validator** | Applies regex-based penalty scoring and character confusion repair (e.g. `O` vs `0`, `I` vs `1`) tailored to Indian license plates. |
| **TorchScript Optimization** | JIT compiles UNets into optimised graph representations for 20–40% faster inference on CPU. |

---

## License

This project is licensed under the [MIT License](LICENSE).
