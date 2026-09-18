# ANPR in Adverse Weather Conditions

An end-to-end Automatic Number Plate Recognition (ANPR) pipeline that handles adverse weather conditions (haze, rain, blur, low-light) using a multi-stage AI decision engine.

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Phase 3: Quality Analyzer  │  ← EfficientNet-B0 classifier
│  condition + severity       │     (clear / haze / rain / blur / lowlight)
└────────────┬────────────────┘
             │ if condition detected with sufficient confidence
             ▼
┌─────────────────────────────┐
│  Phase 4: Restoration       │  ← Custom UNet (3-level)
│  Dehaze / Derain / Deblur   │     trained on synthetic paired data
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 5: YOLOv11 Detection │  ← Fine-tuned yolo11n on plate dataset
│  plate bounding boxes       │
└────────────┬────────────────┘
             │ per detected crop
             ▼
┌─────────────────────────────┐
│  Phase 6: Super-Resolution  │  ← Real-ESRGAN 4× (only on narrow crops)
│  (conditional, if crop <150px wide)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 7: PaddleOCR         │  ← PP-OCRv5 via onnxruntime backend
│  text candidates            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 8: Plate Validation  │  ← Indian plate format scoring
│  best plate text            │     + character confusion correction
└─────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

> **Prerequisites**: Install PyTorch with CUDA support and PaddlePaddle **before** running pip install (see comments in `ml/requirements.txt`).

```powershell
cd ml
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Obtain Model Weights

#### Option A: Quick Start (Download Pre-trained Weights)
To run inference immediately without retraining, download the trained model weights from the [GitHub Release](https://github.com/loysan-mendes/anpr-adverse-weather/releases):
```powershell
# From ml/ directory
python scripts/download_weights.py
```
This automatically fetches and places all model checkpoints into `ml/models/`.

#### Option B: Train Models from Scratch
If you prefer to train your own models:
```powershell
# Phase 1 & 2: Dataset download & weather augmentations
python scripts/download_dataset.py
python scripts/weather_augment.py
python scripts/build_manifest.py

# Phase 3: Quality Analyzer
python scripts/train_quality_analyzer.py --epochs 15

# Phase 4: Restoration models (run once per condition)
python scripts/train_restoration.py --condition haze --epochs 30
python scripts/train_restoration.py --condition rain --epochs 30
python scripts/train_restoration.py --condition blur --epochs 30 --edge_weight 2.0

# Phase 5: YOLO plate detector
python scripts/prepare_yolo_dataset.py
python scripts/train_yolo.py --epochs 100

# Phase 6: Real-ESRGAN weights download automatically on first use
```

### 4. Run inference

```powershell
# Full pipeline on a single image
python scripts/decision_engine.py --image path/to/photo.jpg

# Individual stage testing
python scripts/predict_quality.py --image path/to/photo.jpg
python scripts/ocr_plate.py --image path/to/plate_crop.jpg
python scripts/plate_validator.py --candidates "DL3CD1210"
python scripts/super_resolve.py --image path/to/crop.jpg
```

### Example output

```json
{
  "image": "photo.jpg",
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
      "raw_ocr_regions": [...]
    }
  ]
}
```

## Project Structure

```
anpr-adverse-weather/
├── ml/
│   ├── scripts/
│   │   ├── decision_engine.py          # Full pipeline integration (Phase 9)
│   │   ├── train_quality_analyzer.py   # Phase 3: EfficientNet-B0 classifier
│   │   ├── unet_model.py               # Phase 4: UNet architecture
│   │   ├── train_restoration.py        # Phase 4: Dehaze/Derain/Deblur training
│   │   ├── train_yolo.py               # Phase 5: YOLOv11 fine-tuning
│   │   ├── super_resolve.py            # Phase 6: Real-ESRGAN inference
│   │   ├── ocr_plate.py                # Phase 7: PaddleOCR wrapper
│   │   ├── plate_validator.py          # Phase 8: Indian plate format validator
│   │   ├── predict_quality.py          # Quality analyzer inference
│   │   ├── download_dataset.py         # Phase 1: dataset acquisition
│   │   ├── weather_augment.py          # Phase 2: synthetic degradation
│   │   ├── build_manifest.py           # Phase 2: manifest builder
│   │   ├── prepare_yolo_dataset.py     # Phase 5: YOLO dataset preparation
│   │   ├── evaluate_ocr.py             # Phase 7: OCR evaluation
│   │   ├── detect_and_upscale.py       # Phase 6: detection + upscale pipeline
│   │   ├── eda.py                      # Phase 1: exploratory data analysis
│   │   └── fix_basicsr_compat.py       # One-time compatibility patch for basicsr
│   ├── models/
│   │   ├── quality_analyzer/           # EfficientNet-B0 checkpoint (Git LFS)
│   │   ├── restoration/                # UNet checkpoints per condition (Git LFS)
│   │   │   ├── blur/
│   │   │   ├── haze/
│   │   │   └── rain/
│   │   ├── yolo_runs/                  # YOLO training runs (Git LFS)
│   │   └── realesrgan/                 # Real-ESRGAN weights (downloaded at runtime)
│   ├── data/
│   │   ├── raw/                        # Raw dataset (NOT committed, use download_dataset.py)
│   │   ├── processed/                  # Manifests committed; images NOT committed
│   │   └── yolo_dataset/              # YOLO splits (NOT committed, regenerated)
│   └── requirements.txt
├── backend/                            # (planned)
├── frontend/                           # (planned)
├── docker/                             # (planned)
├── .gitattributes                      # Git LFS config for .pt model weights
└── .gitignore
```

## Key Design Decisions

| Decision | Rationale |
|:---|:---|
| **EfficientNet-B0 as quality classifier** | Pretrained ImageNet features; 5.3M params — enough expressivity without overfitting 47-photo dataset |
| **Separate condition + severity heads** | Decoupled classification; condition routes the pipeline, severity influences confidence threshold |
| **Condition-confidence fallback (0.55 threshold)** | Severity head occasionally misfires as "none"; if condition confidence is ≥55%, restoration runs anyway |
| **UNet at 256×256 working resolution** | Memory/speed tradeoff for small training set; model sees full spatial context at the cost of fine detail |
| **Real-ESRGAN only for narrow crops (<150px)** | Blanket upscaling hurt OCR accuracy (GAN hallucinations confuse OCR on adequately-sized crops) |
| **PaddleOCR via onnxruntime backend** | Paddle's native oneDNN execution hit an unresolved bug on PP-OCRv5/v6; ONNX backend sidesteps it |
| **Plate validator scores by template fit** | Indian plate structure (LLDDLLDDDD etc.) gives strong prior; handles common OCR char confusions (O↔0, I↔1) |

## Known Limitations

- Training dataset: 47 unique source photos — a proof-of-concept, not production-grade
- Restoration models work at 256px working resolution — fine detail lost on high-res inputs
- No dedicated low-light restoration model (Quality Analyzer flags it, pipeline does nothing yet)
- Plate validator assumes Indian plate formats; international plates will score low

## License

MIT
