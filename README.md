# ANPR in Adverse Weather

A research prototype for Indian number-plate recognition from full photographs. The pipeline classifies image quality, optionally restores weather degradation, detects plates, runs OCR, and returns accepted or uncertain readings.

The available corpus contains **47 source photographs**, 611 synthetic variants, and 25 transcribed plate boxes. These counts do not establish real-weather generalization. Historical OCR results used ground-truth crops and are component diagnostics, not full-pipeline accuracy.

## Setup

### Web application

From the project root, start the local app with the configured ML environment:

```powershell
.\run_ml.ps1 backend/server.py
# With a regular activated environment: python backend/server.py
```

Open **http://127.0.0.1:8000**. Choose or drop a JPEG, PNG or WebP image (up to 12 MB and 12 megapixels), select Balanced or Thorough mode, then click **Analyze image**. The optional sample uses a locally installed project image. Click result cards to highlight plate boxes, expand reading evidence, or export the complete result as JSON. Uncertain readings are shown as proposals requiring review.

The app reuses the current active detector and quality/restoration checkpoints. Super-resolution is disabled to match the verified pipeline and avoid downloading missing Real-ESRGAN weights. The first request loads models; later requests reuse them. Images are processed locally, normalized for EXIF orientation, and temporary upload files are removed after analysis. There is no persistent upload history. Keep the server terminal open; Ctrl+C stops it. Use `--port 8001` if port 8000 is occupied.

The standard-library HTTP server binds to loopback only and is intended for local use. It does not provide accounts, TLS, or a public deployment service. One inference request runs at a time; another receives a retry message. Web tests: `.\run_ml.ps1 -m unittest discover -s backend` (requires Pillow, already in ML requirements). Frontend files need no Node installation or build step.

### Python environment

Use a working Python 3.11 installation. A virtual environment cannot be moved between machines or reused after its base interpreter is removed. Keep an old environment for reference and create a fresh `.venv` when necessary.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip

# CPU baseline; install a compatible torch/torchvision pair together.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install paddlepaddle==3.3.0
python -m pip install -r ml/requirements.txt
python ml/scripts/fix_basicsr_compat.py
python -m pip check
```

For NVIDIA training, replace the CPU PyTorch installation with the appropriate pair from the [official PyTorch installer](https://pytorch.org/get-started/locally/). The Python wheels and CUDA runtime must be compatible; the driver version alone does not determine the correct wheel. Do not mix packages from the old virtual environment into a different Python version.

`torchmetrics` is required by default for SSIM loss. BasicSR 1.4.2 needs the compatibility patch with newer torchvision; run it again after reinstalling BasicSR. PaddleOCR uses the ONNX Runtime backend on CPU. OCR and Real-ESRGAN may download their pretrained assets on first use. Loading the project's trained quality checkpoint does **not** download ImageNet weights.

Local verification now runs through `.runtime311/python.exe` (Python 3.11.9), reusing the existing CPython 3.11 packages, with PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130 and an RTX 3050 6GB GPU. `./run_ml.ps1 ml/scripts/decision_engine.py --image path/to/vehicle.jpg --no-upscale` selects this local runtime when present, otherwise `.venv`. The local runtime and caches are ignored by Git; other machines still need the setup above. Full-image inference and a small mixed-precision training smoke check passed. SSIM was explicitly disabled for that smoke check because the local runtime lacks torchmetrics; install the declared requirements before default training. A fresh full dependency installation has not been verified.

## Download weights and infer

### Vehicle classification

Analysis now runs a separate COCO-pretrained YOLO11n detector on the original full
image for **car, motorcycle, bus, and truck**. It uses `ml/yolo11n.pt` (already
present in this workspace). On a fresh installation, download it once before
analysis, with an internet connection:

```powershell
./run_ml.ps1 ml/scripts/vehicle_detector.py --download
```

Inference never downloads vehicle weights automatically. Missing weights fail
with setup instructions. CLI users can override the checkpoint using
`--vehicle-weights path/to/detector.pt`; required class names are checked and class
IDs are read from the checkpoint. An optional `other` class maps to `unknown`.
When present, `ml/models/active_vehicle_detector.json` selects a checkpoint by
path and SHA-256; changed/missing registered weights fail instead of silently
switching models. Without a registry, the pretrained nano model is used.
The vehicle detector is cached independently of the trained plate detector.

The app draws purple dashed vehicle boxes and lists every detected vehicle,
including those without a readable plate. Each plate shows its matched vehicle
type and detector confidence. JSON includes `vehicles`, `num_vehicles_detected`,
and per-plate `vehicle_id`, `vehicle_type`, `vehicle_confidence`, and
`vehicle_match_status`. IDs are local to one image, not tracking identities.

Matching requires at least 90% plate containment in exactly one vehicle box and
vehicle confidence of at least 0.50 (detection starts at 0.35). Overlapping
candidates, multiple plates claiming one vehicle, weak detections, and missing
matches produce `unknown` and a review status. Plate OCR acceptance remains
separate from vehicle matching. These thresholds and confidence scores are not
calibrated probabilities. Unsupported types such as auto-rickshaws may be missed
or misclassified by the pretrained model. Vehicle classification and association
still need an independent labelled evaluation across adverse weather; existing
plate benchmarks do not measure these features. No fine rules or truck weight
subcategories are implemented.

```powershell
python ml/scripts/download_weights.py
python ml/scripts/decision_engine.py --image path/to/vehicle.jpg
# Explicitly choose any trained detector:
python ml/scripts/decision_engine.py --image path/to/vehicle.jpg --weights ml/models/yolo_runs/plate_detector3/weights/best.pt
# Disable optional super-resolution:
python ml/scripts/decision_engine.py --image path/to/vehicle.jpg --no-upscale
```

Training and model download record a checkpoint and SHA-256 in `ml/models/active_detector.json`. An explicit `--weights` overrides it. For older installations without that file, inference warns and selects the most recently modified `*/weights/best.pt`; this is a migration fallback, not a claim that the newest model is the most accurate. A changed or missing registered checkpoint produces an error instead of silently switching models.

The default `--profile balanced` detects and reads original pixels first. Strong, unambiguous readings can skip optional restoration and super-resolution unless weather is confidently severe. `--profile exhaustive` retains the relevant restoration pass and comparison of original/restored readings, which can help find additional plates missed in the original view. Neither profile guarantees detection completeness. A frame with no detections gets one bounded higher-resolution retry; `--imgsz` controls the initial detector size (default 640).

Restoration blends overlapping native-resolution tiles and batches two tiles on GPU. OCR gets a padded crop, with bicubic and contrast alternatives for weak readings; optional super-resolution retains the original evidence. OCR detection is bounded to 640 pixels with four ONNX intra-op threads. Existing restoration weights were trained on resized scenes; the new native-patch training policy requires retraining before its benefits can be measured.

On the 20-image development diagnostic (25 plates, existing weights, super-resolution disabled), exact readings increased **15 to 16**, while warm mean latency fell **2.14 to 0.84 seconds/image**. One incorrect reading remained accepted in both runs. This is a small, non-independent diagnostic, not a real-weather generalization claim. See [measurement details and limitations](docs/optimization.md).

Each detected plate includes:

- `plate_text`: populated only when the selection policy accepts the reading.
- `status`: `accepted`, `uncertain`, or `unreadable`.
- `proposed_text`, `alternatives`, and `readings`: evidence retained for review.
- `ocr_confidence`, `format_score`, and `selection_score`: separate signals. None is a calibrated probability that the registration is correct.
- `ocr_source`, `upscaled`, bounding box, and detector confidence.

Strong competing readings cause abstention. Ambiguous substitutions such as `0 -> O/Q/D` remain alternatives. No-series plate formats receive a conservative review-only score because dropped letters can create false matches. This deliberately reduces automatic coverage. Format support is limited to the templates in `plate_validator.py`; this is not a registration-database lookup. Default acceptance thresholds need calibration on validation data, with the test set left untouched.

## Prepare data and train

Build the raw manifest **before** augmentation. The dataset must contain the annotation/image folders declared by `build_manifest.py`; a different downloaded dataset requires adapting those folder mappings. Missing input images and incomplete annotation matches are errors.

```powershell
python ml/scripts/download_dataset.py
python ml/scripts/build_manifest.py
python ml/scripts/weather_augment.py
python ml/scripts/prepare_yolo_dataset.py

python ml/scripts/train_quality_analyzer.py --epochs 15
python ml/scripts/train_restoration.py --condition haze --epochs 30 --residual
python ml/scripts/train_restoration.py --condition rain --epochs 30 --residual
python ml/scripts/train_restoration.py --condition blur --epochs 30 --residual --edge_weight 2.0
python ml/scripts/train_yolo.py --epochs 100

# Optional, after training:
python ml/scripts/export_unet.py
```

All training stages use the same deterministic source-photo split: 15% validation and 15% test by default, with the remainder for training. Weather variants stay with their original photo. The test partition is excluded from training and checkpoint selection. Use identical `--val_frac` and `--test_frac` across all stages. Prefer an explicit `split` (`train`, `val`, `test`) on every manifest entry when adding data; mixed or conflicting assignments are rejected. Rebuilding data after changing its contents does not make the resulting test partition independent of previously trained weights.

YOLO preparation creates a fresh `ml/data/yolo_dataset/builds/<id>/` and publishes a portable `data.yaml` only after completion. Previous builds are preserved. Each build contains train/validation/test manifests and source provenance. Training reports the actual auto-numbered output directory and activates its best checkpoint.

Restoration now trains on aligned native-resolution patches, focusing 70% of crop choices on labeled plates when available, with 10% clear identity examples and no mirrored character augmentation. Validation crops are deterministic. CUDA mixed precision is enabled by default (`--no-amp` disables it); pinned-memory loading and persistent workers reduce transfer/loading overhead. `--workers`, `--patience` (default 8), and `--out_dir` control loading, early stopping and candidate checkpoint isolation. Use a smaller `--batch_size` on GPUs with limited memory. Native-patch validation PSNR is not directly comparable to historical resized-scene PSNR.

Residual restoration is enabled by default; `--no-residual` explicitly trains the older direct-output architecture. Checkpoints store architecture and loss settings. `--ssim_weight 0` explicitly disables SSIM; otherwise a missing dependency stops training. Quality checkpoints maximize the harmonic mean of condition and severity accuracy, using validation loss as the tie-breaker.

Each newly trained checkpoint has a `.provenance.json` containing checkpoint and seen-data fingerprints. Preserve these files with the weights. Model selection counts as having seen validation data. TorchScript exports similarly have a hash sidecar and are only loaded when both source-checkpoint and exported-file hashes match; stale or unverified exports fall back to the checkpoint.

## Independent end-to-end evaluation

Provide a manifest of full images with all plates transcribed. Empty `boxes` lists are valid negative images; partially labeled images are not suitable. Paths are relative to `--data-root` (default `ml/`), and all variants of a photo must share a stable `source_id`. Augmented variants should also retain their original image's `source_sha256`.

```json
[
  {
    "image_path": "external_test/rain/photo01.jpg",
    "source_id": "external-test:photo01",
    "width": 1920,
    "height": 1080,
    "condition": "rain",
    "severity": "moderate",
    "split": "test",
    "boxes": [
      {"xmin": 500, "ymin": 650, "xmax": 750, "ymax": 720, "text": "KA01AB1234"}
    ]
  }
]
```

```powershell
python ml/scripts/evaluate_pipeline.py --manifest ml/data/processed/independent_test.json --out ml/data/processed/pipeline_eval_results.json
```

The evaluator checks overlap against **all** pipeline checkpoints' training/validation provenance. Legacy checkpoints without sidecars conservatively exclude the entire original corpus in `--historical-manifest`, including photos newly assigned to test. Only genuinely new images, or checkpoints retrained from the new split, can support an independent test. Fingerprints catch exact duplicates, not every near-duplicate or overlapping camera sequence; curate those groups by source identity.

Evaluation runs the actual full pipeline, matches detections to ground truth one-to-one, and reports detection precision/recall, missed/extra plates, exact end-to-end recall, accepted-reading precision, abstentions, and latency. Results are grouped by annotated weather condition and severity. Missing files, incomplete labels, overlap, and inference errors fail the run rather than silently dropping difficult examples. The report records code, model and manifest hashes. A low exact-reading rate cannot be hidden by excluding missed detections.

`evaluate_ocr.py` remains a ground-truth-crop component diagnostic and writes a separate `ocr_component_results_current.json`. The historical `ocr_eval_results.json` is preserved as historical evidence, not a benchmark of the revised pipeline.

## Regression checks

These tests require Python and NumPy, but no model downloads or GPU:

```powershell
python -m unittest discover -s ml/tests -v
```

They exercise real selection, geometry, splitting, provenance, and metric logic, plus pipeline integration with deterministic model doubles. CI runs them on Windows and Linux. They do not substitute for numerical ML-runtime smoke checks or real-image evaluation.

See [the remediation record](docs/remediation.md) for the mapping from the ranked review to code changes and remaining validation work.
