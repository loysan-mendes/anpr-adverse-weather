# DataCluster vehicle experiment

Source: https://huggingface.co/datasets/Dataclusterlabspvtltd/Indian-Vehicle-Dataset

Downloaded revision: `e3d82e51595f1f42ec1ee22b23aff95f442e4013`.
The actual download contains **100 JPEGs and 100 Pascal VOC XML files**, not the
50,000-image full commercial collection. The dataset card lists CC BY-NC-ND 4.0
and permits academic/non-commercial research. Downloaded media and generated
training data/checkpoints remain local and ignored by Git; do not redistribute
them as part of this repository. Attribute DataCluster Labs in the project report.

## Annotation audit and repair

The original filename pairs are incorrect. Only 44 of 100 XML dimension pairs
match the EXIF-oriented image, and matching dimensions alone do not prove a
correct pairing. Contact-sheet inspection identified these repairs:

- Images 1–66 use XML 2–67.
- Images 68–97 use their same-numbered XML.
- Image 98 uses XML 99.
- Image 100 uses XML 1 (XML 100 duplicates XML 1).
- Images 67 and 99 are excluded because a reliable matching annotation was not verified.

`train_vehicle.py` verifies an annotation-set fingerprint before applying this
release-specific repair. Original downloaded files are never modified. It checks
EXIF-oriented dimensions, box bounds and class names, records source and annotation
hashes, and writes a separate resized YOLO dataset. The 98 retained pairs were
inspected as box-overlay contact sheets. Some source annotations omit background
vehicles or use questionable class labels; the scores remain sample diagnostics.

Concrete mixers and tankers map to truck. Auto-rickshaws, tractors, tempos and
bicycles map to an internal `other` class. The application displays it as unknown;
it does not add another supported vehicle category. No plate boxes or OCR
transcriptions are provided in this dataset.

## Split and experiment

Known acquisition-date groups, exact duplicates and perceptually similar images
(64-bit difference-hash distance at most four) stay together. Seed 42 yields
73 training, 11 validation and 14 test images. Incomplete source identity means
this is not a guarantee against all near-duplicate leakage. Validation contains
only one bus box, so class averages are particularly unstable.

The baseline comparison uses IoU 0.5, detection confidence 0.35 and identical
image inputs. Matching is one-to-one within each class; a wrong class counts as
both a false positive and a missed true class. Macro F1 averages car, motorcycle,
bus and truck; other is reported separately. These are not OCR accuracy scores.
Warm detector timing excludes model load, OCR and the rest of the application.

```powershell
# Download the exact audited source revision using the same HF Hub API as the CLI:
./run_ml.ps1 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dataclusterlabspvtltd/Indian-Vehicle-Dataset', repo_type='dataset', revision='e3d82e51595f1f42ec1ee22b23aff95f442e4013', local_dir='indian-vehicle-dataset')"

# Each run needs a fresh output directory; existing splits are protected.
./run_ml.ps1 ml/scripts/train_vehicle.py --output ml/data/vehicle_sample
./run_ml.ps1 ml/scripts/train_vehicle.py --output ml/data/vehicle_sample_preserved --preserve-head --epochs 50

./run_ml.ps1 ml/scripts/evaluate_vehicle.py --weights ml/yolo11n.pt yolo11s.pt ml/models/vehicle_runs/datacluster_nano/weights/best.pt --data ml/data/vehicle_sample --out .runtime_cache/vehicle-val.json
```

The second training policy keeps all 80 COCO output channels to retain learned
vehicle weights, repurposes channel 6 (train) as other, and trains with AdamW at
0.0001. Inference selects classes by checkpoint names. Unused COCO categories
are not exposed by the application. Both runs freeze the backbone's first ten
modules and retain original pretrained and plate checkpoints.
