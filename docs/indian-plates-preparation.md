# Prepared Indian plate dataset

Prepared on 30 September 2026 from the supplied Kaggle download. The isolated plate-detector candidate has now completed 80 epochs. See [the training results](indian-plates-training-results.md) for validation and the comparison with the current detector.

## Current training data

| Partition | Plate-detection images | Images with eligible supplied OCR text |
| --- | ---: | ---: |
| Training | 1,150 | 1,128 |
| Validation | 247 | 247 |
| Test | 247 | 246 |
| Total | **1,644** | **1,621** |

The 23 detection-only images retain their plate boxes but have no OCR ground-truth transcription. Their original annotation strings and reasons remain in the manifest. These include masked fields, another script, unclear character identities and nonstandard illustrative text.

Training configuration: `ml/data/indian_plates/data.yaml`. The current immutable build is identified by `ml/data/indian_plates/current.json`; manifests, labels, corrected VOC annotations and audit records are inside its build directory.

## Resolved preparation issues

- Removed 48 identical decoded-image copies from the prepared dataset while retaining their source aliases and hashes.
- Excluded three vehicle-model display images, one foreign plate image and two additional image encodings without independent annotations. The orphan `MH5.xml` is recorded as missing-source evidence and excluded.
- Corrected the two stale image-size headers and the `ML39.xml` filename in the derived annotations, preserving visually reviewed box coordinates.
- Normalized the EXIF-oriented image and two JPEG containers that YOLO would otherwise rewrite. Lossless PNG conversion preserves decoded image pixels.
- Read transcriptions from the original XML object names, then exported a single `number_plate` detection class with separate text attributes. The unsuitable bundled `classes.txt` is not used.
- Split related plate identities, extracted-video sequences, exact copies and perceptually similar image candidates together. Photo IDs remain unique so future clear/degraded variants can still be paired correctly.
- Added review-only handling for additional observed layouts and retained unusual literal OCR readings as proposals. Automatic acceptance thresholds were not lowered.
- Added source fingerprints, curation policy fingerprints, file inventories and fitting/model-selection provenance. Changed source files require renewed review rather than silently inheriting old repairs.
- Added configurable weather-augmentation output paths and propagation of related-image groups. Unverified clear-weather references require an explicit experimental assumption or a reviewed source manifest.
- Added a full-image evaluation guard: this dataset's provided-target annotations cannot be presented as complete, independently verified ANPR ground truth.

Original downloads, historical manifests and the active detector checkpoint are preserved. Generated media and training derivatives are ignored by Git; the importer and reviewed policy are reproducible source files.

## Reproduce preparation

```powershell
.\run_ml.ps1 ml/scripts/prepare_indian_plates.py
```

Identical input, policy and code reuse a verified build. A changed preparation configuration produces a new build; prior builds remain separate. The curated policy is in `ml/data/curation/new_indian_plates.json`, and every exclusion and repair is recorded in the prepared build.

For a build path read from `current.json`, `prepare_indian_plates.validate_build(...)` checks image and label integrity and partition consistency. Local preparation verification also compares decoded pixels to the source, verifies source/checkpoint hashes, and loads the train, validation and test partitions through YOLO's actual dataset loader.

## Completed candidate training

The completed isolated candidate run used the already present small-model weights:

```powershell
.\run_ml.ps1 ml/scripts/train_yolo.py --data ml/data/indian_plates/data.yaml --model yolo11s.pt --epochs 80 --batch 4 --imgsz 640 --workers 0 --name indian_plates_candidate --no-activate
```

This run records fitting and model-selection provenance and keeps the current inference detector active. Horizontal mirroring is disabled to preserve character orientation. Validation was used for checkpoint selection; the frozen candidate was subsequently evaluated on the held-out test partition, as recorded in the training results. Use a new run name for future experiments to keep this result identifiable.

The new detector experiment is separate from vehicle-category detection and from quality/restoration model training.

## Preparation validation (before training)

- 69 ML regression tests and six backend tests passed.
- All 1,644 prepared images decoded with dimensions and decoded pixels matching the source metadata.
- Every exported YOLO label matched the reviewed box coordinates and used class zero.
- No recorded related groups, identities, sequences or content hashes crossed train/validation/test boundaries. The test partition had no overlap with fitting/model-selection provenance.
- YOLO's actual dataset loader scanned 1,150 training, 247 validation and 247 test images with zero corrupt entries. File inventories still matched after loading.
- A two-photo synthetic-augmentation smoke check produced ten outputs, preserving source groups and splits in separate output locations. Unknown clean-weather labels were rejected before writing by default.
- The raw dataset fingerprint, historical manifests and active detector/checkpoint hashes were unchanged. These preparation checks did not start training; the subsequent candidate run is recorded separately.

Local machine verification reports are `.runtime_cache/indian_preparation_verification.json` and `.runtime_cache/augmentation_preparation_verification.json`.

## Practical limits

Structural validity and format screening do not certify every source transcription. The provided boxes identify target plates; completeness of all plates in full scenes has not been verified. Validation on this corpus therefore measures performance against supplied target annotations, rather than certified full-scene deployment accuracy. The evaluator explicitly prevents that distinction from being lost.

The download has no verified weather/severity labels or original video clips. Real rain, haze and night images with complete labels are still needed for the project's final adverse-weather evaluation. Quality/restoration training also needs reviewed clean references before synthetic degradation is treated as paired supervision.

The original dataset audit remains in [the review](new-indian-dataset-review.md). Its listed issues refer to the raw download; the preparation above records how each issue is handled in training inputs.
