# Indian plate detector: completed candidate training

Reviewed on 1 October 2026. The `indian_plates_candidate` run completed all 80 epochs in the reported 1.038 hours (approximately 62 minutes). The candidate improves detection against this dataset's supplied validation targets. Its full OCR pipeline has subsequently been measured in a development diagnostic; independent real adverse-weather performance remains unmeasured.

Update: the [full OCR pipeline development comparison](indian-plates-pipeline-results.md) is now complete. It improves correct accepted readings on new validation targets but regresses on the historical development images. The current detector remains active; independent real-weather performance is still unmeasured.

## Experiment

- Starting model: COCO-pretrained YOLO11s, fine-tuned for one `number_plate` class.
- Training: 1,150 images, 640-pixel input, batch 4, seed 42, workers 0, horizontal flipping disabled.
- Validation: 247 images in 92 related-image groups. Repeated frames are grouped with their sequences and plate identities, rather than treated as independent scenes.
- Held-out test: 247 images in 134 related-image groups, evaluated after validation comparison and checkpoint selection.
- Prepared dataset build: `ml/data/indian_plates/builds/2b26d88262afc2f3a4c6`.
- Candidate: `ml/models/yolo_runs/indian_plates_candidate/weights/best.pt` (19.2 MB).
- Candidate SHA256: `4b1c7cd0c3bbcc335ec1a6cf17b5a8567efd1ae60c9e6bd031004f895b618f2f`.
- Active detector remains `ml/models/yolo_runs/plate_detector3/weights/best.pt`.

The candidate's training/selection provenance sidecar matches the checkpoint hash and the prepared dataset's recorded fitting/selection groups. The CSV contains 80 epochs; its highest weighted mAP checkpoint-selection score occurs at epoch 70. Use `best.pt` for evaluation. `last.pt` represents the final epoch.

`indian_plates_candidate2` contains the trainer's additional validation outputs. It is not a second trained model. The two validation summaries in the terminal reflect separate passes, with slightly different validation execution settings. The final explicit metrics below were reproduced locally.

## Same-set validation comparison

Both checkpoints were evaluated with Ultralytics 8.3.0 on the same 247 validation images, at input size 640, batch 4, workers 0 and full precision on the local RTX 3050. No test predictions or checkpoint activation were performed.

| Metric | Current detector | Indian candidate |
| --- | ---: | ---: |
| Precision | 94.53% | **99.19%** |
| Recall | 89.47% | **99.37%** |
| mAP50 | 95.30% | **99.49%** |
| mAP50-95 | 60.88% | **82.66%** |

The mAP50-95 increase is 21.79 percentage points. This metric averages performance across more demanding box-overlap thresholds, so the improvement supports better localization against the supplied boxes. mAP is a detection metric; exact registration reading needs an OCR evaluation.

The saved training curves show falling training and validation losses without an obvious late deterioration. This is useful training evidence, but it does not establish generalization to different cameras, weather or complete scenes.

## Application-threshold check

A separate original-image detection pass used the existing app confidence threshold of 0.25, input size 640, one image at a time, NMS IoU 0.7, and target matching IoU 0.5. It did not run quality assessment, restoration, OCR or the app's additional detection views.

| Counts against supplied target boxes | Current detector | Indian candidate |
| --- | ---: | ---: |
| Matched target plates | 214 / 247 | **246 / 247** |
| Missed target plates | 33 | **1** |
| Predictions without a matching supplied box | 8 | 6 |

These fixed-threshold counts differ from the standard precision/recall summaries, which use the validator's confidence-curve operating point and validation preprocessing. They must be reported with their own settings.

The six unmatched candidate predictions and one miss were visually inspected:

- `State-wise_OLX/CG/CG22.jpg`: an extra box on the vehicle's grille.
- Google image ending `nudge-guard.jpg.jpeg`: an extra box on the `I DRIVE SAFE` sticker.
- Google image ending `IMG_9072.jpg.jpeg`: the target has unusually large, stylized digits and extends to the image boundary; the candidate produced no box at confidence 0.25.
- `video_images/video2_1750.jpg`: an extra box overlaps a partially visible background vehicle plate without a supplied annotation.
- `video_images/video2_3030.jpg`: an extra box overlaps writing on the rear window.
- `video_images/video3_560.jpg`: an extra box overlaps the bumper guard.
- `video_images/video3_650.jpg`: an extra box covers a second printed registration marking on the vehicle's side, which has no supplied box.

The unmatched predictions cannot all be counted as confirmed false detections. The scenes contain additional registration evidence omitted from the target-only labels. No raw labels or prepared partitions were changed during this review.

## Completed held-out detector test

The frozen candidate was evaluated on the reserved test partition on 1 October 2026. The first user attempt failed before loading images: Ultralytics 8.3.0 interpreted the relative YAML filename beneath its configured `Desktop/datasets` directory. Resolving the YAML to an absolute path fixed the lookup. The prepared YAML, dataset partitions, checkpoint and global dataset-directory setting were unchanged.

Before evaluation, the checkpoint provenance hash, test-group separation, test-source/pixel separation and complete prepared-build inventory were verified. Ultralytics resolved each partition to the expected project build directory and scanned all 247 test images with zero corrupt entries. The checkpoint, active registry and prepared inventory were checked again after evaluation.

| Metric | Candidate validation | Candidate held-out test |
| --- | ---: | ---: |
| Precision | 99.19% | **99.11%** |
| Recall | 99.37% | **97.98%** |
| mAP50 | 99.49% | **99.43%** |
| mAP50-95 | 82.66% | **83.18%** |

Settings were input size 640, batch 4, workers 0 and full precision, with standard validation confidence-curve metrics. The results support consistent detection of supplied target plates across these two partitions. Full-scene labeling, exact OCR correctness and weather-specific performance remain unverified.

Results and plots are in `ml/models/yolo_runs/indian_plates_test2`; the exact metrics, hashes and settings are in that directory's `test_metrics.json`. The earlier `indian_plates_test` directory belongs to the failed path-resolution attempt.

When calling YOLO directly, pass an absolute YAML filename. For example, the following is the corrected equivalent command for reproducing this evaluation; the completed test above does not need another run:

```powershell
.\run_ml.ps1 -c "from pathlib import Path; from ultralytics import YOLO; YOLO('ml/models/yolo_runs/indian_plates_candidate/weights/best.pt').val(data=str(Path('ml/data/indian_plates/data.yaml').resolve()), split='test', imgsz=640, batch=4, workers=0, project='ml/models/yolo_runs', name='indian_plates_test')"
```

The audited local evaluation script is `.runtime_cache/evaluate_indian_plate_test.py`. The normal `train_yolo.py` entry point already resolves `--data` before invoking YOLO.

## What to do next

The [literal-reading fixes and broader candidate preparation](indian-plates-next-candidate.md) are complete. The broader candidate has now completed training and its [2 October development evaluation](indian-plates-balanced-results.md) recovers historical detection coverage but shows mixed complete-reading results. OCR/crop reliability and acceptance calibration are the next priorities; the active detector remains unchanged.

1. Retain the frozen candidate and its completed held-out test report. Further checkpoint or threshold choices belong on validation or new development data; keep this test result separate from tuning.
2. Address the reading-selection failures and historical detection regression found in the completed [pipeline development comparison](indian-plates-pipeline-results.md). Audit evaluation labels, broaden training-source coverage and re-run the paired diagnostics for changed candidates or policies.
3. Collect and label independent real rain, haze, night and clear examples. Use a frozen, fully verified set for final full-pipeline evaluation and for a controlled restoration comparison.
4. Replace the active detector after the detection and full-pipeline checks support that change. Additional detector training is not justified by these detection results alone; choose the next experiment from observed failures.

The user-reported 5.5 ms and 12.0 ms inference figures concern detector validation with different execution settings. They do not measure the complete ANPR pipeline. The local comparison also recovered from GPU allocator warnings, so its timings are unsuitable for a speed comparison. Its numerical candidate metrics reproduced the reported final validation values.

Detailed per-image comparison evidence is in `.runtime_cache/indian_plate_validation_comparison.json`; its reproduction script is `.runtime_cache/compare_indian_plate_validation.py`. These local diagnostic artifacts are ignored by Git. Both checkpoint files and the active-detector registry were hash-checked and unchanged after evaluation.

## Wording for the project review

> We fine-tuned a YOLO11s plate detector on 1,150 prepared Indian vehicle images. On 247 grouped validation images, it achieved 99.49% mAP50 and 82.66% mAP50-95, improving over our previous detector on the same validation set. On the separate 247-image test partition, it achieved 99.43% mAP50 and 83.18% mAP50-95. Our next step is evaluating complete plate-reading correctness and real adverse-weather performance.
