# Full OCR pipeline comparison

Initial comparison completed on 1 October 2026, before the literal-reading policy fix. The new detector improves accepted exact readings against the new dataset's supplied validation targets, but regresses on the existing development examples. The [subsequent OCR fixes and prepared broader candidate](indian-plates-next-candidate.md) include verified before/after results and the next training command. The current detector remains active.

## What was evaluated

Both detectors processed the same full images through the current quality assessment, vehicle detection, plate detection, conditional restoration, OCR and reading-selection pipeline. OCR received predicted plate crops, not ground-truth crops. Ground-truth crops were used only for visual annotation/failure review.

- New dataset: all 247 validation images, one supplied target per image, in 92 related-image groups. These comprise 71 OLX images, 16 Google images and 160 extracted frames from five video-related groups.
- Historical diagnostic: the existing 20 fully transcribed positive images, with 25 supplied target boxes. Legacy checkpoints may have seen these photos during training. Some visible source transcriptions, including `KL34F`, are partial.
- Settings: balanced profile, confidence 0.25, detector input size 640, matching IoU 0.5, and Real-ESRGAN disabled, matching the current app configuration. Bicubic/contrast fallback and conditional restoration remain enabled by the normal pipeline policy.
- Only the plate-detector checkpoint changed between each paired run. Vehicle, quality and restoration checkpoints and all other options were hash-checked and identical.
- The active-detector registry and model checkpoints remained unchanged. No model was trained, activated or tuned during this comparison.

This is a **development diagnostic against supplied target annotations**. New source transcriptions and full-scene annotation completeness are not all independently verified. Unmatched detections are retained for review, rather than being labeled false detections. The reserved detector test was not used for OCR tuning or this diagnostic, and the independent full-image evaluator's guards remain in place.

## New dataset: 247 validation targets

| Measurement | Current detector | Indian candidate |
| --- | ---: | ---: |
| Targets matched by full-pipeline detections | 224 / 247 | **246 / 247** |
| Targets not matched | 23 | **1** |
| Correct complete readings accepted | 146 / 247 | **157 / 247** |
| Exact accepted target recall | 59.11% | **63.56%** |
| Accepted target readings disagreeing with source text | 19 | 18 |
| Accepted reading precision among matched targets | 88.48% | 89.71% |
| Correct proposals left in review | 27 | 27 |
| Unmatched scene detections | 14 | 9 |
| Accepted unmatched scene readings needing review | 5 | 1 |

The candidate yields 11 additional correct accepted target readings, a 4.45 percentage-point gain. It does not turn the detector's approximately 99% mAP50 into 99% complete registration-reading accuracy. Reading all characters correctly and deciding whether to accept them are separate stages.

The candidate's 247 targets divide into 157 correct accepted readings, 18 accepted label disagreements, 27 correct review proposals, 35 incorrect review proposals, nine unreadable matched targets and one missed target. Including correct proposals would produce 184/247 (74.49%), but those proposals require review and cannot be reported as automatically accepted successes.

Equal-weight averaging over related-image groups gives exact accepted target recall of 59.34% for the current detector and 61.57% for the candidate. This supplements the image-weighted results because 160 extracted frames come from only five groups. It does not make these groups an independent deployment sample.

| Candidate source | Matched targets | Correct accepted readings |
| --- | ---: | ---: |
| OLX | 71 / 71 | 38 / 71 (53.52%) |
| Google | 15 / 16 | 15 / 16 (93.75%) |
| Extracted video frames | 160 / 160 | 104 / 160 (65.00%) |

The full pipeline rescues some current-detector misses through its normal retry/restoration paths, so its 224 matches differ from the earlier original-image-only detector check's 214 matches. The candidate triggered restoration on 23 images and the current detector on 26. These are routing counts, not verified weather labels or evidence that restoration improves recognition; an explicit restoration ablation remains future work.

## Historical development regression

| Measurement | Current detector | Indian candidate |
| --- | ---: | ---: |
| Matched supplied targets | **24 / 25** | 16 / 25 |
| Targets not matched | **1** | 9 |
| Correct accepted readings | **16 / 25 (64%)** | 13 / 25 (52%) |
| Accepted target label disagreements | 1 | 1 |

Five of the candidate's unmatched targets occur in the same high-resolution scene with multiple vehicles and small plates. Other misses include a bus, autorickshaw, truck and motorcycle plate. Visual review used EXIF-oriented images consistent with OpenCV and the manifest coordinates.

This is a useful compatibility check for the prototype, not an independent superiority benchmark for the historical detector. It provides a concrete reason to test broader source coverage before replacing the active checkpoint.

## Annotation and failure review

A deterministic pilot was selected before pipeline inference: 12 OLX images, eight Google images and one frame from each of the five video groups. Full-image and target-crop inspection found 24 supplied target transcriptions visually consistent; one small blurred plate had limited independent legibility. This assistant visual review does not certify full-scene labeling or substitute for independent human verification.

On the 24 visually consistent pilot targets, both detectors correctly accepted 17 readings, left three correct proposals for review, disagreed with two source texts and missed one target. There was no exact accepted-reading gain on this small pilot.

All 18 candidate accepted-label disagreements were visually reviewed. Repeated `MH02BT6482` annotations contain a difficult `I`/`T` series distinction; that transcription needs independent confirmation. Keep the source-text diagnostic counts and the uncertainty visible. No prepared labels were rewritten to agree with a model.

Two confirmed and actionable examples:

- `PY036993`: raw OCR returned the correct literal string at approximately 0.947 confidence. Format scoring proposed `PY03G993`, gave the altered layout a higher score and accepted it. The literal no-series layout remains review-only, while its coerced alternative currently passes acceptance. Reading selection should preserve this ambiguity instead of promoting the altered value.
- `MH43AF5037`: the visible plate includes district digits `43`; the accepted reading dropped `4` and became `MH3AF5037`. A syntactically plausible layout does not establish correctness. Other failures lose or substitute state-prefix characters on tilted or blurred frames.

The candidate's one accepted unmatched reading occurs on the second visible side registration marking in `video3_650.jpg`: `MH02ER9194`. Visual inspection supports that reading, but the supplied box labels only the rear plate. Counting this as a confirmed false positive would misrepresent the scene.

The review record is `ml/data/curation/indian_plate_ocr_review.json`. Review sheets and detailed per-image output are in the ignored local directory `ml/benchmarks/indian_candidate_ocr/`.

## Timing and checks

Full-pipeline warm mean time on the new validation images was 0.344 seconds with the current detector and 0.350 seconds with the candidate. Historical warm means were 0.845 and 0.808 seconds respectively. First-image initialization was excluded from these warm means. These were single sequential diagnostic runs, with differing cache warmth at startup; treat them as descriptive measurements rather than a speed-improvement claim. The detector, quality/restoration and vehicle stages ran on the RTX 3050; PaddleOCR used ONNX Runtime on CPU.

Seventy-five ML tests passed, including six new checks covering omitted-scene predictions, one-to-one matching, missed-target denominators, review-only proposals, wrong accepted readings and missing transcriptions. Paired reports were checked for identical manifests, image hashes, pipeline code, settings and shared checkpoints. Outcome totals cover every supplied target. Active weights and the original prepared dataset remain unchanged.

## Next experiment

1. Fix reading selection so it retains correct literal evidence and sends incompatible format alternatives for review. Add tests for the observed literal-to-coerced promotion before calibrating any policy change on development/validation data.
2. Review the detector's training coverage and prepare a balanced candidate combining suitable historical training examples with the new training partition, emphasizing multiple vehicles, small plates, two-line plates and different vehicle types. Preserve source/sequence groups and keep held-out images outside fitting and checkpoint selection. Audit incomplete historical labels before reuse.
3. Re-run both the new validation diagnostic and historical regression check for any changed candidate or policy. Confirm improvements in exact accepted readings and wrong accepted readings, as well as detection.
4. Collect independently labeled real rain, haze, night and clear scenes, including negatives and all visible plates, for the final full-pipeline evaluation. An independent real-weather accuracy claim is not available from these data.

The candidate remains available for experiments. Keep the current detector active until broader detection coverage and the complete reading behavior support replacement.

## Reproduce the new validation comparison

```powershell
$build = (Get-Content ml/data/indian_plates/current.json | ConvertFrom-Json).build
$manifest = "ml/data/indian_plates/$build/val_manifest.json"
.\run_ml.ps1 ml/scripts/benchmark_plate_pipeline.py --manifest $manifest --weights ml/models/yolo_runs/plate_detector3/weights/best.pt --out ml/benchmarks/indian_candidate_ocr/new_current.json
.\run_ml.ps1 ml/scripts/benchmark_plate_pipeline.py --manifest $manifest --weights ml/models/yolo_runs/indian_plates_candidate/weights/best.pt --out ml/benchmarks/indian_candidate_ocr/new_candidate.json
```

The benchmark fails on missing transcriptions or invalid boxes instead of silently omitting them. `--resume` reuses saved per-image results only when code, inputs, settings and checkpoint fingerprints match. It always records its target-only development scope and keeps original OCR region evidence. Independent complete-scene evaluation still uses `evaluate_pipeline.py` with its annotation and provenance checks.

The completed paired outputs are `new_current.json`, `new_candidate.json`, `legacy_current.json` and `legacy_candidate.json`, with a joined `comparison.json`, in `ml/benchmarks/indian_candidate_ocr/`.

For the next project review:

> Our new detector achieved strong held-out plate-detection scores. We then checked the complete recognition pipeline: correct accepted target readings on the new validation data rose from 59.11% to 63.56%. We also found a regression on our existing examples and cases where OCR selection accepts altered or incomplete readings. We are using these failures to improve reading selection and dataset coverage before replacing the active model, followed by independent real adverse-weather evaluation.
