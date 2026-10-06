# Balanced candidate: training and development evaluation

Completed on 2 October 2026. The balanced candidate restores much of the first candidate's lost historical detection coverage, but does not improve complete reading reliability across both development datasets. Keep it as an experimental checkpoint and retain `plate_detector3` as the active detector while addressing OCR/crop behavior.

## Training confirmed

- Run: `ml/models/yolo_runs/indian_plates_balanced_candidate`.
- 80 epochs completed; the user reports 1.183 hours. The highest CSV checkpoint fitness is at epoch 75, rather than the final epoch.
- Initial weights: local COCO-pretrained `yolo11s.pt`; input 640, batch 4, workers 0, seed 42 and the documented augmentation settings.
- Frozen prepared build: `ml/data/indian_plates/balanced/builds/21a8846051c43b1b26ac`: 1,272 train, 338 validation and 247 test images.
- Candidate checkpoint SHA-256: `42e94af496c357310f27d048a7c492a2d257a9b4ee1c0a63a2a02042aab28b32`.
- The checkpoint's provenance matches the prepared training/validation provenance exactly. The reserved Indian test sources, groups and pixels remain excluded from fitting.
- The supplied combined validation results are precision **88.88%**, recall **92.22%**, mAP50 **94.74%** and mAP50-95 **74.78%**.

The 338-image validation set contains 247 Indian examples plus 91 historical synthetic weather variants. Its overall score is not directly comparable with the first candidate's score on only 247 Indian examples. The separate `indian_plates_balanced_candidate2` folder contains the training script's additional validation output; it is not a second trained checkpoint.

## Same-domain detector comparison

All three checkpoints were newly evaluated on each of the two fixed validation domains, with input 640, batch 4, workers 0 and full precision. Each also underwent original-image prediction at the app confidence threshold 0.25, with matching IoU 0.5 and NMS IoU 0.7. This diagnostic uses the supplied boxes, which do not certify every plate in a scene.

| Detector | Indian validation mAP50 | Indian mAP50-95 | Supplied Indian targets matched at confidence 0.25 |
| --- | ---: | ---: | ---: |
| Active current detector | 95.30% | 60.88% | 214 / 247 |
| First Indian candidate | **99.49%** | **82.66%** | 246 / 247 |
| Balanced candidate | 99.09% | 80.48% | 246 / 247 |

| Detector | Historical weather mAP50 | Historical mAP50-95 | Supplied historical targets matched at confidence 0.25 |
| --- | ---: | ---: | ---: |
| Active current detector | **79.72%** | **67.36%** | 60 / 91 |
| First Indian candidate | 60.55% | 43.44% | 56 / 91 |
| Balanced candidate | 68.59% | 50.74% | **67 / 91** |

The historical domain comprises only **seven source photos**, each with 13 clear/weather derivatives. It is a development stress check, not 91 independent examples or a verified real-weather benchmark. The active detector's stages may have seen these source photos previously.

The balanced candidate's historical recall improves at the app threshold, while its box precision/localization remains weaker than the active detector's. It produces 29 unmatched historical predictions, compared with 36 for the first candidate and zero for the active detector. Representative visual review found differently sized or overlapping boxes around real target plates, a wide bus-front box that fails the IoU cutoff, and an actual bus plate omitted from the car-only scene annotation. These 29 predictions cannot all be called false detections. The original labels were retained, and the matching threshold was not changed to favor a candidate.

Several GPU allocator warnings recovered during detector evaluation. All domain evaluations and prediction passes completed; their timings should not be used to claim a deployment speed improvement.

## Complete recognition results

The balanced checkpoint processed the same full 247 Indian validation images and 20 historical diagnostic images through quality assessment, vehicle detection, conditional restoration, plate detection and OCR. OCR used predicted crops. Settings remain balanced profile, confidence 0.25, input 640 and Real-ESRGAN disabled; normal bicubic/contrast fallback and restoration remain enabled.

The other two models' baselines reuse the verified 1 October results after the literal-reading fix. All recorded OCR processing code, input hashes, package versions, shared stage checkpoints and processing options were checked against the new run. The preparation-only cache-check change has no effect on inference. This comparison is about recognition outcomes; it is not a simultaneous latency benchmark.

| Measurement | Active current detector | First Indian candidate | Balanced candidate |
| --- | ---: | ---: | ---: |
| Indian targets matched by full pipeline | 224 / 247 | 246 / 247 | **247 / 247** |
| Correct complete Indian readings accepted | 144 / 247 | **157 / 247** | 152 / 247 |
| Exact accepted Indian target recall | 58.30% | **63.56%** | 61.54% |
| Accepted Indian readings disagreeing with source text | **17** | **17** | 22 |
| Accepted target-reading precision | 89.44% | **90.23%** | 87.36% |
| Correct Indian review proposals | 32 | 31 | 35 |
| Historical targets matched | **24 / 25** | 16 / 25 | **24 / 25** |
| Correct complete historical readings accepted | **16 / 25** | 13 / 25 | 15 / 25 |
| Historical accepted source-text disagreements | 0 | 0 | 0 |

The balanced model recovers eight historical matched targets compared with the first Indian candidate, but gains only two correct accepted historical readings. On Indian validation, it loses five correct accepted readings and adds five accepted source-label disagreements. Better box coverage has not produced more reliable complete registrations.

The paired Indian outcomes help explain the difference. All 152 of the balanced candidate's correctly accepted Indian readings were also correct acceptances for the first candidate. Two of that candidate's other correct acceptances move to correct review proposals; three move to incorrect review proposals. Six previously incorrect review proposals become accepted source-label disagreements, while one prior accepted disagreement moves to review.

All six newly accepted disagreements were inspected. They include a low-resolution R/B confusion, a blurred final-digit disagreement, and missing/corrupted leading characters such as source `MH02ER9194` becoming accepted `HO2ER9194`. These are concrete OCR/crop and acceptance-policy failures. Some supplied transcriptions remain ambiguous, including the previously documented I/T case; source texts were not rewritten to agree with predictions. Review notes are in `ml/data/curation/balanced_candidate_failure_review.json`.

The historical 20-image diagnostic spans existing source partitions and may overlap legacy training. It is a compatibility check, not an independent held-out ANPR result. No inference was run on the reserved **247 Indian test images**, and the first candidate's completed detector-test result remains its frozen result. Complete-scene labeling and independent real-weather recognition are still unverified.

## Decision and next work

The [paired crop diagnostic and confidence-gate projection](ocr-diagnostic-next-steps.md) are now complete. Supplied crops improve correct original-only acceptances from 153 to 157, while still accepting 24 source-label disagreements. A final 0.90 confidence gate projects 147 correct full-pipeline acceptances and seven disagreements, compared with the recorded 0.80 baseline of 152 and 22. The [gate has now been implemented](ocr-acceptance-policy-results.md) with a 0.90 prototype default and all-detector comparison. The tables above preserve the original experiment's 0.80 outcomes.

The balanced checkpoint remains inactive. It improves coverage relative to the narrow first candidate, but the current evidence does not justify a universal replacement of the active model. No additional training was started during evaluation.

1. Diagnose OCR separately from detection: compare literal OCR on audited target crops and predicted crops, inspect leading-character clipping, angle and crop margins, and retain all original reading evidence. Choose crop/recognition changes on development data.
2. Calibrate the acceptance policy on reviewed examples. A format-compatible string is not sufficient evidence of a complete registration. Study prefix plausibility and missing-character failures while preserving unusual literal readings for review; do not invent replacement characters or digits.
3. Audit small/angled historical box geometry and complete all visible plate annotations on a new evaluation collection. Keep the existing scores and supplied labels unchanged as the record of this experiment.
4. Collect a separate set of actual rain, haze, night and clear scenes, grouped by vehicle/session. Freeze it for the final full-pipeline evaluation. Recorded-video tracking and agreement across readings remain the next product milestone through November.

Another detector run with more epochs alone is not the next justified experiment. The measured failures now point to recognition and confidence handling, with localization on some small or angled plates.

## Evidence and verification

Local reports are in `ml/benchmarks/balanced_candidate_evaluation/`:

- `training_audit.json`: checkpoint, epochs, training configuration and provenance verification.
- `detector/{current,indian,balanced}.json`: both fixed validation domains, metrics and per-image boxes.
- `pipeline/new_balanced.json` and `pipeline/legacy_balanced.json`: 267 new full-pipeline image runs.
- `comparison.json`: verified comparison with 534 saved baseline runs, paired target outcomes and the inactive-candidate decision.
- Review contact sheets and `historical_unmatched_review_index.json`: representative annotation/localization cases.

All 267 unique pipeline input hashes and eight stage/checkpoint hashes were verified. The active registry and model weights remained unchanged. The frozen prepared inventory was verified after training; only the known Ultralytics `labels/train.cache`, `labels/val.cache` and `labels/test.cache` files are ignored as derived indexes. Changed images, labels and unexpected extra files still fail validation. A regression test covers that distinction.

The tracked dataset checker was corrected without rebuilding the trained dataset or changing its current pointer. **90 ML regression tests pass.**

For the project review:

> We completed a broader detector experiment and recovered most of the detection coverage lost on our earlier vehicle examples. We then measured complete recognition and found that OCR errors still limit reliable plate reading. We are keeping the candidate experimental while improving crop/recognition quality and review decisions, followed by an independently annotated real-weather evaluation and recorded-video demonstration.
