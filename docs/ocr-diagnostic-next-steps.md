# Next step: OCR reliability and review decisions

Completed on 2 October 2026, after the [balanced detector comparison](indian-plates-balanced-results.md). The next implementation experiment should be a stricter **final acceptance gate**, followed by recognition improvements on reviewed development examples. Further detector training alone is not justified by these results.

Follow-up: the [final gate is now implemented](ocr-acceptance-policy-results.md), compared across all three detectors and set to a 0.90 prototype default. The diagnostic measurements below retain their original policy and scope.

## Crop diagnostic completed

The new diagnostic compares OCR from supplied plate boxes against the balanced detector's saved predicted boxes. Both use original image pixels, the existing crop padding and balanced OCR fallback, with Real-ESRGAN disabled. Supplied crops were newly processed; predicted-crop evidence was reused after checking input hashes, processing code, runtime packages and stage checkpoints. Image restoration is excluded from this crop comparison.

| Indian validation, 247 paired targets | Predicted-box crops | Supplied-box crops |
| --- | ---: | ---: |
| Correct complete readings accepted | 153 | 157 |
| Accepted readings disagreeing with source text | 25 | 24 |
| Correct review proposals | 34 | 29 |
| Incorrect review proposals | 28 | 29 |
| Unreadable | 7 | 8 |
| Exact literal evidence available in any OCR reading | 166 | 169 |

Supplied boxes yield ten newly correct acceptances while losing six, for a net gain of four. Five previously correct acceptances become source-label disagreements. The supplied boxes are a localization control, not guaranteed ideal crops. Cropping affects individual results but does not explain most errors.

All 25 predicted-crop accepted disagreements lack the correct literal text in their saved original-image OCR readings. Recognition errors therefore remain even before selection among readings. Missing or corrupted leading characters and blurred digits are priorities for review.

The historical diagnostic has 25 targets in 20 photographs, with 24 matched by the balanced full pipeline. On those 24 paired crops, both crop sources correctly accept 15 targets. Supplied crops accept one source-label disagreement, whereas predicted crops leave that target unreadable. The unmatched target remains outside the predicted-crop denominator. Results for all supplied historical crops are recorded separately.

These crop-only numbers differ from full-pipeline results because the full pipeline also considers restored-image readings and conditional detection. They must not be presented as new end-to-end accuracy.

## Final confidence gate: offline projection

This experiment replays the balanced model's saved **full-pipeline final decisions**. A previously accepted reading below the chosen OCR-confidence cutoff moves to review; text proposals, boxes and inference routing remain unchanged. It does not promote existing review results or rerun inference.

| Minimum final OCR confidence | Correct Indian acceptances / 247 | Accepted source-label disagreements | Matched targets needing review | Correct historical acceptances / 25 |
| --- | ---: | ---: | ---: | ---: |
| 0.80, existing decisions | 152 | 22 | 66 | 15 |
| 0.85 | 152 | 18 | 70 | 15 |
| **0.90, candidate to test** | **147** | **7** | **86** | **15** |
| 0.95 | 105 | 0 | 135 | 12 |
| 0.975 | 77 | 0 | 163 | 9 |

Seven Indian targets remain unreadable in every row. The historical baseline includes one missed target, one unreadable target and eight review proposals; its 0.90 projection preserves those outcomes. Unmatched scene predictions are recorded separately because the supplied annotations do not cover every plate.

At 0.90, 15 accepted Indian source-label disagreements move to review, along with five correct acceptances. Accepted target precision projects from 87.36% to 95.45%, while exact accepted target recall projects from 61.54% to 59.51%. These are development measurements against supplied texts, not a calibrated confidence guarantee or independent deployment performance. A 0.95 cutoff loses 47 correct Indian acceptances; zero observed disagreements in that row does not establish zero future errors.

| Source | Targets | Correct acceptances, 0.80 to 0.90 | Accepted disagreements, 0.80 to 0.90 |
| --- | ---: | ---: | ---: |
| OLX photographs | 71 | 38 to 37 | 2 to 1 |
| Google images | 16 | 15 to 15 | 0 to 0 |
| Extracted video frames | 160 | 99 to 95 | 20 to 6 |

Most of the apparent benefit is on frames from only five related video groups. Equal weighting of the 92 related groups gives exact accepted recall of 61.54% at 0.80 and 60.42% at 0.90. Grouping reduces frame-count dominance; it does not certify independence or label correctness. Some supplied transcriptions remain ambiguous, and historical photos may overlap legacy stage training.

Changing an earlier OCR candidate-selection threshold can alter fallback and restoration calls. This replay supports testing a gate **after final reading selection**; it does not predict the behavior or latency of changing earlier thresholds.

## Concrete work order

1. Implement an experimental final acceptance cutoff with 0.80, 0.85 and 0.90 options. Preserve the literal proposal and evidence when moving a reading to review. Verify the expected transitions, compare all three detectors on the same development images, and measure review workload before choosing a default.
2. Inspect the remaining high-confidence disagreements and incorrect review proposals. Compare a small number of crop margins and recognition variants on reviewed development examples, including leading-character loss, angle and blur. Keep supplied transcriptions unchanged unless an independent label audit justifies a recorded correction.
3. Have the team collect an initial **20-30 fresh source photographs and a few short clips** across clear daylight, night and available real rain/haze. Include motorcycles, cars, buses, distant/angled plates and multi-vehicle scenes. Label every visible plate box and readable text; record unreadable plates explicitly. Keep each vehicle/session together. This pilot establishes the collection workflow; the larger collection and independent final set remain in the [November plan](future-plan.md).
4. After static-image confidence and review behavior are measured, add recorded-video tracking and agreement across readings from the same vehicle passage. Then build the timestamped event log and review interface for the November demonstration.

The balanced detector remains experimental, and the current active detector is unchanged. No further training was started. No new inference was performed on the reserved 247 Indian test images. A final model/policy decision needs independently reviewed development evidence, followed by the separately frozen real-weather evaluation.

## Evidence and reproduction

- `ml/scripts/diagnose_ocr_crops.py`: reusable paired crop diagnostic; rejects explicitly marked test manifests and verifies its saved evidence.
- `ml/benchmarks/ocr_crop_diagnostic/indian_balanced.json`: 247 paired Indian targets.
- `ml/benchmarks/ocr_crop_diagnostic/historical_balanced.json`: all 25 historical supplied crops and 24 paired targets.
- `ml/benchmarks/ocr_crop_diagnostic/acceptance_tradeoffs.json`: threshold projections, source/group breakdowns and input-report hashes.
- `.runtime_cache/analyze_ocr_confidence_tradeoffs.py`: local replay helper; no production mutation or inference.

The original diagnostic used the processing-code snapshot bound to its saved reports. After the gate implementation, first create fresh baseline pipeline evidence at 0.80, then use it for the crop comparison. Keep these new outputs separate from the archived results:

```powershell
.\run_ml.ps1 -PythonArguments @(
  'ml/scripts/benchmark_plate_pipeline.py',
  '--manifest', 'ml/data/indian_plates/builds/2b26d88262afc2f3a4c6/ocr_val_manifest.json',
  '--weights', 'ml/models/yolo_runs/indian_plates_balanced_candidate/weights/best.pt',
  '--acceptance-min-confidence', '0.80',
  '--out', 'ml/benchmarks/ocr_acceptance_policy/reproduced_pipeline.json',
  '--resume'
)
.\run_ml.ps1 -PythonArguments @(
  'ml/scripts/diagnose_ocr_crops.py',
  '--manifest', 'ml/data/indian_plates/builds/2b26d88262afc2f3a4c6/ocr_val_manifest.json',
  '--pipeline-report', 'ml/benchmarks/ocr_acceptance_policy/reproduced_pipeline.json',
  '--out', 'ml/benchmarks/ocr_acceptance_policy/reproduced_crops.json',
  '--resume'
)
```

Resume checks reject changed evidence or processing code. Five new regression tests cover exact accepted versus review scoring, literal evidence, unmatched-crop denominators, source-label disagreements and fractional crop geometry. **95 ML regression tests passed.** Model weights, active registry and source labels remained unchanged.

For the next project meeting:

> We completed detector training and compared complete plate readings. Our next milestone is improving OCR reliability and sending doubtful readings for review. Crop tests show that recognition remains the main limitation, and an experimental confidence gate reduces accepted disagreements with a modest loss of automatic readings. We will validate that on fresh real-weather examples, then combine observations across recorded video and demonstrate a vehicle-entry log by 30 November.
