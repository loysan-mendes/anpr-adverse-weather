# Final OCR confidence gate implemented

Completed on 2 October 2026, following the [crop and confidence diagnostic](ocr-diagnostic-next-steps.md). The image pipeline now defaults to a **0.90 final OCR acceptance cutoff**. Readings below that cutoff retain their text proposal and evidence with `uncertain` status; they are excluded from the automatic accepted count. This is a development-selected prototype policy, not a calibrated probability of correctness.

## Behavior

The gate runs after `choose_reading` has finished combining original/restored OCR evidence. It does not change detector boxes, OCR candidate selection, fallback, restoration, vehicle association or model weights. It cannot promote an uncertain or unreadable result, fix characters or resolve a disagreement. Existing review reasons are preserved for those results. Invalid/missing OCR confidence cannot pass the final gate.

The local upload app uses the same default automatically. A gated result appears as **Needs review**, retaining its proposal and reading evidence, with a short explanation. Individual evidence readings keep their original status so the observation is preserved separately from the final decision.

Both `decision_engine.py` and `benchmark_plate_pipeline.py` accept `--acceptance-min-confidence`. Use 0.80 to reproduce the previous acceptance policy, 0.85 for the intermediate experiment, or the new default 0.90. The result records `acceptance_min_confidence`, and benchmark resume fingerprints include it. Detector `--conf` remains a separate setting.

## All-detector development comparison

The implemented gate was replayed over six immutable full-pipeline reports: 247 Indian images and 20 historical photographs for each of the active detector, first Indian candidate and balanced candidate (801 saved image runs total). Source images and checkpoints were verified against their recorded hashes. The 0.80 replay reproduces the recorded correct/wrong accepted counts exactly. All source-label disagreements below refer to supplied text, not independently adjudicated correctness.

| Detector | Cutoff | Correct Indian acceptances / 247 | Accepted Indian label disagreements | Indian matched review targets | Correct historical acceptances / 25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active current | 0.80 | 144 | 17 | 56 | 16 |
| Active current | 0.85 | 142 | 12 | 63 | 15 |
| **Active current** | **0.90** | **137** | **5** | **75** | **14** |
| First Indian candidate | 0.80 | 157 | 17 | 63 | 13 |
| First Indian candidate | 0.85 | 157 | 15 | 65 | 13 |
| First Indian candidate | 0.90 | 148 | 5 | 84 | 13 |
| Balanced candidate | 0.80 | 152 | 22 | 66 | 15 |
| Balanced candidate | 0.85 | 152 | 18 | 70 | 15 |
| Balanced candidate | 0.90 | 147 | 7 | 86 | 15 |

No model has accepted historical label disagreements at these cutoffs. Missed/unreadable targets and unmatched scene predictions remain separate from review counts; the full comparison contains these and the source/group breakdowns.

For the active detector, the new default moves **12 accepted Indian label disagreements and seven correct readings** to review. Accepted target precision projects from 89.44% to 96.48%; exact accepted target recall projects from 58.30% to 55.47%. It also moves two correct historical readings to review. The policy deliberately increases review workload to reduce incorrect automatic acceptance; it does not improve recognition itself.

The 160 Indian video frames come from only five related groups, transcriptions contain unresolved ambiguities, and historical photographs may overlap legacy stage training. These are development comparisons. Source/group metrics are included to expose repeated-frame dominance. This experiment does not establish independent real-weather reliability, deployment accuracy or a latency improvement.

## Validation and evidence

- `ml/scripts/acceptance_policy.py`: shared pure final gate.
- `ml/tests/test_acceptance_policy.py`: evidence preservation, exact threshold boundary, invalid scores/settings, idempotence, no promotion, and full-pipeline routing/vehicle-association checks.
- `ml/benchmarks/ocr_acceptance_policy/comparison.json`: all six saved-report replays, thresholds, source/group breakdowns and provenance.
- `.runtime_cache/compare_final_acceptance_policy.py`: local reproduction helper using the same gate as inference.
- `ml/benchmarks/ocr_acceptance_policy/fresh_smoke.json`: fresh paired 0.80/0.90 inference on two selected Indian development images with the active detector. This is an integration smoke check, not another accuracy benchmark.

The current detector remains active. Candidate checkpoints and all source labels remain unchanged. No detector training or inference on the reserved 247 Indian test images was started.

**103 ML tests and six backend tests passed.** Both fresh paired-image runs matched the implemented final-decision replay exactly, including unchanged detector boxes, OCR evidence, restoration decisions, quality outputs, vehicle results and stage-call counts. One weak acceptance moved to review and one strong acceptance remained accepted. This verifies integration for those examples, not general correctness.

## Try it

Restart the local app using its normal command so the backend loads the changed pipeline. Existing app requests then use the 0.90 policy.

For an explicit image experiment from the repository root:

```powershell
.\run_ml.ps1 -PythonArguments @(
  'ml/scripts/decision_engine.py', '--image', 'C:/path/to/photo.jpg',
  '--no-upscale', '--acceptance-min-confidence', '0.90'
)
```

Use `--weights` only when intentionally evaluating a candidate. Retain the current active detector for ordinary app use.

## Next work

Review the remaining high-confidence disagreements and measure crop/recognition variants on reviewed development examples. Start the 20-30-photo collection pilot and short clips from the [data-collection plan](ocr-diagnostic-next-steps.md), with all visible plates labeled and repeated vehicle/session observations kept together. Establish independent real-weather results before selecting a detector replacement. Recorded-video tracking and agreement across readings remain the following milestone.
