# OCR fixes and next detector candidate

Preparation completed on 1 October 2026. The literal-reading selection bug is fixed and a separate dataset was prepared for a broader detector experiment. That candidate has now completed 80 epochs; the [2 October training and development results](indian-plates-balanced-results.md) record its measured tradeoffs and next work. The active detector remains `plate_detector3`.

## Reading-selection change

A plausible literal OCR reading now outranks alternatives produced by changing its digits or letters to fit another layout. For example, `PY036993` remains `PY036993` as a review proposal, rather than becoming the automatically accepted `PY03G993`. The historical `KL498262` case similarly stays literal instead of becoming `KL49B262`.

Confident, conflicting literal readings across OCR regions or image views also require review, even when format preference gives one interpretation a much higher selection score. Raw readings, alternatives and the review reason remain available. Low-confidence alternatives do not veto a strong reading, valid one-digit districts still work, and corrections of implausible OCR strings remain possible. These scores are heuristic gates, not calibrated probabilities.

## Completed before/after checks

Both detectors processed the same 247 new validation images and 20 historical development images again, for 534 full-pipeline image runs. The new data supplies 247 target plates; the historical diagnostic supplies 25. Input images, checkpoint hashes, package versions and runtime options were identical to the original comparison; only `plate_validator.py` changed during these runs. No test predictions were generated.

| Diagnostic | Correct accepted, before → after | Accepted source-text disagreements, before → after | Correct review proposals, before → after |
| --- | ---: | ---: | ---: |
| New validation, current detector | 146 → 144 | 19 → 17 | 27 → 32 |
| New validation, Indian candidate | 157 → 157 | 18 → 17 | 27 → 31 |
| Historical, current detector | 16 → 16 | 1 → 0 | 5 → 6 |
| Historical, Indian candidate | 13 → 13 | 1 → 0 | 2 → 3 |

The Indian candidate still accepts the correct complete supplied text on 157/247 targets (63.56%). Its accepted target-reading precision rises from 89.71% to 90.23%; exact proposals rise from 184 to 188, including readings that need review. The current detector moves two previously correct accepted readings into review because its views conflict. That is an explicit tradeoff, not an across-the-board accuracy gain.

Detection coverage is unchanged: the candidate matches 246/247 new targets but only 16/25 historical targets, compared with the current detector's 224/247 and 24/25. The OCR fix does not repair missing detection boxes. The remaining 17 candidate accepted label disagreements include OCR prefix/character failures and previously documented transcription ambiguities. They are not all solved by format selection.

These are development results against supplied targets. Complete scene annotations and all source transcriptions are not independently certified. Historical stages may have seen the old photos. The [original report](indian-plates-pipeline-results.md) remains the before-fix record; new per-image evidence and verified transitions are in `ml/benchmarks/indian_candidate_ocr_literal_fix/comparison.json` and the four paired report files in that directory.

## Broader training data prepared

The isolated configuration is `ml/data/indian_plates/balanced/data.yaml`. It combines selected Indian training images with existing weather derivatives of reviewed historical photos. The original prepared dataset and raw files remain intact.

| Partition | Indian images | Historical derivatives | Total | Related groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 869 | 403 | **1,272** | 666 |
| Validation | 247 | 91 | **338** | 99 |
| Test | 247 | 0 | **247** | 134 |

- Training retains at most 20 images per existing Indian related group, selected by fixed hash ordering. This removes 281 repeated training views without moving any source to another split. Validation and test retain every original Indian example.
- The 403 historical training images represent **31 source photos**, each with one clear and 12 synthetic weather variants. Their reviewed vehicle contexts include 11 motorcycles, six trucks, six cars, three autorickshaws, two buses, one van, one utility vehicle and one mixed scene. Augmented image count does not mean additional independent sources.
- Historical validation uses 13 derivatives each from the original seven validation sources. These remain development examples; synthetic weather is not evidence of performance in actual rain, haze or night scenes.
- The original seven historical test sources are excluded from this dataset. Review identified the same motorcycle across historical train/test photos and matching front/rear views of a utility vehicle with an occluded plate. Their two training photos and all derivatives are also excluded. The historical split is not silently rewritten or presented as independent.
- Cross-dataset checks found no matching reviewed plate identities, exact decoded pixels or near-image hashes between the retained new corpus and the 47 historical originals. The bound curation record stores those checks. Future source changes require a new audit.
- All original Indian images and YOLO labels are copied byte-for-byte into the combined build, including its test partition. Original target coordinates and source grouping remain unchanged. Historical derivatives are already oriented; unhandled EXIF orientation, invalid dimensions, changed pixels or invalid target boxes stop preparation.
- Each build records input/code hashes, exclusions, complete image/label inventories, manifests and training/validation provenance. Publication occurs after validation. All 47 historical source hashes are checked, including excluded photos; every selected derivative is bound to the audit inventory.

The source-box review is recorded in `ml/data/curation/legacy_detector_sources.json`. It is an assistant visual target-box review and conservative identity audit; it does not certify full-scene labels or create new OCR ground truth. Some old scenes omit other visible plates, and `KL34F` is a partial source transcription. The independent full-image evaluator's safeguards remain enabled.

## Completed candidate training command

This is the command used for the completed experiment, retained for reproduction. The latest results do not call for repeating it unchanged. Run from the project root in PowerShell:

```powershell
.\run_ml.ps1 ml/scripts/train_yolo.py --data ml/data/indian_plates/balanced/data.yaml --model yolo11s.pt --epochs 80 --imgsz 640 --batch 4 --workers 0 --name indian_plates_balanced_candidate --no-activate
```

This starts from the local COCO-pretrained YOLO11s weights, with the same image size and batch size as the completed first experiment. It avoids carrying the narrower candidate's training exposure into a dataset that caps repeated views. `train_yolo.py` resolves the YAML to an absolute filename, so the earlier `Desktop/datasets` lookup error is avoided. `--no-activate` preserves the current app detector.

After training, compare detector and full-pipeline behavior on the fixed new validation set and historical diagnostics, and report the Indian and historical validation domains separately. Select checkpoints and any subsequent input-size experiment using development evidence. Keep the completed first candidate's detector-test result as its frozen result; repeated test tuning cannot provide a fresh final evaluation. Collect a separate, fully annotated real-weather set for the final project claim.

To rebuild or validate this preparation:

```powershell
.\run_ml.ps1 ml/scripts/prepare_balanced_plates.py
```

An unchanged build is verified and reused. Source curation is bound to the original trained Indian build and historical manifests; rerunning the Indian importer under a different preparation policy is not part of this experiment.

Validation completed: **89 ML regression tests and six backend tests passed**, all four pipeline diagnostics completed, and model/input hashes remained unchanged.

For the project review:

> We found and fixed a case where OCR post-processing altered a correctly read registration. We verified the change on both development datasets and prepared a broader detector experiment with buses, motorcycles, trucks and synthetic weather examples. Our next steps are to train and compare that candidate, then evaluate complete recognition on independently annotated real adverse-weather data.
