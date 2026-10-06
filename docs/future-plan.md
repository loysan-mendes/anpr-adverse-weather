# Future plan: ANPR in adverse weather

Prepared on 30 September 2026. Next major milestone: 30 November 2026.

## Direction

Develop the current photograph-based research prototype into a tested, weather-aware vehicle-entry monitoring prototype for a campus gate or parking entrance. It should identify a vehicle, read its plate when the evidence is sufficient, combine observations across video frames, and send uncertain readings for human review.

The November deliverable should be a measured prototype using recorded video. A live camera demonstration is a stretch goal after that works reliably. This scope builds on the existing project and gives it a concrete application.

## What already exists

The literal-reading selection fix and broader detector training are complete. The [2 October candidate results](indian-plates-balanced-results.md) recover historical detection coverage but show mixed complete-reading behavior. OCR/crop reliability, acceptance calibration and independent real-weather evaluation are the next research milestones, followed by recorded-video tracking and reading agreement.

The [crop diagnostic and final-confidence replay](ocr-diagnostic-next-steps.md) now identify the next experiment: test a stricter final acceptance gate while retaining doubtful text as a review proposal. The team can start a 20-30-photo collection pilot alongside that work, then expand the independently labeled real-weather collection.

That [final gate is now implemented and compared across all three detectors](ocr-acceptance-policy-results.md), with a 0.90 prototype default. The active detector remains unchanged. The immediate work is to review remaining recognition failures and begin the collection pilot; the stricter policy reduces automatic acceptance while increasing human review.

The repository implements image-quality and severity assessment, selective restoration for supported degradations, plate detection, OCR, conservative reading selection, and a local image-upload interface. It also detects cars, motorcycles, buses and trucks and associates plates with vehicle boxes in individual images. These vehicle IDs currently do not track vehicles across frames.

The README records 47 source photographs, 611 synthetic variants and 25 transcribed plate boxes. Synthetic variants provide additional training examples but do not represent hundreds of independent scenes.

The retained September 28 development diagnostic correctly accepted 16 of 25 plates across 20 images, missed one plate and accepted one incorrect reading. That is 64% exact end-to-end recall on that diagnostic. Its approximately 94% accepted-reading precision describes only accepted predictions; it is not overall recognition accuracy. The diagnostic is small and may overlap historical model training, so independent real-weather performance still needs to be established.

The DataCluster vehicle experiment supplies vehicle-category boxes, without plate boxes or plate transcriptions. It supports vehicle-detection experiments but does not replace an ANPR evaluation dataset.

A dataset added on 30 September supplies 1,698 number-plate images and 1,697 XML annotations, including extracted video frames and plate text. Reviewed preparation now provides 1,644 detector images, grouped into 1,150 training, 247 validation and 247 test examples, with separate eligible OCR manifests. Repairs and exclusions are recorded in [the preparation record](indian-plates-preparation.md). An isolated detector candidate completed 80 epochs, with reported validation mAP50 of 99.49% and mAP50-95 of 82.66%; see [the training results](indian-plates-training-results.md). These are target-plate detection validation scores. Independent complete-pipeline and real adverse-weather evaluation remain necessary.

The 1 October [full OCR pipeline development comparison](indian-plates-pipeline-results.md) increased exact accepted target recall on the new validation set from 59.11% to 63.56%, but the historical diagnostic fell from 16/25 to 13/25 correct accepted readings. The current detector remains active. Reading-selection ambiguity and training coverage for small plates, multiple vehicles and different vehicle types are now concrete priorities for the next experiment.

## Plan through 30 November

| Dates | Work | Reviewable deliverable |
| --- | --- | --- |
| 1-14 October | Collect and label new photographs and a few short videos. Audit existing plate labels. | A documented dataset, grouped train/validation/test split, and a frozen independent test set. |
| 15-28 October | Establish an original-image baseline, compare restoration policies, inspect failures and improve the weakest component. Tune acceptance thresholds on validation data. | Comparison tables by condition, examples of failures, and one justified model or policy improvement. |
| 29 October-11 November | Add recorded-video processing, vehicle tracking and agreement across plate readings from the same track. | An annotated video and one proposed or accepted plate result per vehicle passage. |
| 12-22 November | Add a local event log and review workflow. Evaluate vehicle-to-plate matching and video events on held-out clips. | A campus-entry demonstration with timestamps, vehicle types, evidence crops and review status. |
| 23-30 November | Freeze the final version, run the held-out evaluation, prepare the report, presentation and demonstration. | Reproducible results, an honest limitations section, and a demonstration that can run during the review. |

These dates are proposed work windows. Data collection and annotation can continue alongside implementation, while the test set remains untouched by training and tuning.

## Priority 1: establish reliable evidence

Start with a pilot collection target of roughly 200-300 new source photographs, adjusted to what the team can collect and label carefully. This is a planning target, not a claim that that many photographs establish general reliability. Include clear daylight, available real rain or haze, night lighting, motion blur, distant plates, angled plates and scenes with multiple vehicles. Include negative images with no plates to expose false detections.

Annotate every visible target plate with its box and exact text, along with condition and severity. Record genuinely unreadable text separately instead of guessing a ground-truth transcription. Such plates can contribute to detection evaluation but need a defined exclusion policy for exact-text scoring; the current full-pipeline evaluator requires a transcription for every annotated test plate.

Keep photographs of the same vehicle passage and frames from the same clip together. Prefer grouping by collection session, camera and location where practical. All synthetic versions stay with their source photograph. New real-weather test images must remain separate from training, checkpoint selection and threshold tuning.

For the video evaluation, label vehicle passages, their plate text when readable, and their associated vehicle type. Separate collection sessions for development and final testing. Collect footage through access the team has permission to use.

Measure:

- Detection recall: how many labeled plates the detector finds.
- Exact end-to-end recall: how many labeled readable plates the whole system correctly accepts, including missed detections in the denominator.
- Accepted-reading precision: how many automatically accepted results are actually correct.
- Review rate and incorrect accepted readings: how often the system abstains, and how often it is confidently wrong.
- Mean and slow-case processing time on the actual project machine, reporting first-request loading separately.

Report these by condition and severity with the number of examples in each group. The existing `evaluate_pipeline.py` provides much of the static-image evaluation; video-event and vehicle-association evaluation require additional work.

## Priority 2: make the research question explicit

Proposed research question:

> Does weather-dependent restoration improve exact Indian plate recognition compared with reading the original image, and can selective processing retain those gains with less computation?

Compare the same detector and OCR stack under three policies:

1. Read original images without restoration.
2. Apply the appropriate available restoration or low-light enhancement before recognition on every applicable image.
3. Use the current adaptive policy, comparing original and processed evidence when needed.

Use the same held-out images, annotated truth, hardware and reporting rules. Select thresholds and candidate checkpoints using validation data before final testing. The current balanced and exhaustive modes both contain conditional behavior; they are useful comparisons, but neither alone supplies a strict original-only baseline or an always-process baseline. Add explicit experiment controls for those comparisons.

Choose improvements from observed failures: missed small plates, poor crops, character confusions, night glare, or vehicle-association errors. Retraining restoration using the implemented plate-focused native-patch training is a candidate experiment, not an already measured improvement. A restoration result should earn its place by improving correct plate reads, even if another image looks cleaner.

The project contribution can be the measured combination of weather-aware processing, preserving original evidence and abstaining on ambiguous readings. Establishing novelty beyond existing work requires a literature comparison; using YOLO and OCR alone is not a research novelty claim.

## Priority 3: use video as additional evidence

Begin with uploaded or local recorded clips. Track each vehicle through its passage, associate its plates, and read selected clear crops rather than running the entire expensive pipeline on every frame.

For example, the same vehicle could produce `KA01AB1234`, `KA01A81234` and `KA01AB1234` in successive observations. Agreement can support the first reading. Conflicting or consistently weak evidence should still trigger review. Repeated wrong readings are possible, so voting must be evaluated rather than treated as proof of correctness.

Ultralytics documents video tracking with persistent object identities and trackers including ByteTrack and BoT-SORT. That makes tracking a practical extension of the existing YOLO-based system. Check the installed library version when implementing it. [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track).

License-plate research has also evaluated fusion of OCR predictions from multiple sequential images and reported improvements in its own benchmark. This supports testing temporal agreement here; those results are not accuracy estimates for Indian plates or this project. [Nascimento et al., 2025](https://arxiv.org/abs/2505.06393).

Measure correct plate readings per vehicle passage, incorrect accepted events, missed passages and duplicate events. Also record how long it takes to reach an accepted reading. Tracking IDs are temporary video identities, not verified registration identities.

## Priority 4: demonstrate a useful workflow

Build a local campus-entry event view containing timestamp, plate result, vehicle type, evidence crop, weather assessment and accepted/review status. Log one event per passage and let a reviewer confirm or correct uncertain text, preserving the original proposal.

A local SQLite database is sufficient for this prototype. Keep evidence for a defined short period and allow deletion. Support CSV export so the result can be shown and assessed during the project review.

An optional approved-vehicle list can use entries supplied by the campus for a demonstration. Plate text matching does not verify the vehicle owner's identity. Any gate-opening integration should be a later stage after identification and review behavior have been validated.

## Scope choices

Core November commitments:

- New labeled data and an independent full-image evaluation.
- A controlled comparison showing whether restoration helps plate reading.
- A recorded-video prototype that combines readings per tracked vehicle.
- A local entry log with human review and a final report.

Stretch goals after the core works:

- Live camera input, beginning with one fixed camera.
- Model export and speed measurements on a smaller device.
- Additional vehicle categories or plate layouts backed by new labels.
- A dashboard showing performance trends across weather conditions.

If video processing takes longer than expected, finish the independently tested image system and recorded-video proof of concept. Keep live streaming as future work. Promise measured improvements and deliverables rather than an unsupported accuracy percentage or full-speed real-time operation.

## Answer for the next meeting

> Ma'am, our next goal is to develop this into a weather-aware vehicle-entry monitoring prototype. We already have image restoration, plate recognition and vehicle classification. First, we will expand our real-world dataset and test the complete pipeline on unseen images. We will compare recognition with and without restoration to measure its actual benefit. Then we will extend the system to recorded video, track each vehicle and combine plate readings across frames. By 30 November, we aim to demonstrate a campus-gate or parking-entry prototype with an entry log and human review for uncertain readings. Live CCTV integration and deployment on a smaller device are our longer-term extensions.

If asked what makes it different:

> We are investigating when weather restoration actually helps recognition and when reading the original image is sufficient. We will measure correctness, wrong accepted readings and processing time, then test whether agreement across video frames improves the result. Our claim will be based on those comparisons.

Immediate team actions: agree on the campus-entry use case, assign data collection and annotation, define the dataset and split rules, and schedule the first baseline-results review for 28 October.

Project evidence: [README](../README.md), [optimization diagnostic](optimization.md), [vehicle dataset experiment](vehicle-dataset-experiment.md).
