# Review of the new Indian number-plate dataset

Inspected on 30 September 2026. Source supplied by the user: [Saisirishan's Indian vehicle license plate dataset on Kaggle](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset).

**Preparation update:** the raw-download issues below have now been handled in a separate prepared build containing 1,644 detector images and 1,621 eligible supplied OCR transcriptions. See [the preparation record](indian-plates-preparation.md) for repairs, exclusions, grouped splits and the next training command.

## Assessment

This is a useful addition for the plate-detection and OCR parts of the project. Despite the local folder name, its annotations describe number plates and their text, rather than full-vehicle boxes and vehicle categories. It directly supports expanding the small existing ANPR corpus. Its value for the adverse-weather research question still needs separate validation.

The review decoded every image, parsed every XML, checked pairing and dimensions, checked bounding-box bounds, detected exact duplicates, screened text compatibility, and visually inspected 48 representative annotated images, 16 separate plate crops and eight targeted issue examples. Visual transcription correctness has not been checked for every image. No model training or inference evaluation was performed.

## Inventory

| Folder | Images | XML annotations | Matched image/XML records |
| --- | ---: | ---: | ---: |
| `State-wise_OLX` | 602 | 603 | 602 |
| `google_images` | 442 | 440 | 440 |
| `video_images` | 654 | 654 | 654 |
| Total | **1,698** | **1,697** | **1,696** |

The folder contains 3,396 files in total, including one `classes.txt`. Images comprise 1,339 JPG, 306 JPEG and 53 PNG files. Every image decoded and every XML parsed successfully. The file count was unchanged at the end of the audit.

Each matched XML contains one annotated plate box. There are 983 distinct normalized label strings across the 1,696 matched records; this is a label count, not a verified count of independent vehicles or valid registration numbers.

Median annotated plate dimensions are 123 by 37.5 pixels. There are 130 plate boxes narrower than 60 pixels and 378 shorter than 20 pixels. These cases provide useful small-text challenges, subject to ground-truth review.

`video_images` contains extracted still frames, not video files. Filename prefixes such as `video11` identify related groups: `video11` has 185 frames, while other groups contain dozens of frames. These images can support experiments with sequential observations, but the supplied files do not establish original frame rate, timestamps, or complete vehicle-passage boundaries.

## Issues to resolve before training or benchmarking

1. **Duplicate images and repeated observations.** There are 47 groups of byte-identical files and 48 groups of identical decoded images, leaving 1,650 distinct decoded images. Twenty-four additional pairs passed a simple perceptual-similarity screen; those are review candidates rather than confirmed duplicates. Two plate labels occur 41 times each. Deduplicate exact copies and keep related video sequences, vehicle identities and derivative images together when assigning train, validation and test splits. A random split by individual file could make evaluation misleading. Exact duplicate groups with annotations had no conflicting normalized text in this audit.

2. **Special labels are not ordinary registration transcriptions.** Five labels are `TERRANO`, `CRETA`, `DUSTER`, `Devanagri` and `blur`. Visual inspection confirmed vehicle-model display plates, a plate written in another script, and an unreadable small plate. Define separate treatment for detection training, readable registration OCR, display plates and unreadable cases. Do not use these five strings as normal registration-number ground truth.

3. **Two XML/image dimension disagreements.** `State-wise_OLX/AN/AN2.xml` declares 272 by 363 pixels, while its image is 271 by 325. `AN6.xml` declares the same size, while its image is 271 by 306. The boxes are within the actual image bounds and appeared aligned in the targeted overview. Verify the original images and coordinates before repairing the derived manifest; do not automatically rescale boxes from stale header dimensions.

4. **One orphan annotation and one stale filename.** `State-wise_OLX/MH/MH5.xml` has no matching image. `State-wise_OLX/ML/ML39.xml` names `NL1.jpg`, while the same-stem image is `ML39.jpg`; its box and label appeared consistent in the targeted inspection. Record the verified match explicitly in an importer instead of silently trusting every XML filename.

5. **Two extra image variants.** Both JPEG and PNG versions exist for `car-wbs-MH20EE7598_00000` and `car-wbs-TN99F2378_00000`. Their XML filenames explicitly select PNG files. Those two PNGs were matched; the JPEG variants do not have separate matched annotations. Keep the selected source or review both variants without counting them as independent examples.

6. **The existing registration templates do not cover every supplied label.** Eighty-five annotated records, containing 66 distinct strings, do not fit the current `plate_validator.py` templates. Examples include three-letter series and unusual or temporary plate layouts, as well as the special labels above. Unsupported format does not itself mean an annotation is wrong. Review these cases and expand format handling only for verified formats; preserve uncertainty for the others.

7. **Annotations need a completeness check for full-image testing.** One box per image does not prove that every target plate in a multi-vehicle scene has been annotated. The project's full-pipeline evaluator expects complete labeling. Review final test images for missing plate boxes and transcriptions before using them to report missed/extra detections or accepted-reading precision.

The existing raw ANPR folders and previous DataCluster download contained 149 image files during the comparison. No byte-identical or identical-decoded-pixel overlap was found with the new dataset. This check does not rule out resized copies, related views or overlapping acquisition sequences.

## Integration requirements

The supplied XML stores plate text in `<object><name>`, for example `<name>MH01DE2780</name>`. The current `build_manifest.py` reads text only from a `number_plate_text` attribute and uses hard-coded source folders. Applying it unchanged to this dataset would not import the labels correctly.

Build a separate importer and a separate versioned manifest. For each record:

- Read the registration transcription from `<object><name>` and retain its original spelling alongside normalized text.
- Use a single `number_plate` class for plate detection; each registration string is text ground truth, not a separate detector class.
- Read image dimensions from the decoded image and record any reviewed annotation repairs.
- Record file hashes, provenance, related sequence/identity groups, and an explicit split.
- Retain special-script, display-plate and unreadable cases with an explicit status instead of silently guessing transcriptions.

The bundled `google_images/classes.txt` includes unrelated default classes such as dog, food names and cavity, as well as individual plate strings. It is not a suitable vehicle-category or one-class plate-detector mapping.

Leave existing datasets and active checkpoints intact while preparing and evaluating a candidate. Select checkpoints and thresholds on validation data; freeze the final test set before those choices.

## Effect on the November plan

This dataset supplies a substantial candidate collection for the October data-preparation milestone. It can support retraining a plate detector, evaluating OCR across different layouts and small plates, and testing agreement across related frames.

The inspected representative images were predominantly daylight scenes. There are no explicit weather/severity labels or supplied paired clear/degraded targets. Clean image candidates can be selected for synthetic weather augmentation, but final adverse-weather claims still need new labeled real rain, haze and night images. Raw video clips will also be needed for an actual video-input demonstration and passage-level evaluation.

A practical next step is to prepare the reviewed manifest and grouped split, establish a baseline, then train an isolated candidate plate detector. Whether recognition improves must be measured; image count alone does not establish an accuracy gain.

## Evidence files

Generated evidence is stored locally under `.runtime_cache/new_indian_dataset_audit/`:

- `audit.json`: inventory, dimensions, pairings, duplicates and source-overlap findings.
- `records.json`: matched annotation records and hashes.
- `compatibility.json`: current format-template compatibility and duplicate-label conflicts.
- `near_duplicate_pairs.json`: perceptual-similarity candidates.
- `*_overview.jpg`, `*_plate_crops.jpg`, and `issues_overview.jpg`: visual inspection sheets.

The local archive contains no dataset README or license file. The supplied Kaggle URL identifies the source, but its usage terms could not be read through the web tool during this review; no license assumption is recorded here.
