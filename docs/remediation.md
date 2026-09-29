# Ranked review remediation

This records the changes made for the 13 findings in the project review. The review found no confirmed critical vulnerability. Code fixes do not establish model accuracy; the outstanding empirical validation is listed below.

| Rank | Severity | Finding | Change | Verification |
| --- | --- | --- | --- | --- |
| 1 | High | Old detector used after retraining | `model_artifacts.py` records the actual best checkpoint with a hash; inference accepts `--weights`; legacy installs discover the newest checkpoint with a warning. Training uses the real run directory. | Explicit, registered, newest-legacy and modified-checkpoint regression checks. Existing local `plate_detector3` is activated. |
| 2 | High | Restoration destroys high-resolution plate detail | `image_ops.tiled_restore` pads and blends native-resolution tiles. Original detections and OCR remain available alongside restored and upscaled views. | Pixel-exact identity reconstruction on small, large and nonsquare images; pipeline fallback test. Actual restoration quality still needs benchmarking. |
| 3 | High | Format score overrides recognition reliability | `select_candidate` uses OCR confidence, correction cost, confidence thresholds and competing readings. Accepted text is empty for uncertain/unreadable results. No-series formats require review. | Non-plate, low-confidence, competing text, missing-series and cross-view disagreement checks. |
| 4 | High | No independent full-pipeline evaluation | `evaluate_pipeline.py` evaluates full images, rejects training/model-selection overlap, counts missed and extra detections, and reports exact-reading recall and accepted-reading precision by weather. Training stages share a source-level train/validation/test split and checkpoint provenance. | Metric-denominator, one-to-one matching and overlap checks. The real legacy model is correctly rejected on the newly partitioned old corpus. **Independent real-weather accuracy remains unmeasured.** |
| 5 | High | Unsupported residual-training flag | `--residual` and `--no-residual` configure the model. Residual mode defaults on; architecture and loss settings are saved in the checkpoint. | Documented CLI flags parse correctly. A subsequent CUDA mixed-precision training smoke check passed; see optimization.md. |
| 6 | High | Dataset rebuild can leak images between splits | Each build has an isolated directory; a new YAML is published only after successful preparation. Existing builds are preserved. | Changed-fraction rebuild and failed-rebuild checks. Actual rebuild: 429/91/91 images from 33/7/7 disjoint source photos. |
| 7 | Moderate | Two-row OCR ordering | OCR retains full boxes. Vertical-overlap row grouping precedes left-to-right reading within each row. | Three-region, two-row plate regression and missing-geometry fallback checks. |
| 8 | Moderate | Reverse confusion mapping discards O/Q/D alternatives | Substitutions preserve all alternatives. Equally supported alternatives cause abstention. | Ambiguous `0D01AB1234` retains `OD01AB1234` as an alternative and is not silently accepted as `DD01AB1234`. |
| 9 | Moderate | Stale TorchScript model used after retraining | Export metadata fingerprints checkpoint and script. Inference rejects mismatched or unverified exports; export is published after numerical parity validation. | Missing-sidecar and changed-checkpoint checks. Numerical export validation is built into the export command. |
| 10 | Moderate | SSIM silently disabled | `torchmetrics` is declared. Positive SSIM weight with a missing dependency raises an error; zero explicitly disables it. Logs and checkpoints record actual weights. | Required-dependency failure and explicit-opt-out checks. |
| 11 | Moderate | Severity ignored when selecting checkpoint | Quality training uses harmonic mean of both heads' validation accuracy, with validation loss as tie-breaker. Both heads' metrics are saved and plotted. | Severity regression reduces the checkpoint selection score, even if condition accuracy rises. |
| 12 | Moderate | Setup order and paths are not portable | README builds manifest before augmentation and includes the BasicSR patch. Manifests use forward slashes and stable source IDs/fingerprints; YOLO paths are relative to its YAML. | Portable-path checks, actual dataset rebuild, and Windows/Linux CI configuration. Fresh full dependency installation has not been verified. |
| 13 | Moderate | Inference requests unnecessary pretrained weights | Quality inference constructs `QualityNet(pretrained=False)` before loading the checkpoint. | Loader integration check verifies no pretrained-backbone request. |

## Validation performed

- Regression suite: **41 tests passed** using `python -m unittest discover -s ml/tests -v`.
- Syntax parsing of all project scripts and tests.
- Git whitespace/error check.
- Real manifest and image fingerprint validation; no exact duplicate raw-image files were found across the 47 source photographs.
- Real YOLO dataset rebuild into a new directory; previous data remains intact.
- Active detector resolution confirmed as `plate_detector3/weights/best.pt`.
- Independent-test guard confirmed to reject the existing detector on the old corpus, including photos newly assigned to the test partition.

The pipeline integration tests use deterministic model/backend doubles. They verify routing and data flow, not learned model quality or third-party binary compatibility.

## Remaining empirical work

The old `ml/venv` launcher targets a missing Microsoft Store Python. A local Python 3.11.9 runtime now runs the existing compatible packages. Full-image neural inference was benchmarked, and a small CUDA training smoke check passed with SSIM explicitly disabled. No trained weights were replaced, and numerical export validation has not been rerun. See [optimization measurements](optimization.md).

Independent real-weather testing still requires new labeled images or retraining every pipeline model while holding out the shared test split. Legacy checkpoints cannot be made independent by relabeling the data split.

Native-resolution tiling changes the input distribution relative to the original resized-scene restoration training. The original-image fallback limits the impact of destructive restoration, but tiling quality, acceptance thresholds, false acceptances, latency and real-weather generalization still need measurement. Record a dependency lock only after that runtime has passed its smoke checks.
