# Accuracy and efficiency changes

Measured on September 28, 2026 with unchanged project weights, Python 3.11.9, PyTorch 2.13.0+cu130 and an NVIDIA RTX 3050 6GB Laptop GPU. PaddleOCR uses ONNX Runtime on CPU. Both runs disabled Real-ESRGAN because its weights were unavailable locally.

| Measurement | Before | Optimized balanced profile |
| --- | ---: | ---: |
| Images / annotated plates | 20 / 25 | 20 / 25 |
| Correct accepted plate readings | 15 | 16 |
| Exact end-to-end recall | 60% | 64% |
| Incorrect accepted readings | 1 | 1 |
| Detected / missed plates | 24 / 1 | 24 / 1 |
| Accepted-reading precision | 93.75% | 94.12% |
| Warm mean seconds per image | 2.140 | 0.835 |
| First-image seconds | 17.274 | 5.932 |

Warm latency decreased about 61% (2.56 times the throughput). These are single sequential runs on the same small development corpus, which legacy checkpoints may have seen. Warm means exclude the first image; first-image timing includes lazy initialization but is not a controlled cold-cache measurement. No independent accuracy or real-weather claim follows from these results. Seven detections still abstained, one plate was missed, and one incorrect reading remained accepted.

## Changes

- Bound OCR text detection to 640 pixels and ONNX execution to four intra-op threads and one inter-op thread.
- Decode each image once, cache checkpoint digests with file-change invalidation, eliminate redundant edge tiles, and batch restoration tiles on GPU.
- Read original crops first; skip optional work on strong, unambiguous readings under the balanced profile, except confidently severe weather. Exhaustive mode retains relevant restoration and additional detection. Balanced mode can miss additional plates if already-found plates read strongly.
- Preserve crop context; try bicubic and contrast variants for weak OCR, and retry detection at a bounded larger size when the first pass finds no plates. Acceptance and abstention remain evidence-based; model confidence is not calibrated correctness.
- Train restoration with aligned native-pixel patches, plate-focused sampling, clear identity examples, mixed precision, efficient loading and early stopping. These training changes have not yet produced replacement weights or a measured accuracy gain.

## Validation and reproduction

All 41 regression tests passed in the actual Python 3.11 runtime. A CUDA smoke check used two real haze pairs, 128-pixel patches, a small base-8 residual U-Net, mixed precision and L1/Sobel loss. Training and validation losses were finite and parameters changed. SSIM was explicitly disabled because torchmetrics is missing from the local legacy packages. This smoke check does not establish convergence or the memory requirements of full-size training.

```powershell
.\run_ml.ps1 -m unittest discover -s ml/tests
.\run_ml.ps1 ml/scripts/benchmark_pipeline.py --manifest ml/data/processed/manifest.json --no-upscale --profile balanced --out ml/benchmarks/current.json
# Compare exhaustive routing on the same data:
.\run_ml.ps1 ml/scripts/benchmark_pipeline.py --manifest ml/data/processed/manifest.json --no-upscale --profile exhaustive --out ml/benchmarks/exhaustive.json
```

The diagnostic deliberately includes only fully transcribed positive images, so it does not measure false positives on negative images. Use `evaluate_pipeline.py` with new, independently labeled full images (including negatives) for deployment decisions. That evaluator retains overlap rejection and records the selected profile and detector size.

The [retained summary](benchmarks/optimization-summary.json) includes metrics, code and manifest fingerprints, and the training smoke result. Raw per-image reports remain under the ignored `ml/benchmarks/` directory. Existing trained weights were preserved.
