"""
Phase 7 - Step 1: PaddleOCR wrapper -- reads plate text from a crop.

Uses PaddleOCR 3.x's new pipeline API (built on PaddleX), NOT the legacy
2.x `.ocr(img, cls=True)` API -- that legacy engine is incompatible with
PaddlePaddle 3.x's internal rewrite (this is what caused the
"OneDnnContext does not have the input Filter" / fused_conv2d crash).

device="cpu" forced on purpose: paddle's Windows GPU build hit repeated
low-level DLL dependency issues in this environment (missing cublasLt,
then a cuDNN loader bug). OCR on small plate crops is fast enough on CPU
that it's not worth chasing further -- unlike training, this runs once
per crop, not per epoch.

Usage (standalone test):
    python ml/scripts/ocr_plate.py --image path/to/plate_crop.jpg
"""

import argparse
import re

import cv2
import numpy as np

_ocr = None  # lazy-loaded singleton


# ----------------------------------------------------------- preprocessing ---
def preprocess_crop(img_bgr):
    """5a: Pre-process a plate crop before handing it to PaddleOCR.

    Two operations applied in sequence:
      1. Adaptive Gaussian thresholding -- binarises the image using a locally
         computed threshold so uneven illumination (shadow, glare, faded paint)
         doesn't knock out entire regions of the plate.
      2. Sharpening kernel -- amplifies high-frequency edges so stroke
         boundaries between characters are crisper going into the OCR model.

    Returns a 3-channel BGR image so PaddleOCR receives the same data type
    it expects regardless of whether preprocessing ran.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # blockSize=11 and C=2 are conservative defaults that work across a wide
    # range of plate fonts and sizes without over-binarising thin strokes.
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )
    kernel = np.array([[0, -1, 0],
                        [-1,  5, -1],
                        [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(thresh, -1, kernel)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)



def load_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    from paddleocr import PaddleOCR
    # use_doc_orientation_classify / use_doc_unwarping: meant for full
    # document/page images (skewed scans etc.) -- not relevant for tight
    # plate crops, so disabled for speed. use_textline_orientation: our
    # source XML annotations show rotation=0.0 for every plate, so also
    # disabled; flip to True if you start feeding it rotated/angled plates.
    #
    # engine="onnxruntime": Paddle's own native CPU execution path hit an
    # unresolved internal bug (oneDNN/PIR instruction conversion) on both
    # PP-OCRv5 and PP-OCRv6 detection models in this environment --
    # reproducible across model versions, so it's a framework bug, not a
    # model bug. onnxruntime is a separate, more mature CPU inference
    # backend PaddleOCR 3.x can delegate to instead, sidestepping Paddle's
    # broken code path entirely. Falls back to Paddle's native engine if
    # this paddleocr version doesn't support the `engine` kwarg.
    common_kwargs = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        device="cpu",
        ocr_version="PP-OCRv5",
    )
    try:
        _ocr = PaddleOCR(engine="onnxruntime", **common_kwargs)
    except TypeError:
        _ocr = PaddleOCR(**common_kwargs)
    return _ocr


def clean_text(s: str) -> str:
    """Uppercase, strip anything that isn't a letter/digit."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def recognize_plate_candidates(img_bgr):
    """Returns a list of individually detected text regions (each cleaned,
    uppercase alphanumeric), sorted left-to-right by box x-position, plus
    their confidences. This preserves natural OCR token boundaries -- use
    this (not recognize_plate) when feeding into plate_validator, since
    validation needs real region boundaries, not a pre-joined blob.

    5a: img_bgr is pre-processed (adaptive threshold + sharpen) before OCR.
    """
    ocr = load_ocr()
    result = ocr.predict(preprocess_crop(img_bgr))

    items = []  # (x_center, cleaned_text, confidence)
    for res in result:
        rec_texts = res["rec_texts"]
        rec_scores = res["rec_scores"]
        rec_boxes = res["rec_boxes"] if "rec_boxes" in res else None

        for i, raw_text in enumerate(rec_texts):
            cleaned = clean_text(raw_text)
            if cleaned == "IND":
                continue  # the "IND" country stamp isn't part of the registration number
            if not cleaned:
                continue
            conf = float(rec_scores[i])
            if rec_boxes is not None and len(rec_boxes) > i:
                x_center = float(rec_boxes[i][0]) + float(rec_boxes[i][2])
            else:
                x_center = i
            items.append((x_center, cleaned, conf))

    items.sort(key=lambda t: t[0])
    return [{"text": t[1], "conf": t[2]} for t in items]


def recognize_plate(img_bgr):
    """Returns (predicted_text, avg_confidence): all detected regions
    naively joined left-to-right. This is the Phase 7 baseline behavior --
    kept as-is for comparison against plate_validator's smarter region
    scoring (see evaluate_ocr.py).

    5a: img_bgr is pre-processed before OCR (same path as recognize_plate_candidates).
    """
    candidates = recognize_plate_candidates(img_bgr)  # preprocessing applied inside
    if not candidates:
        return "", 0.0
    full_text = "".join(c["text"] for c in candidates)
    avg_conf = sum(c["conf"] for c in candidates) / len(candidates)
    return full_text, avg_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    text, conf = recognize_plate(img)
    print(f"Predicted text: {text}")
    print(f"Confidence:     {conf:.3f}")


if __name__ == "__main__":
    main()