"""Resolution-preserving restoration and box geometry."""


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length-tile_size+1, tile_size-overlap))
    if starts[-1] != length-tile_size:
        starts.append(length-tile_size)
    return starts


def tiled_restore(image, predict_tile, tile_size, overlap=32, predict_batch=None, batch_size=1):
    import numpy as np
    if image.ndim != 3 or not image.size or tile_size <= 0 or batch_size < 1 or overlap < 0:
        raise ValueError("Expected nonempty HWC image, positive tile/batch size and nonnegative overlap")
    overlap = min(overlap, tile_size // 4)
    height, width = image.shape[:2]
    output = np.zeros(image.shape, dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    ramp = np.minimum(np.arange(tile_size)+1, np.arange(tile_size, 0, -1))
    ramp = np.minimum(ramp / max(overlap, 1), 1).astype(np.float32)
    window = (ramp[:, None] * ramp[None, :])[..., None]
    positions = [(y, x) for y in tile_starts(height, tile_size, overlap)
                 for x in tile_starts(width, tile_size, overlap)]
    for offset in range(0, len(positions), batch_size):
        group = positions[offset:offset+batch_size]
        tiles, shapes = [], []
        for y, x in group:
            tile = image[y:y+tile_size, x:x+tile_size]
            h, w = tile.shape[:2]
            shapes.append((h, w))
            tiles.append(np.pad(tile, ((0, tile_size-h), (0, tile_size-w), (0, 0)), mode="edge"))
        predictions = predict_batch(np.stack(tiles)) if predict_batch else [predict_tile(tile) for tile in tiles]
        if len(predictions) != len(tiles):
            raise ValueError("Wrong restoration batch length")
        for (y, x), (h, w), tile, restored in zip(group, shapes, tiles, predictions):
            restored = np.asarray(restored, dtype=np.float32)
            if restored.shape != tile.shape or not np.isfinite(restored).all():
                raise ValueError("Restoration returned invalid pixels or shape")
            weights = window[:h, :w]
            output[y:y+h, x:x+w] += restored[:h, :w] * weights
            weight[y:y+h, x:x+w] += weights
    return np.rint(output / weight).clip(0, 255).astype(np.uint8)


def padded_box(box, width, height, fraction=0.04):
    x1, y1, x2, y2 = box
    dx, dy = max(2, round((x2-x1)*fraction)), max(2, round((y2-y1)*fraction))
    return [max(0, x1-dx), max(0, y1-dy), min(width, x2+dx), min(height, y2+dy)]


def aligned_patch(degraded, clean, size, boxes=(), rng=None):
    """Crop paired native pixels; emphasize plates during training, no resizing."""
    import numpy as np
    if degraded.shape != clean.shape or degraded.ndim != 3 or size <= 0:
        raise ValueError("Restoration pairs must have matching HWC shapes and positive patch size")
    height, width = degraded.shape[:2]
    if not height or not width:
        raise ValueError("Empty restoration pair")
    if boxes and (rng is None or rng.random() < 0.7):
        box = boxes[0] if rng is None else rng.choice(boxes)
        cx = (box["xmin"]+box["xmax"])/2
        cy = (box["ymin"]+box["ymax"])/2
        if rng is not None:
            cx += rng.uniform(-size/8, size/8)
            cy += rng.uniform(-size/8, size/8)
        x, y = round(cx-size/2), round(cy-size/2)
    elif rng is not None:
        x, y = rng.randint(0, max(0, width-size)), rng.randint(0, max(0, height-size))
    else:
        x, y = (width-size)//2, (height-size)//2
    x, y = min(max(0, x), max(0, width-size)), min(max(0, y), max(0, height-size))
    def crop(image):
        patch = image[y:y+size, x:x+size]
        return np.pad(patch, ((0, size-patch.shape[0]), (0, size-patch.shape[1]), (0, 0)), mode="edge")
    return crop(degraded), crop(clean)


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area2 = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return intersection / max(area1 + area2 - intersection, 1e-12)


def merge_detections(detections, iou_threshold=0.5):
    kept = []
    for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if not any(box_iou(detection["box"], other["box"]) >= iou_threshold for other in kept):
            kept.append(detection)
    return kept


def sharpen_unsharp_mask(img_bgr, sigma=1.0, strength=1.5):
    """Amplify high-frequency edges by subtracting a Gaussian-blurred version.
    
    Counters motion and optical defocus blur on license plate character strokes.
    """
    import cv2
    import numpy as np
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_plate_strokes(img_bgr):
    """Morphological Black-Hat filter to isolate dark character strokes on reflective plate surfaces.
    
    Helps separate smeared or low-contrast characters from plate background blur.
    """
    import cv2
    import numpy as np
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.subtract(gray, blackhat)
    norm = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

