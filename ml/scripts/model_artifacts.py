"""Checkpoint selection and export provenance (no ML imports required)."""

from functools import lru_cache
import hashlib
import json
import warnings
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _cached_digest(path, size, mtime_ns, ctime_ns):
    return sha256_file(path)


def cached_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def activate_detector(weights, ml_dir=ML_DIR):
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    root = Path(ml_dir).resolve()
    try:
        stored = weights.relative_to(root).as_posix()
    except ValueError:
        stored = weights.as_posix()
    atomic_json(root / "models" / "active_detector.json", {
        "weights": stored, "sha256": sha256_file(weights),
    })


def resolve_detector(weights=None, ml_dir=ML_DIR):
    root = Path(ml_dir).resolve()
    if weights is not None:
        selected = Path(weights).resolve()
    else:
        registry = root / "models" / "active_detector.json"
        if registry.exists():
            record = json.loads(registry.read_text(encoding="utf-8"))
            selected = (root / record["weights"]).resolve()
            if not selected.is_file() or cached_sha256(selected) != record["sha256"]:
                raise ValueError("Active detector changed or is missing; select it explicitly with --weights.")
        else:
            candidates = list((root / "models" / "yolo_runs").glob("*/weights/best.pt"))
            if not candidates:
                raise FileNotFoundError("No detector checkpoint. Download/train weights or pass --weights.")
            selected = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
            warnings.warn(f"No active detector recorded; using newest checkpoint: {selected}. "
                          "Use --weights to select another checkpoint.", stacklevel=2)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    return selected


def export_is_current(checkpoint, scripted):
    scripted = Path(scripted)
    metadata = scripted.with_suffix(".json")
    if not scripted.is_file() or not metadata.is_file():
        return False
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return (record["checkpoint_sha256"] == sha256_file(checkpoint)
                and record["scripted_sha256"] == sha256_file(scripted))
    except (ValueError, KeyError, OSError):
        return False
