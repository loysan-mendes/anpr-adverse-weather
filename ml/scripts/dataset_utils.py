"""Portable manifests and one deterministic source-photo split for every stage."""

import json
import random
from pathlib import Path, PurePosixPath
from model_artifacts import sha256_file, atomic_json

ML_DIR = Path(__file__).resolve().parents[1]


def portable_path(value):
    return str(value).replace("\\", "/")


def image_path(entry, root=ML_DIR):
    return Path(root) / portable_path(entry["image_path"])


def source_id(entry):
    return entry.get("source_id") or (
        entry.get("orig_source", entry.get("source", "unknown")) + ":"
        + PurePosixPath(portable_path(entry["image_path"])).stem)


def load_manifest(path):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest must be a nonempty list: {path}")
    for entry in entries:
        entry["image_path"] = portable_path(entry["image_path"])
    return entries


def split_entries(entries, val_frac=0.15, test_frac=0.15, seed=42):
    if not 0 < val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Require 0 < val_frac, 0 <= test_frac and val_frac + test_frac < 1")
    explicit = ["split" in e for e in entries]
    if any(explicit) and not all(explicit):
        raise ValueError("Every entry must have a split when explicit splits are used")
    assignment = {}
    if all(explicit):
        for entry in entries:
            key, split = source_id(entry), entry["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split: {split}")
            if key in assignment and assignment[key] != split:
                raise ValueError(f"Source photo crosses splits: {key}")
            assignment[key] = split
    else:
        keys = sorted({source_id(e) for e in entries})
        random.Random(seed).shuffle(keys)
        n_val = max(1, int(len(keys) * val_frac))
        n_test = max(1, int(len(keys) * test_frac)) if test_frac else 0
        if n_val + n_test >= len(keys):
            raise ValueError("Too few source photos for nonempty train/validation/test splits")
        assignment = {key: "val" if i < n_val else "test" if i < n_val+n_test else "train"
                      for i, key in enumerate(keys)}
    splits = {name: [] for name in ("train", "val", "test")}
    hashes = {}
    for entry in entries:
        split = assignment[source_id(entry)]
        for digest in (entry.get("source_sha256"), entry.get("image_sha256")):
            if digest and digest in hashes and hashes[digest] != split:
                raise ValueError("Duplicate image content crosses splits")
            if digest:
                hashes[digest] = split
        splits[split].append({**entry, "source_id": source_id(entry), "split": split})
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training and validation splits must both be nonempty")
    return splits


def provenance(entries, root=ML_DIR):
    """Record every image used for fitting OR model selection, excluding test."""
    seen = [e for e in entries if e.get("split") != "test"]
    return {"source_ids": sorted({source_id(e) for e in seen}),
            "image_sha256": sorted({sha256_file(image_path(e, root)) for e in seen}),
            "source_sha256": sorted({e["source_sha256"] for e in seen if e.get("source_sha256")})}


def write_provenance(checkpoint, entries, root=ML_DIR):
    record = provenance(entries, root)
    record["checkpoint_sha256"] = sha256_file(checkpoint)
    atomic_json(Path(checkpoint).with_suffix(".provenance.json"), record)
