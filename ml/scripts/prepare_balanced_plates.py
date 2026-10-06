"""Prepare a separate detector candidate dataset; never train or activate it.

Keep the Indian dataset's validation and test partitions fixed. Limit repeated
training views per related group and add reviewed historical weather variants.
Historical held-out identities and cross-partition matches are excluded.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from dataset_utils import image_path, load_manifest, provenance, split_entries
from model_artifacts import atomic_json, sha256_file
from prepare_indian_plates import validate_build as validate_indian_build
from prepare_yolo_dataset import voc_to_yolo_line

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ML_DIR / 'data/indian_plates/balanced'
DEFAULT_POLICY = ML_DIR / 'data/curation/legacy_detector_sources.json'


def select_entries(indian, originals, augmented, policy, max_train_group=20):
    """Select by identity and fixed ordering, without using model predictions."""
    if max_train_group < 1:
        raise ValueError('Training group limit must be positive')
    indian_splits = split_entries(indian)
    old_splits = split_entries(originals)
    original_by_id = {e['source_id']: e for values in old_splits.values() for e in values}
    reviews = {r['source_id']: r for r in policy['reviews']}
    if len(reviews) != len(policy['reviews']) or set(reviews) != set(original_by_id):
        raise ValueError('Every historical source needs exactly one bound review')
    for sid, entry in original_by_id.items():
        r = reviews[sid]
        if r['source_sha256'] != entry['source_sha256'] or r['split'] != entry['split']:
            raise ValueError('Historical review no longer matches the source split')
        if r['box_review'] != 'supplied_boxes_visually_locate_plates':
            raise ValueError('Historical target boxes need visual review')
    group_splits = defaultdict(set)
    for r in reviews.values():
        group_splits[r['leakage_group']].add(r['split'])
    excluded, eligible = [], {}
    for sid, r in reviews.items():
        boundaries = group_splits[r['leakage_group']]
        matches = r.get('cross_dataset_matches', [])
        if 'test' in boundaries:
            reason = 'reserved_historical_test_identity'
        elif len(boundaries) != 1:
            reason = 'historical_identity_crosses_partitions'
        elif any(m['split'] != r['split'] for m in matches):
            reason = 'cross_dataset_identity_crosses_partitions'
        elif len({m['leakage_group'] for m in matches}) > 1:
            reason = 'cross_dataset_group_needs_resolution'
        else:
            eligible[sid] = {**r, 'leakage_group': matches[0]['leakage_group'] if matches else r['leakage_group']}
            continue
        excluded.append({'source_id': sid, 'original_split': r['split'], 'reason': reason})
    training_groups = defaultdict(list)
    for e in indian_splits['train']:
        training_groups[e['leakage_group']].append(e)
    chosen, removed = [], []
    for group in sorted(training_groups):
        values = sorted(training_groups[group], key=lambda e: hashlib.sha256(
            ('balanced-sampling-v1:' + e['image_path']).encode()).hexdigest())
        chosen.extend({**e, 'training_domain': 'indian_source'} for e in values[:max_train_group])
        removed.extend(e['source_id'] for e in values[max_train_group:])
    # Retain every Indian validation/test example, in its original partition.
    for split in ('val', 'test'):
        chosen.extend({**e, 'training_domain': 'indian_source'} for e in indian_splits[split])
    seen_variants = set()
    variants_by_source = defaultdict(set)
    for e in augmented:
        sid = e['source_id']
        if sid not in original_by_id:
            raise ValueError('Augmentation has an unknown source')
        if e['source_sha256'] != original_by_id[sid]['source_sha256']:
            raise ValueError('Augmentation source hash changed')
        key = (sid, e['condition'], e['severity'])
        if key in seen_variants:
            raise ValueError('Repeated historical weather variant')
        seen_variants.add(key)
        if sid not in eligible:
            continue
        r = eligible[sid]
        variants_by_source[sid].add((e['condition'], e['severity']))
        chosen.append({**e, 'split': r['split'], 'leakage_group': r['leakage_group'],
                       'image_sha256': policy['augmented_image_sha256'][e['image_path']],
                       'source_pixel_sha256': r['pixel_sha256'], 'training_domain': 'legacy_synthetic',
                       'vehicle_context': r.get('vehicle_context', 'unspecified'),
                       'annotation_scope': 'provided_target_only', 'full_image_labels_verified': False,
                       'weather_labels_verified': False, 'weather_origin': 'synthetic',
                       'target_box_review': r['box_review']})
    expected = {('clear', 'none')} | {(c, s) for c in ('haze', 'rain', 'blur', 'lowlight')
                                    for s in ('mild', 'moderate', 'severe')}
    if any(variants_by_source[sid] != expected for sid in eligible):
        raise ValueError('Each included historical source needs its complete 13-variant set')
    splits = split_entries(chosen)
    assert [e['source_id'] for e in splits['test']] == [e['source_id'] for e in indian_splits['test']]
    return splits, {'excluded_historical_sources': excluded,
                    'training_views_omitted_by_group_cap': len(removed),
                    'omitted_training_source_ids': removed,
                    'historical_unique_sources': dict(Counter(r['split'] for r in eligible.values()))}


def preflight(entries, ml_dir=ML_DIR):
    from PIL import Image
    for e in entries:
        path = image_path(e, ml_dir)
        if sha256_file(path) != e['image_sha256']:
            raise ValueError(f'Input image hash changed: {path}')
        with Image.open(path) as im:
            im.load()
            if im.size != (e['width'], e['height']) or im.getexif().get(274, 1) not in (0, 1):
                raise ValueError(f'Use normalized, oriented source derivatives: {path}')
        if not e['boxes']:
            raise ValueError('Positive source targets must not disappear')
        for b in e['boxes']:
            values = [b[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            if not all(math.isfinite(v) for v in values) or not (
                    0 <= b['xmin'] < b['xmax'] <= e['width'] and
                    0 <= b['ymin'] < b['ymax'] <= e['height']):
                raise ValueError(f'Invalid target geometry: {path}')
            if voc_to_yolo_line(b, e['width'], e['height']) is None:
                raise ValueError(f'Degenerate target geometry: {path}')


def validate_build(build, ml_dir=ML_DIR):
    entries = load_manifest(build/'manifest.json')
    splits = split_entries(entries)
    preflight(entries, ml_dir)
    inventory = json.loads((build/'files.sha256.json').read_text())
    # Ultralytics writes these derived indexes during training/validation.
    # They are not source images or labels and never replace inventory checks.
    derived_caches = {'labels/train.cache', 'labels/val.cache', 'labels/test.cache'}
    actual = {p.relative_to(build).as_posix() for p in build.rglob('*') if p.is_file()
              and p.name != 'files.sha256.json' and p.relative_to(build).as_posix() not in derived_caches}
    if actual != set(inventory):
        raise ValueError('Prepared inventory changed')
    for relative, digest in inventory.items():
        if sha256_file(build/relative) != digest:
            raise ValueError(f'Prepared artifact changed: {relative}')
    for e in entries:
        path = image_path(e, ml_dir)
        label = build/'labels'/e['split']/(path.stem+'.txt')
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        if len(rows) != len(e['boxes']):
            raise ValueError('Prepared target count changed')
        for row, box in zip(rows, e['boxes']):
            expected = [float(v) for v in voc_to_yolo_line(box, e['width'], e['height']).split()[1:]]
            if len(row) != 5 or row[0] != '0' or any(
                    not math.isfinite(float(v)) or abs(float(v)-x) > 1e-6 for v, x in zip(row[1:], expected)):
                raise ValueError('Prepared labels no longer match the manifest')
        if e.get('input_label_sha256') and sha256_file(label) != e['input_label_sha256']:
            raise ValueError('Original Indian target labels changed during copying')
    return {split: len(values) for split, values in splits.items()}


def prepare(indian_build, out=DEFAULT_OUT, policy_path=DEFAULT_POLICY, max_train_group=20, ml_dir=ML_DIR):
    ml_dir, indian_build, out = Path(ml_dir).resolve(), Path(indian_build).resolve(), Path(out).resolve()
    out.relative_to(ml_dir)
    if out.is_relative_to(indian_build) or indian_build.is_relative_to(out):
        raise ValueError('Combined output must be separate from the original Indian build')
    policy = json.loads(Path(policy_path).read_text())
    manifests = {'indian': indian_build/'manifest.json',
                 'original': ml_dir/'data/processed/manifest.json',
                 'augmented': ml_dir/'data/processed/augmented_manifest.json'}
    for name, path in manifests.items():
        if sha256_file(path) != policy[name + '_manifest_sha256']:
            raise ValueError(f'{name} manifest changed since the source audit')
    validate_indian_build(indian_build, ml_dir)
    originals = load_manifest(manifests['original'])
    reviews = {r['source_id']: r for r in policy['reviews']}
    for e in originals:
        if (e['source_id'] not in reviews or sha256_file(image_path(e, ml_dir)) != e['source_sha256']
                or e['source_sha256'] != reviews[e['source_id']]['image_sha256']):
            raise ValueError('Historical original changed since visual review')
    splits, selection = select_entries(load_manifest(manifests['indian']), originals,
                                       load_manifest(manifests['augmented']), policy, max_train_group)
    selected = [e for values in splits.values() for e in values]
    preflight(selected, ml_dir)
    configuration = {'policy_sha256': sha256_file(policy_path), 'builder_sha256': sha256_file(__file__),
                     'split_code_sha256': sha256_file(Path(__file__).with_name('dataset_utils.py')),
                     'label_writer_sha256': sha256_file(Path(__file__).with_name('prepare_yolo_dataset.py')),
                     'max_train_views_per_group': max_train_group,
                     'input_manifests': {name: sha256_file(p) for name, p in manifests.items()}}
    build_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:20]
    build = out/'builds'/build_id
    if build.exists():
        validate_build(build, ml_dir)
    else:
        build.mkdir(parents=True)
        prepared = []
        for split, entries in splits.items():
            (build/'images'/split).mkdir(parents=True)
            (build/'labels'/split).mkdir(parents=True)
            rows = []
            for i, e in enumerate(entries):
                source = image_path(e, ml_dir)
                name = f'{i:05d}_{e["image_sha256"][:16]}'
                target = build/'images'/split/(name+source.suffix.lower())
                shutil.copy2(source, target)
                label = build/'labels'/split/(name+'.txt')
                copied_label = {}
                if e['training_domain'] == 'indian_source':
                    source_label = source.parents[2]/'labels'/split/(source.stem+'.txt')
                    shutil.copy2(source_label, label)
                    copied_label = {'input_label_sha256': sha256_file(source_label)}
                else:
                    label.write_text('\n'.join(voc_to_yolo_line(b, e['width'], e['height']) for b in e['boxes']))
                rows.append({**e, **copied_label, 'input_image_path': e['image_path'],
                             'image_path': target.relative_to(ml_dir).as_posix()})
            prepared.extend(rows)
            atomic_json(build/(split+'_manifest.json'), rows)
        atomic_json(build/'manifest.json', prepared)
        atomic_json(build/'provenance.json', provenance(prepared, ml_dir))
        report = {'configuration': configuration, **selection,
                  'splits': {s: len(es) for s, es in splits.items()},
                  'domains': {s: dict(Counter(e['training_domain'] for e in es)) for s, es in splits.items()},
                  'groups': {s: len({e['leakage_group'] for e in es}) for s, es in splits.items()},
                  'legacy_train_vehicle_context': dict(Counter(
                      reviews[e['source_id']].get('vehicle_context', 'unspecified')
                      for e in originals if e['source_id'] in {r['source_id'] for r in splits['train']
                                                             if r['training_domain'] == 'legacy_synthetic'})),
                  'test_source_manifest_sha256': sha256_file(indian_build/'test_manifest.json'),
                  'test_images_unchanged': True, 'indian_labels_copied_without_changes': True,
                  'validation_source_images_unchanged': True,
                  'scope': 'Development detector dataset using supplied target boxes; scene completeness remains unverified.',
                  'historical_validation_scope': 'Legacy stages may have seen these sources; synthetic weather is not real-weather evidence.'}
        atomic_json(build/'report.json', report)
        (build/'data.yaml').write_text("train: images/train\nval: images/val\ntest: images/test\n"
                                      "provenance: provenance.json\nnc: 1\nnames: ['number_plate']\n")
        atomic_json(build/'files.sha256.json', {p.relative_to(build).as_posix(): sha256_file(p)
                   for p in sorted(build.rglob('*')) if p.is_file()})
        validate_build(build, ml_dir)
    # Publish only after every image, label and manifest has passed validation.
    prefix = build.relative_to(out).as_posix()
    yaml = (f'train: {prefix}/images/train\nval: {prefix}/images/val\ntest: {prefix}/images/test\n'
            f"provenance: {prefix}/provenance.json\nnc: 1\nnames: ['number_plate']\n")
    temporary = out/'data.yaml.tmp'
    temporary.write_text(yaml)
    temporary.replace(out/'data.yaml')
    atomic_json(out/'current.json', {'build': prefix, 'configuration': configuration})
    print(json.dumps(json.loads((build/'report.json').read_text()), indent=2))
    print(f'Prepared candidate data: {out / "data.yaml"}')
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indian-build', type=Path)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    parser.add_argument('--max-train-group', type=int, default=20)
    args = parser.parse_args()
    build = args.indian_build
    if build is None:
        parent = ML_DIR/'data/indian_plates'
        build = parent/json.loads((parent/'current.json').read_text())['build']
    prepare(build, args.out, args.policy, args.max_train_group)


if __name__ == '__main__':
    main()
