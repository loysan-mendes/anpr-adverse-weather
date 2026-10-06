"""Compare supplied-target crops with saved predicted-crop OCR evidence.

Development diagnostic only: source text is not independently certified. The
supplied boxes are a localization control, not an end-to-end accuracy claim.
No detector training, activation, restoration or test inference is performed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path

from dataset_utils import image_path, load_manifest
from image_ops import padded_box
from model_artifacts import atomic_json, sha256_file
from plate_validator import choose_reading, clean, order_regions

ML_DIR = Path(__file__).resolve().parents[1]
PROCESSING_FILES = ('decision_engine.py', 'image_ops.py', 'ocr_plate.py', 'plate_validator.py')


def crop_box(box, width, height):
    values = [float(v) for v in box]
    if not all(math.isfinite(v) for v in values) or not (
            0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height):
        raise ValueError('Invalid crop geometry')
    # Round supplied fractional coordinates outward before the same context
    # padding used by the app. Saved detector boxes are already integers.
    native = [math.floor(values[0]), math.floor(values[1]), math.ceil(values[2]), math.ceil(values[3])]
    return padded_box(native, width, height)


def literal_evidence(readings, expected):
    """Exact contiguous OCR regions, without substituting characters."""
    expected = clean(expected)
    for reading in readings:
        regions = order_regions(reading.get('raw_ocr_regions', []))
        parts = [clean(r['text']) for r in regions if clean(r['text']) not in ('', 'IND')]
        for start in range(len(parts)):
            for end in range(start+1, len(parts)+1):
                if ''.join(parts[start:end]) == expected:
                    return True
    return False


def score_readings(readings, expected):
    expected = clean(expected)
    selected = choose_reading(readings)
    accepted = selected['status'] == 'accepted'
    exact = clean(selected.get('plate_text', '')) == expected
    proposed_exact = clean(selected.get('proposed_text', '')) == expected
    outcome = ('correct_accepted' if exact else 'wrong_accepted') if accepted else (
        'unreadable' if selected['status'] == 'unreadable' else
        'correct_uncertain' if proposed_exact else 'wrong_uncertain')
    return {'outcome': outcome, 'exact_proposal': proposed_exact,
            'literal_exact_available': literal_evidence(readings, expected),
            'selected': selected, 'readings': readings}


def summarize(rows, key):
    results = [r[key] for r in rows if r.get(key) is not None]
    counts = Counter(r['outcome'] for r in results)
    for outcome in ('correct_accepted', 'wrong_accepted', 'correct_uncertain', 'wrong_uncertain', 'unreadable'):
        counts.setdefault(outcome, 0)
    counts['crops'] = len(results)
    counts['literal_exact_available'] = sum(r['literal_exact_available'] for r in results)
    counts['exact_proposals'] = sum(r['exact_proposal'] for r in results)
    accepted = counts['correct_accepted'] + counts['wrong_accepted']
    return {**dict(counts), 'exact_accepted_fraction': counts['correct_accepted']/len(results) if results else None,
            'accepted_target_precision': counts['correct_accepted']/accepted if accepted else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pipeline-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    entries = load_manifest(args.manifest)
    if any(e.get('split') == 'test' for e in entries):
        raise ValueError('Use development data, not the reserved test partition')
    pipeline = json.loads(args.pipeline_report.read_text())
    if not pipeline.get('completed') or sha256_file(args.manifest) != pipeline['manifest_sha256']:
        raise ValueError('Completed pipeline evidence must match this manifest')
    for name in PROCESSING_FILES:
        if sha256_file(Path(__file__).with_name(name)) != pipeline['code_sha256'][name]:
            raise ValueError(f'Processing code changed since the predicted-crop run: {name}')
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'ultralytics', 'paddleocr', 'onnxruntime')}
    if packages != pipeline['package_versions']:
        raise ValueError('OCR runtime packages changed')
    saved = {r['image_path']: r for r in pipeline['images']}
    if len(saved) != len(entries) or set(saved) != {e['image_path'] for e in entries}:
        raise ValueError('Pipeline image inventory differs')
    for e in entries:
        if not e['boxes'] or any(not b.get('text') for b in e['boxes']):
            raise ValueError('Every diagnostic target needs supplied text')
        if sha256_file(image_path(e)) != pipeline['image_sha256'][e['image_path']]:
            raise ValueError('Input image changed')
    for path, digest in pipeline['checkpoints'].items():
        if sha256_file(path) != digest:
            raise ValueError('Stage checkpoint changed')
    registry = ML_DIR/'models/active_detector.json'
    registry_hash = sha256_file(registry)
    fingerprint = {'manifest_sha256': sha256_file(args.manifest),
                   'pipeline_report_sha256': sha256_file(args.pipeline_report),
                   'code_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
                   'packages': packages, 'active_registry_sha256': registry_hash,
                   'options': {'profile': 'balanced', 'use_upscale': False, 'padding_fraction': .04,
                               'restoration': False, 'predicted_crop_readings': 'saved_original_image_only'}}
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'completed': False,
              'scope': 'Development OCR localization control against supplied text; not end-to-end performance',
              **fingerprint, 'rows': []}
    if args.resume and args.out.exists():
        previous = json.loads(args.out.read_text())
        if any(previous.get(k) != v for k, v in fingerprint.items()):
            raise ValueError('Resume inputs, processing code, packages or registry changed')
        report = previous
    done = {(r['image_path'], r['target_index']) for r in report['rows']}
    import cv2
    from decision_engine import read_crop
    for i, e in enumerate(entries):
        if all((e['image_path'], n) in done for n in range(len(e['boxes']))):
            continue
        image = cv2.imread(str(image_path(e)))
        if image is None or image.shape[:2] != (e['height'], e['width']):
            raise ValueError('Decoded/oriented image dimensions differ from the manifest')
        prior = saved[e['image_path']]
        details = {d['target_index']: d for d in prior['target_details']}
        for n, b in enumerate(e['boxes']):
            if (e['image_path'], n) in done:
                continue
            expected = clean(b['text'])
            if expected != details[n]['expected']:
                raise ValueError('Target transcription changed')
            geometry = crop_box([b[k] for k in ('xmin','ymin','xmax','ymax')], e['width'], e['height'])
            x1,y1,x2,y2 = geometry
            readings = read_crop(image[y1:y2,x1:x2], 'diagnostic_ground_truth', use_upscale=False, profile='balanced')
            predicted, predicted_geometry = None, None
            if 'prediction_index' in details[n]:
                p = prior['result']['plates'][details[n]['prediction_index']]
                predicted_readings = [r for r in p['readings'] if r['ocr_source'] == 'original']
                if not predicted_readings:
                    raise ValueError('Missing original-image predicted-crop evidence')
                predicted = score_readings(predicted_readings, expected)
                predicted_geometry = crop_box(p['box'], e['width'], e['height'])
            report['rows'].append({'image_path': e['image_path'], 'source_relative': e.get('source_relative',e['image_path']),
                                   'source': e.get('source','unspecified'), 'leakage_group': prior['leakage_group'],
                                   'target_index': n, 'expected': expected, 'ground_truth_crop_box': geometry,
                                   'predicted_crop_box': predicted_geometry,
                                   'ground_truth': score_readings(readings, expected), 'predicted': predicted,
                                   'full_pipeline_outcome': details[n]['outcome']})
        atomic_json(args.out, report)
        if (i+1) % 10 == 0 or i+1 == len(entries):
            print(f'OCR_CROP_DIAGNOSTIC: {i+1}/{len(entries)} images; {len(report["rows"])} targets', flush=True)
    paired = [r for r in report['rows'] if r['predicted'] is not None]
    by_source = defaultdict(list)
    for row in paired:
        by_source[row['source']].append(row)
    report['metrics'] = {'supplied_crops_all_targets': summarize(report['rows'],'ground_truth'),
                         'supplied_crops_matched_targets': summarize(paired,'ground_truth'),
                         'predicted_crops_matched_targets': summarize(paired,'predicted'),
                         'unmatched_targets': len(report['rows'])-len(paired)}
    report['paired_transitions'] = dict(Counter(r['predicted']['outcome']+' -> '+r['ground_truth']['outcome'] for r in paired))
    report['by_source'] = {s: {'supplied_crops':summarize(rs,'ground_truth'), 'predicted_crops':summarize(rs,'predicted')}
                           for s,rs in by_source.items()}
    if sha256_file(registry) != registry_hash:
        raise ValueError('Active detector registry changed')
    for path, digest in pipeline['checkpoints'].items():
        if sha256_file(path) != digest:
            raise ValueError('Stage checkpoint changed during the diagnostic')
    for e in entries:
        if sha256_file(image_path(e)) != pipeline['image_sha256'][e['image_path']]:
            raise ValueError('Input image changed during the diagnostic')
    report.update(completed=True, active_detector_unchanged=True, reserved_test_inference=False)
    atomic_json(args.out, report)
    print(json.dumps(report['metrics'],indent=2),flush=True)


if __name__ == '__main__':
    main()
