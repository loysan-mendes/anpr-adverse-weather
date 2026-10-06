"""Full-image pipeline diagnostic against supplied targets, without activation.

Predicted boxes feed OCR. Unannotated predictions are retained for review and
excluded from reading precision because supplied targets may omit scene plates.
This is a development diagnostic, not an independent full-scene ANPR benchmark.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time

from dataset_utils import image_path, leakage_group, load_manifest
from evaluate_pipeline import match_boxes
from image_ops import box_iou
from model_artifacts import atomic_json, resolve_detector, sha256_file
from plate_validator import clean
from acceptance_policy import DEFAULT_ACCEPTANCE_MIN_CONFIDENCE, validate_acceptance_confidence

ML_DIR = Path(__file__).resolve().parents[1]


def score_targets(entry, result, threshold=0.5):
    """Score labeled targets only; don't assert omitted predictions are false."""
    truth, predictions = entry['boxes'], result['plates']
    if any(not b.get('text') or not clean(b['text']) for b in truth):
        raise ValueError('Every diagnostic target needs a supplied transcription')
    matches = dict(match_boxes(truth, predictions, threshold))
    matched_predictions = set(matches.values())
    counts = Counter(images=1, targets=len(truth), matched_targets=len(matches),
                     missed_targets=len(truth)-len(matches), correct_accepted_targets=0,
                     wrong_accepted_targets=0, accepted_targets=0,
                     exact_proposals=0, correct_uncertain_targets=0,
                     unreadable_targets=0, wrong_uncertain_targets=0,
                     raw_literal_exact_targets=0,
                     unmatched_predictions=len(predictions)-len(matches),
                     unmatched_accepted_predictions=0)
    details = []
    for g, box in enumerate(truth):
        expected = clean(box['text'])
        detail = dict(target_index=g, expected=expected)
        if g not in matches:
            detail['outcome'] = 'missed'
        else:
            p = matches[g]
            prediction = predictions[p]
            accepted = prediction['status'] == 'accepted'
            exact = clean(prediction.get('plate_text', '')) == expected
            proposed_exact = clean(prediction.get('proposed_text', '')) == expected
            counts['accepted_targets'] += accepted
            counts['exact_proposals'] += proposed_exact
            raw_exact = any(clean(''.join(r['text'] for r in reading.get('raw_ocr_regions', []))) == expected
                            for reading in prediction.get('readings', []))
            counts['raw_literal_exact_targets'] += raw_exact
            if accepted:
                outcome = 'correct_accepted' if exact else 'wrong_accepted'
            elif prediction['status'] == 'unreadable':
                outcome = 'unreadable'
            else:
                outcome = 'correct_uncertain' if proposed_exact else 'wrong_uncertain'
            counts[outcome+'_targets'] += 1
            detail.update(prediction_index=p, outcome=outcome,
                          selected=prediction.get('plate_text', ''),
                          proposed=prediction.get('proposed_text', ''), status=prediction['status'],
                          raw_literal_exact=raw_exact,
                          iou=box_iou([box[k] for k in ('xmin','ymin','xmax','ymax')], prediction['box']))
        details.append(detail)
    for p, prediction in enumerate(predictions):
        if p not in matched_predictions:
            counts['unmatched_accepted_predictions'] += prediction['status'] == 'accepted'
    return dict(counts), details


def summarize_targets(rows):
    totals = Counter()
    for row in rows:
        totals.update(row)
    def ratio(n, d):
        return totals[n]/totals[d] if totals[d] else None
    return dict(totals, target_detection_recall=ratio('matched_targets','targets'),
                exact_accepted_target_recall=ratio('correct_accepted_targets','targets'),
                accepted_target_reading_precision=ratio('correct_accepted_targets','accepted_targets'),
                exact_proposal_target_recall=ratio('exact_proposals','targets'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=('balanced','exhaustive'), default='balanced')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--acceptance-min-confidence', type=float, default=DEFAULT_ACCEPTANCE_MIN_CONFIDENCE)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz < 32:
        parser.error('Require confidence in (0,1] and imgsz >= 32')
    try:
        validate_acceptance_confidence(args.acceptance_min_confidence)
    except ValueError as exc:
        parser.error(str(exc))
    from decision_engine import process_image
    from vehicle_detector import resolve_vehicle_weights
    entries = load_manifest(args.manifest)
    for e in entries:
        if not e['boxes'] or any(not b.get('text') for b in e['boxes']):
            raise ValueError('Use an explicit manifest of fully transcribed positive targets')
        for b in e['boxes']:
            if not (0 <= b['xmin'] < b['xmax'] <= e['width'] and 0 <= b['ymin'] < b['ymax'] <= e['height']):
                raise ValueError('Invalid target box')
    weights = resolve_detector(args.weights)
    checkpoints = [weights, ML_DIR/'models/quality_analyzer/best_model.pt', resolve_vehicle_weights()]
    checkpoints += [ML_DIR/f'models/restoration/{c}/best_model.pt' for c in ('blur','haze','rain')]
    checkpoint_hashes = {str(p): sha256_file(p) for p in checkpoints}
    code_hashes = {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    image_hashes = {e['image_path']: sha256_file(image_path(e)) for e in entries}
    if len(image_hashes) != len(entries):
        raise ValueError('Duplicate manifest image paths')
    for e in entries:
        if e.get('image_sha256') and image_hashes[e['image_path']] != e['image_sha256']:
            raise ValueError('Changed manifest image')
    registry = ML_DIR/'models/active_detector.json'
    registry_hash = sha256_file(registry)
    options = dict(conf_threshold=args.conf, weights=str(weights), use_upscale=False,
                   profile=args.profile, imgsz=args.imgsz, vehicle_weights=str(resolve_vehicle_weights()),
                   acceptance_min_confidence=args.acceptance_min_confidence)
    fingerprint = dict(manifest_sha256=sha256_file(args.manifest), checkpoints=checkpoint_hashes,
                       code_sha256=code_hashes, image_sha256=image_hashes, options=options)
    report = dict(evaluation='development_diagnostic_supplied_targets', completed=False,
                  created_utc=datetime.now(timezone.utc).isoformat(), **fingerprint,
                  package_versions={p:importlib.metadata.version(p) for p in ('torch','ultralytics','paddleocr','onnxruntime')},
                  source_images=len(entries), related_groups=len({leakage_group(e) for e in entries}),
                  partitions=sorted({e.get('split','unspecified') for e in entries}),
                  scope='Matched supplied targets only; unmatched scene predictions require review.',
                  images=[])
    if args.resume and args.out.is_file():
        prior = json.loads(args.out.read_text(encoding='utf-8'))
        if any(prior.get(k) != v for k,v in fingerprint.items()):
            raise ValueError('Resume inputs, code, settings or checkpoints changed')
        report = prior
    completed = {row['image_path'] for row in report['images']}
    print(f'PIPELINE_DIAGNOSTIC_READY: {len(entries)} images; checkpoint={weights}', flush=True)
    for i,e in enumerate(entries):
        if e['image_path'] in completed:
            continue
        started = time.perf_counter()
        result = process_image(image_path(e), **options)
        seconds = time.perf_counter()-started
        counts, details = score_targets(e, result)
        report['images'].append(dict(image_path=e['image_path'],
                                     source_relative=e.get('source_relative',e['image_path']),
                                     leakage_group=leakage_group(e), source=e.get('source','unspecified'),
                                     seconds=seconds, counts=counts, target_details=details, result=result))
        atomic_json(args.out,report)
        print(f'{i+1}/{len(entries)} {seconds:.2f}s accepted_exact={counts["correct_accepted_targets"]}/{counts["targets"]} '
              f'wrong_accepted={counts["wrong_accepted_targets"]} unmatched={counts["unmatched_predictions"]}', flush=True)
    groups, sources = defaultdict(list), defaultdict(list)
    for row in report['images']:
        groups[row['leakage_group']].append(row['counts'])
        sources[row['source']].append(row['counts'])
    report['metrics'] = summarize_targets([row['counts'] for row in report['images']])
    report['by_source'] = {source:summarize_targets(rows) for source,rows in sources.items()}
    report['group_macro_exact_accepted_recall'] = statistics.mean(
        summarize_targets(rows)['exact_accepted_target_recall'] for rows in groups.values())
    times = [row['seconds'] for row in report['images']]
    report['latency_seconds'] = dict(first_image=times[0], mean=statistics.mean(times),
                                   warm_mean=statistics.mean(times[1:]) if len(times)>1 else None,
                                   p95=sorted(times)[max(0,math.ceil(.95*len(times))-1)])
    report['stage_calls'] = dict(sum((Counter(row['result']['stage_calls']) for row in report['images']), Counter()))
    report['restoration_applied'] = dict(Counter(row['result']['restoration_applied'] or 'none' for row in report['images']))
    for path,digest in checkpoint_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError('Checkpoint changed during diagnostic')
    if sha256_file(registry) != registry_hash:
        raise ValueError('Active detector changed during diagnostic')
    report.update(completed=True, active_registry_unchanged=True)
    atomic_json(args.out,report)
    print(json.dumps({k:report[k] for k in ('metrics','by_source','group_macro_exact_accepted_recall',
                                         'latency_seconds','stage_calls','restoration_applied')},indent=2),flush=True)


if __name__ == '__main__':
    main()
