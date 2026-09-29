"""Audit the requested DataCluster sample and train an isolated vehicle candidate.

Never activates weights. Source-date and near-duplicate groups stay in one split.
This small sample is a development experiment, not a generalization benchmark.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['car', 'motorcycle', 'bus', 'truck', 'other']
MAPPING = {'car': 0, 'two_wheelers': 1, 'bus': 2, 'vehicle_truck': 3,
           'concrete_mixer': 3, 'truck_tanker': 3, 'auto': 4, 'tractor': 4,
           'tempo': 4, 'bicycle': 4}


def prepare(source, destination):
    signature = hashlib.sha256()
    for file in sorted((source/'annotations').glob('*.xml')):
        signature.update(file.name.encode())
        signature.update(file.read_bytes())
    if signature.hexdigest() != 'd02c8d2557b45d539c1727928d7488d16d891881dca82163cc3783bdb79d504e':
        raise ValueError('This pairing repair is only verified for Hugging Face revision e3d82e51595f1f42ec1ee22b23aff95f442e4013. Audit this release first.')
    entries = []
    # Audited repair for this specific public release: images 1..66 are offset,
    # image 98 uses XML 99, and image 100 uses XML 1 (XML 100 duplicates XML 1).
    # Images 67 and 99 lack a visually verified matching annotation; exclude them.
    for index in range(1, 101):
        if index in {67, 99}:
            continue
        annotation_index = index+1 if index <= 66 or index == 98 else (1 if index == 100 else index)
        annotation = source / 'annotations' / f'image_{annotation_index:04}.xml'
        root = ET.parse(annotation).getroot()
        path = source / 'images' / f'image_{index:04}.jpg'
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
            if (width, height) != (int(root.findtext('size/width')), int(root.findtext('size/height'))):
                raise ValueError(f'Annotation/image dimensions disagree: {path}')
            gray = list(ImageOps.grayscale(image).resize((9, 8)).getdata())
            dhash = sum((gray[y*9+x] > gray[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        boxes = []
        for obj in root.findall('object'):
            name = obj.findtext('name')
            cls = MAPPING[name]  # Unknown labels are errors, not silently discarded.
            x1, y1, x2, y2 = [float(obj.findtext('bndbox/'+key)) for key in ('xmin','ymin','xmax','ymax')]
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError(f'Invalid box: {annotation}')
            boxes.append([cls, (x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height])
        entries.append(dict(image=str(path.resolve()), annotation=str(annotation.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                            folder=root.findtext('folder') or '', dhash=dhash, boxes=boxes))
    if len(entries) < 20:
        raise ValueError('Too few annotated images for this experiment')
    groups = list(range(len(entries)))
    def find(i):
        while groups[i] != i:
            i = groups[i]
        return i
    for i, a in enumerate(entries):
        for j, b in enumerate(entries[:i]):
            same_date = a['folder'].isdigit() and len(a['folder']) == 8 and a['folder'] == b['folder']
            if same_date or a['sha256'] == b['sha256'] or (a['dhash'] ^ b['dhash']).bit_count() <= 4:
                groups[find(i)] = find(j)
    unique = sorted({find(i) for i in range(len(entries))})
    random.Random(42).shuffle(unique)
    split_by_group = {g: ('test' if n < max(1, round(len(unique)*.15)) else
                         'val' if n < max(2, round(len(unique)*.30)) else 'train') for n, g in enumerate(unique)}
    destination.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(entries):
        split = split_by_group[find(i)]
        entry.update(split=split, group=find(i))
        images, labels = destination/'images'/split, destination/'labels'/split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        path = Path(entry['image'])
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((1280,1280))
            image.save(images/path.name, quality=95)
        (labels/(path.stem+'.txt')).write_text('\n'.join(' '.join(map(str,b)) for b in entry['boxes'])+'\n')
    counts = {s: dict(Counter(NAMES[b[0]] for e in entries if e['split']==s for b in e['boxes'])) for s in ['train','val','test']}
    if any(set(counts[s]) != set(NAMES) for s in counts):
        raise ValueError(f'Every split needs all classes: {counts}')
    (destination/'manifest.json').write_text(json.dumps(entries, indent=2))
    (destination/'data.yaml').write_text(yaml.safe_dump(dict(path=str(destination.resolve()), train='images/train', val='images/val', test='images/test', names=NAMES)))
    audit = dict(images=len(entries), groups=len(unique), split_images=dict(Counter(e['split'] for e in entries)), boxes=counts,
                 source='Dataclusterlabspvtltd/Indian-Vehicle-Dataset', revision='e3d82e51595f1f42ec1ee22b23aff95f442e4013', mapping=MAPPING,
                 limitations='Small sample; source identity incomplete; perceptual grouping cannot guarantee no near-duplicate leakage. No plate boxes or transcriptions.')
    (destination/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    return destination/'data.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'indian-vehicle-dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'ml/data/vehicle_sample')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preserve-head', action='store_true', help='Retain COCO output dimensions and vehicle IDs; repurpose train class as other')
    args = parser.parse_args()
    if (args.output/'data.yaml').exists():
        raise ValueError('Use a new output directory to preserve the previous split and run.')
    data = prepare(args.source, args.output)
    if args.prepare_only:
        return
    from ultralytics import YOLO
    model = YOLO(str(ROOT/'ml/yolo11n.pt'))
    if args.preserve_head:
        config = yaml.safe_load(data.read_text())
        names = dict(model.names)
        names[6] = 'other'
        config['names'] = names
        data.write_text(yaml.safe_dump(config))
        mapping = {0:2, 1:3, 2:5, 3:7, 4:6}
        for label in (args.output/'labels').rglob('*.txt'):
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            label.write_text('\n'.join(' '.join([str(mapping[int(row[0])]), *row[1:]]) for row in rows)+'\n')
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=4, workers=0, device=0,
                patience=12, seed=42, deterministic=True, amp=True, freeze=10,
                project=str(ROOT/'ml/models/vehicle_runs'), name='datacluster_preserved' if args.preserve_head else 'datacluster_nano',
                optimizer='AdamW' if args.preserve_head else 'auto', lr0=.0001 if args.preserve_head else .001,
                mosaic=.3, degrees=5, translate=.1, scale=.25, fliplr=.5,
                plots=False, cache=False)
    best = Path(model.trainer.best)
    (best.with_suffix('.provenance.json')).write_text(json.dumps(dict(
        dataset_manifest=str(args.output/'manifest.json'),
        manifest_sha256=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
        seen_sources=[e['sha256'] for e in json.loads((args.output/'manifest.json').read_text()) if e['split']!='test']), indent=2))
    print(f'CANDIDATE_READY: {best}', flush=True)


if __name__ == '__main__':
    main()
