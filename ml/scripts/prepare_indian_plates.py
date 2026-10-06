"""Prepare a reviewed, isolated plate dataset; never train or activate models.

Reads plate text from VOC object names. Original downloads remain untouched.
Exact copies, matching plate identities, sequence prefixes and perceptually
similar images share a leakage group. Full-image annotation completeness is
explicitly unverified: this corpus is not an independent end-to-end benchmark.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import uuid
import xml.etree.ElementTree as ET

from dataset_utils import split_entries, provenance
from model_artifacts import atomic_json, sha256_file
from plate_validator import clean, template_fits

ML_DIR = Path(__file__).resolve().parents[1]
ROOT = ML_DIR.parent
DEFAULT_POLICY = ML_DIR / 'data/curation/new_indian_plates.json'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


class Groups:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def dataset_fingerprint(files, source):
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resolve_image(annotation, declared_filename, images):
    # XML paths originate on another computer; only a local basename is used.
    basename = declared_filename.replace('\\', '/').rsplit('/', 1)[-1]
    direct = annotation.parent / basename
    if direct in images:
        return direct
    candidates = [p for p in images if p.parent == annotation.parent
                  and p.stem.lower() == annotation.stem.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise ValueError(f'Ambiguous annotation/image match: {annotation}')


def classify_text(raw_text, policy):
    text = clean(raw_text)
    if text in policy['exclude_labels']:
        return None, 'excluded', policy['exclude_labels'][text]
    if text in policy['detection_only_labels']:
        return None, 'detection_only', policy['detection_only_labels'][text]
    if not re.fullmatch(r'[A-Z]{2}[A-Z0-9]{3,12}', text) or not re.search(r'\d', text):
        raise ValueError(f'Unreviewed special text label: {raw_text!r}')
    return text, 'transcribed', None


def inspect_source(source, policy):
    from PIL import Image, ImageOps
    files = sorted(p for p in source.rglob('*') if p.is_file())
    fingerprint = dataset_fingerprint(files, source)
    if fingerprint != policy['dataset_sha256']:
        raise ValueError('Source dataset changed; review and fingerprint the new release before applying curation.')
    images = {p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS}
    annotations = [p for p in files if p.suffix.lower() == '.xml']
    metadata = {}
    for path in sorted(images):
        with path.open('rb') as stream:
            stream.seek(-2,2)
            jpeg_marker_repair = path.suffix.lower() in {'.jpg','.jpeg'} and stream.read()!=b'\xff\xd9'
        with Image.open(path) as original:
            orientation = original.getexif().get(274, 1)
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.load()
            pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            gray = list(image.convert('L').resize((9, 8)).getdata())
            dhash = sum(int(gray[y*9+x] > gray[y*9+x+1]) << (y*8+x)
                        for y in range(8) for x in range(8))
            metadata[path] = {'width': image.width, 'height': image.height,
                              'orientation': orientation, 'pixel_sha256': pixels,
                              'source_sha256': sha256_file(path), 'dhash': dhash,
                              'jpeg_marker_repair':jpeg_marker_repair}
    records, exclusions, repairs, matched = [], [], [], set()
    for annotation in annotations:
        relative = annotation.relative_to(source).as_posix()
        xml = ET.parse(annotation).getroot()
        declared_filename = xml.findtext('filename') or ''
        path = resolve_image(annotation, declared_filename, images)
        if path is None:
            if relative not in policy['missing_images']:
                raise FileNotFoundError(f'Unreviewed missing image: {annotation}')
            exclusions.append({'annotation': relative, 'reason': 'missing_source_image'})
            continue
        matched.add(path)
        data = metadata[path]
        if data['jpeg_marker_repair']:
            repairs.append({'image':path.relative_to(source).as_posix(), 'kind':'jpeg_container',
                            'corrected':'lossless PNG with identical decoded pixels'})
        declared_size = [int(float(xml.findtext('size/'+key))) for key in ('width', 'height')]
        actual_size = [data['width'], data['height']]
        basename = declared_filename.replace('\\', '/').rsplit('/', 1)[-1]
        if basename.lower() != path.name.lower():
            if policy['filename_repairs'].get(relative) != path.name:
                raise ValueError(f'Unreviewed filename mismatch: {annotation}')
            repairs.append({'annotation': relative, 'kind': 'filename',
                            'original': declared_filename, 'corrected': path.name})
        if declared_size != actual_size:
            decision = policy['dimension_repairs'].get(relative)
            if decision != {'declared': declared_size, 'actual': actual_size}:
                raise ValueError(f'Unreviewed dimension mismatch: {annotation}')
            repairs.append({'annotation': relative, 'kind': 'dimensions',
                            'original': declared_size, 'corrected': actual_size,
                            'boxes': 'visually reviewed; coordinates preserved'})
        boxes, excluded_reason = [], None
        for obj in xml.findall('object'):
            bnd = obj.find('bndbox')
            if bnd is None:
                raise ValueError(f'Missing box: {annotation}')
            coords = [float(bnd.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            x1,y1,x2,y2 = coords
            if not all(math.isfinite(v) for v in coords) or not (
                0 <= x1 < x2 <= data['width'] and 0 <= y1 < y2 <= data['height']):
                raise ValueError(f'Out-of-bounds or degenerate box: {annotation}')
            raw_text = obj.findtext('name') or ''
            text, status, reason = classify_text(raw_text, policy)
            if status == 'excluded':
                excluded_reason = reason
            boxes.append(dict(zip(('xmin','ymin','xmax','ymax'),coords),
                              text=text, raw_text=raw_text, text_status=status,
                              text_review_reason=reason,
                              leakage_identity=policy.get('identity_aliases', {}).get(clean(raw_text), clean(raw_text)),
                              transcription_verification='source_provided' if text else 'not_eligible'))
        if not boxes:
            raise ValueError(f'Unreviewed empty annotation: {annotation}')
        if excluded_reason:
            exclusions.append({'annotation':relative, 'image':path.relative_to(source).as_posix(),
                               'reason':excluded_reason, 'raw_labels':[b['raw_text'] for b in boxes]})
            continue
        records.append({**data, 'path':path, 'annotation_path':annotation,
                        'source_relative':path.relative_to(source).as_posix(),
                        'source':'indian_plates:'+path.relative_to(source).parts[0],
                        'boxes':boxes})
    for path in sorted(images-matched):
        relative = path.relative_to(source).as_posix()
        if relative not in policy['unannotated_variants']:
            raise ValueError(f'Unreviewed image without annotation: {relative}')
        exclusions.append({'image':relative, 'reason':'alternate_encoding_without_separate_annotation'})
    return records, fingerprint, exclusions, repairs, len(images), len(annotations)


def group_and_deduplicate(records, distance=4):
    groups = Groups(len(records))
    pixels, identities, sequences = {}, {}, {}
    exact_sets = defaultdict(list)
    near_edges = []
    for index, record in enumerate(records):
        digest = record['pixel_sha256']
        exact_sets[digest].append(index)
        if digest in pixels:
            groups.join(index,pixels[digest])
        pixels[digest] = index
        for box in record['boxes']:
            identity = box.get('leakage_identity') or clean(box['raw_text'])
            if re.fullmatch(r'[A-Z]{2}[A-Z0-9]{3,12}', identity) and re.search(r'\d', identity):
                if identity in identities:
                    groups.join(index,identities[identity])
                identities[identity] = index
        if record['source_relative'].startswith('video_images/'):
            sequence = re.sub(r'[_-]\d+$','',Path(record['source_relative']).stem)
            if sequence in sequences:
                groups.join(index,sequences[sequence])
            sequences[sequence] = index
            record['sequence_id'] = sequence
        else:
            record['sequence_id'] = None
        for previous, other in enumerate(records[:index]):
            if digest != other['pixel_sha256'] and (record['dhash'] ^ other['dhash']).bit_count() <= distance:
                groups.join(index,previous)
                near_edges.append([record['source_relative'],other['source_relative']])
    members = defaultdict(list)
    for index,record in enumerate(records):
        members[groups.find(index)].append(record['pixel_sha256'])
    group_ids = {key:'indian-plates-group:'+hashlib.sha256('\n'.join(sorted(set(values))).encode()).hexdigest()
                 for key,values in members.items()}
    clean_records, duplicate_map = [], []
    for digest,indices in sorted(exact_sets.items()):
        versions = [records[i] for i in indices]
        text_sets = {tuple(b['text'] for b in v['boxes']) for v in versions}
        if len(text_sets)>1:
            raise ValueError(f'Conflicting annotations for identical pixels: {[v["source_relative"] for v in versions]}')
        selected = min(versions,key=lambda r:r['source_relative'])
        index = records.index(selected)
        selected = {**selected, 'leakage_group':group_ids[groups.find(index)],
                    'source_id':'indian-plates:'+selected['source_sha256'],
                    'duplicate_source_sha256':sorted({v['source_sha256'] for v in versions}),
                    'source_aliases':sorted(v['source_relative'] for v in versions)}
        clean_records.append(selected)
        if len(versions)>1:
            duplicate_map.append({'kept':selected['source_relative'],
                                  'removed_from_build':[v['source_relative'] for v in versions if v['source_relative']!=selected['source_relative']],
                                  'reason':'identical_decoded_pixels'})
    return clean_records, duplicate_map, near_edges


def assign_group_splits(records, val_frac=.15, test_frac=.15, seed=42):
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac+test_frac >= 1:
        raise ValueError('Require nonempty train, validation and test fractions')
    buckets = defaultdict(list)
    for record in records:
        buckets[record['leakage_group']].append(record)
    if len(buckets)<3:
        raise ValueError('Need at least three independent groups')
    keys = sorted(buckets)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key:-len(buckets[key]))
    targets = {'train':len(records)*(1-val_frac-test_frac),
               'val':len(records)*val_frac, 'test':len(records)*test_frac}
    counts = Counter()
    assignment = {}
    for key in keys:
        # Largest groups first, filling the greatest remaining target fraction.
        split = max(targets,key=lambda s:(targets[s]-counts[s])/targets[s])
        assignment[key] = split
        counts[split] += len(buckets[key])
    if any(not counts[s] for s in targets):
        raise ValueError(f'Cannot form nonempty grouped partitions: {dict(counts)}')
    return [{**r,'split':assignment[r['leakage_group']]} for r in records]


def write_voc(path,entry):
    root = ET.Element('annotation')
    ET.SubElement(root,'filename').text=Path(entry['image_path']).name
    size=ET.SubElement(root,'size')
    for key,value in (('width',entry['width']),('height',entry['height']),('depth',3)):
        ET.SubElement(size,key).text=str(value)
    for box in entry['boxes']:
        obj=ET.SubElement(root,'object')
        ET.SubElement(obj,'name').text='number_plate'
        bnd=ET.SubElement(obj,'bndbox')
        for key in ('xmin','ymin','xmax','ymax'):
            ET.SubElement(bnd,key).text=str(box[key])
        attrs=ET.SubElement(obj,'attributes')
        for name,value in (('number_plate_text',box['text']),('text_status',box['text_status'])):
            if value is not None:
                attr=ET.SubElement(attrs,'attribute')
                ET.SubElement(attr,'name').text=name
                ET.SubElement(attr,'value').text=value
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def validate_build(build, ml_dir=ML_DIR):
    entries=json.loads((build/'manifest.json').read_text())
    splits=split_entries(entries)
    pixel_splits={}
    sequences={}
    for split,values in splits.items():
        for entry in values:
            path=ml_dir/entry['image_path']
            if sha256_file(path)!=entry['image_sha256']:
                raise ValueError(f'Prepared image changed: {path}')
            for mapping,key in ((pixel_splits,entry['pixel_sha256']),(sequences,entry.get('sequence_id'))):
                if key and mapping.setdefault(key,split)!=split:
                    raise ValueError('Related prepared images cross splits')
            label=build/'labels'/split/(path.stem+'.txt')
            rows=[line.split() for line in label.read_text().splitlines() if line.strip()]
            if len(rows)!=len(entry['boxes']):
                raise ValueError(f'Prepared labels changed: {label}')
            for row,box in zip(rows,entry['boxes']):
                x1,y1,x2,y2=(box[k] for k in ('xmin','ymin','xmax','ymax'))
                expected=[(x1+x2)/2/entry['width'],(y1+y2)/2/entry['height'],
                          (x2-x1)/entry['width'],(y2-y1)/entry['height']]
                if len(row)!=5 or row[0]!='0' or any(abs(float(a)-b)>1e-7 for a,b in zip(row[1:],expected)):
                    raise ValueError(f'Prepared box changed: {label}')
    inventory=json.loads((build/'files.sha256.json').read_text())
    for relative,digest in inventory.items():
        if sha256_file(build/relative)!=digest:
            raise ValueError(f'Prepared artifact changed: {relative}')
    return {split:len(values) for split,values in splits.items()}


def prepare(source, out_dir, policy_path=DEFAULT_POLICY, val_frac=.15, test_frac=.15, seed=42, ml_dir=ML_DIR):
    from PIL import Image, ImageOps
    source,out_dir,ml_dir=Path(source).resolve(),Path(out_dir).resolve(),Path(ml_dir).resolve()
    out_dir.relative_to(ml_dir)  # Prepared manifests remain portable under ml/.
    if out_dir.is_relative_to(source) or source.is_relative_to(out_dir):
        raise ValueError('Source and prepared output must be separate directories')
    policy=json.loads(Path(policy_path).read_text())
    records,fingerprint,exclusions,repairs,n_images,n_annotations=inspect_source(source,policy)
    records,duplicates,near_edges=group_and_deduplicate(records)
    records=assign_group_splits(records,val_frac,test_frac,seed)
    config={'source_sha256':fingerprint,'policy_sha256':sha256_file(policy_path),
            'importer_sha256':sha256_file(__file__),
            'format_policy_sha256':sha256_file(Path(__file__).with_name('plate_validator.py')),
            'split_code_sha256':sha256_file(Path(__file__).with_name('dataset_utils.py')),
            'val_frac':val_frac,'test_frac':test_frac,'seed':seed}
    build_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:20]
    final=out_dir/'builds'/build_id
    if final.exists():
        validate_build(final,ml_dir)
        atomic_json(out_dir/'current.json',{'build':final.relative_to(out_dir).as_posix(),'configuration':config})
        publish_yaml(out_dir,final)
        print(f'Existing verified build: {final}')
        return final
    staging=out_dir/'builds'/('.preparing-'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    entries=[]
    for index,record in enumerate(sorted(records,key=lambda r:r['source_relative'])):
        split=record['split']
        for folder in ('images','labels','annotations'):
            (staging/folder/split).mkdir(parents=True,exist_ok=True)
        normalize = record['orientation']!=1 or record['jpeg_marker_repair']
        suffix='.png' if normalize else record['path'].suffix.lower()
        name=f'{index:05d}_{record["source_sha256"][:16]}'
        destination=staging/'images'/split/(name+suffix)
        if normalize:
            with Image.open(record['path']) as image:
                ImageOps.exif_transpose(image).convert('RGB').save(destination)
        else:
            shutil.copy2(record['path'],destination)
        entry={k:v for k,v in record.items() if k not in {'path','annotation_path','dhash','orientation','jpeg_marker_repair'}}
        entry.update(image_path=(final/'images'/split/destination.name).relative_to(ml_dir).as_posix(),
                     image_sha256=sha256_file(destination),condition='unspecified',severity='unspecified',
                     annotation_scope='provided_target_only',full_image_labels_verified=False,
                     weather_labels_verified=False,
                     original_annotation_sha256=sha256_file(record['annotation_path']))
        lines=[]
        for box in entry['boxes']:
            x1,y1,x2,y2=(box[k] for k in ('xmin','ymin','xmax','ymax'))
            values=((x1+x2)/2/entry['width'],(y1+y2)/2/entry['height'],
                    (x2-x1)/entry['width'],(y2-y1)/entry['height'])
            lines.append('0 '+' '.join(f'{value:.9f}' for value in values))
        (staging/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n')
        write_voc(staging/'annotations'/split/(name+'.xml'),entry)
        entries.append(entry)
    splits=split_entries(entries)
    atomic_json(staging/'manifest.json',entries)
    ocr=[e for e in entries if all(b['text'] is not None for b in e['boxes'])]
    atomic_json(staging/'ocr_manifest.json',ocr)
    for split,values in splits.items():
        atomic_json(staging/(split+'_manifest.json'),values)
        atomic_json(staging/('ocr_'+split+'_manifest.json'),[e for e in ocr if e['split']==split])
    # Hash final images via their staging counterparts before publishing.
    for entry in entries:
        entry['_prepared_path']=str(staging/'images'/entry['split']/Path(entry['image_path']).name)
    seen=provenance([{**e,'image_path':e['_prepared_path']} for e in entries],ml_dir)
    for entry in entries:
        del entry['_prepared_path']
    atomic_json(staging/'provenance.json',seen)
    atomic_json(staging/'exclusions.json',exclusions)
    atomic_json(staging/'repairs.json',repairs)
    atomic_json(staging/'duplicates.json',duplicates)
    atomic_json(staging/'near_duplicate_groups.json',near_edges)
    report={'configuration':config,'source_url':policy['source_url'],'source_images':n_images,
            'source_annotations':n_annotations,'retained_images':len(entries),'ocr_eligible_images':len(ocr),
            'detection_only_images':len(entries)-len(ocr),'splits':{k:len(v) for k,v in splits.items()},
            'split_groups':{k:len({e['leakage_group'] for e in v}) for k,v in splits.items()},
            'ocr_splits':dict(Counter(e['split'] for e in ocr)),
            'duplicate_images_removed':sum(len(d['removed_from_build']) for d in duplicates),
            'excluded':dict(Counter(e['reason'] for e in exclusions)),
            'repairs':repairs,'near_duplicate_edges_grouped':len(near_edges),
            'annotation_scope':'provided_target_only',
            'limitations':['Source-provided transcriptions are not all visually verified.',
                           'No certification of complete full-image labels or real weather conditions.',
                           'Perceptual grouping is a heuristic; acquisition identities are incomplete.']}
    atomic_json(staging/'report.json',report)
    yaml="train: images/train\nval: images/val\ntest: images/test\nprovenance: provenance.json\nnc: 1\nnames: ['number_plate']\n"
    (staging/'data.yaml').write_text(yaml)
    inventory={p.relative_to(staging).as_posix():sha256_file(p) for p in sorted(staging.rglob('*')) if p.is_file()}
    atomic_json(staging/'files.sha256.json',inventory)
    staging.rename(final)
    validate_build(final,ml_dir)
    atomic_json(out_dir/'current.json',{'build':final.relative_to(out_dir).as_posix(),'configuration':config})
    publish_yaml(out_dir,final)
    print(json.dumps(report,indent=2))
    print(f'Training configuration ready: {out_dir / "data.yaml"}')
    return final


def publish_yaml(out_dir,build):
    relative=build.relative_to(out_dir).as_posix()
    yaml=(f'train: {relative}/images/train\nval: {relative}/images/val\ntest: {relative}/images/test\n'
          f'provenance: {relative}/provenance.json\nnc: 1\nnames: [\'number_plate\']\n')
    temporary=out_dir/'data.yaml.tmp'
    temporary.write_text(yaml)
    temporary.replace(out_dir/'data.yaml')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'new-indian-vehicle-dataset')
    parser.add_argument('--out-dir',type=Path,default=ML_DIR/'data/indian_plates')
    parser.add_argument('--policy',type=Path,default=DEFAULT_POLICY)
    parser.add_argument('--val-frac',type=float,default=.15)
    parser.add_argument('--test-frac',type=float,default=.15)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    prepare(args.source,args.out_dir,args.policy,args.val_frac,args.test_frac,args.seed)


if __name__=='__main__':
    main()
