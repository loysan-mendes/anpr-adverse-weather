from pathlib import Path
import json
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from model_artifacts import sha256_file
from prepare_balanced_plates import preflight, select_entries, validate_build
from prepare_yolo_dataset import voc_to_yolo_line


def indian(split, index=0, group=None):
    return {'image_path': f'data/{split}/{index}.jpg', 'source_id': f'{split}:{index}',
            'leakage_group': group or f'new:{split}:{index}', 'split': split, 'boxes': []}


def legacy(split, index):
    return {'image_path': f'old/{index}.jpg', 'source_id': f'old:{index}',
            'leakage_group': f'old:{index}', 'split': split, 'source_sha256': f'hash{index}', 'boxes': []}


def fixtures():
    new = [indian('train', i, 'sequence') for i in range(30)] + [indian('val'), indian('test')]
    original = [legacy('train', 0), legacy('val', 1), legacy('test', 2)]
    reviews = [{**e, 'pixel_sha256': f'pixel{e["source_id"]}', 'cross_dataset_matches': [],
                'box_review': 'supplied_boxes_visually_locate_plates'} for e in original]
    aug = []
    for e in original:
        for condition, severity in [('clear', 'none')] + [(c, s) for c in ('haze','rain','blur','lowlight') for s in ('mild','moderate','severe')]:
            aug.append({**e, 'image_path': f'{e["source_id"]}/{condition}/{severity}.jpg',
                        'condition': condition, 'severity': severity})
    policy = {'reviews': reviews, 'augmented_image_sha256': {e['image_path']: e['image_path'] for e in aug}}
    return new, original, aug, policy


class BalancedDatasetTests(unittest.TestCase):
    def test_training_caches_do_not_invalidate_a_build_or_hide_changed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root/'prepared'
            entries = []
            for split in ('train', 'val'):
                image = build/'images'/split/'a.png'
                label = build/'labels'/split/'a.txt'
                image.parent.mkdir(parents=True)
                label.parent.mkdir(parents=True)
                Image.new('RGB', (30, 20), 'black' if split == 'train' else 'white').save(image)
                box = {'xmin': 0, 'ymin': 0, 'xmax': 20, 'ymax': 10}
                label.write_text(voc_to_yolo_line(box, 30, 20))
                entries.append({'image_path': image.relative_to(root).as_posix(),
                                'source_id': split, 'split': split, 'image_sha256': sha256_file(image),
                                'width': 30, 'height': 20, 'boxes': [box]})
            (build/'manifest.json').write_text(json.dumps(entries))
            inventory = {p.relative_to(build).as_posix(): sha256_file(p) for p in build.rglob('*') if p.is_file()}
            (build/'files.sha256.json').write_text(json.dumps(inventory))
            for split in ('train', 'val', 'test'):
                (build/'labels'/(split+'.cache')).write_bytes(b'derived cache')
            self.assertEqual(validate_build(build, root), {'train': 1, 'val': 1, 'test': 0})
            unexpected = build/'labels'/'extra.cache'
            unexpected.write_bytes(b'unexpected')
            with self.assertRaises(ValueError):
                validate_build(build, root)
            unexpected.unlink()
            (build/'labels'/'val'/'a.txt').write_text('0 0.5 0.5 0.1 0.1')
            with self.assertRaises(ValueError):
                validate_build(build, root)

    def test_group_cap_only_changes_training_and_is_order_independent(self):
        new, old, aug, policy = fixtures()
        first, _ = select_entries(new, old, aug, policy, 5)
        second, _ = select_entries(list(reversed(new)), old, aug, policy, 5)
        self.assertEqual({e['source_id'] for e in first['train']}, {e['source_id'] for e in second['train']})
        self.assertEqual(len(first['train']), 5+13)
        self.assertEqual(len(first['val']), 1+13)
        self.assertEqual([e['source_id'] for e in first['test']], ['test:0'])

    def test_training_copy_of_historical_test_vehicle_is_excluded(self):
        new, old, aug, policy = fixtures()
        policy['reviews'][0]['leakage_group'] = policy['reviews'][2]['leakage_group']
        splits, report = select_entries(new, old, aug, policy)
        self.assertNotIn('old:0', {e['source_id'] for e in splits['train']})
        self.assertEqual(len(report['excluded_historical_sources']), 2)

    def test_cross_dataset_validation_or_test_match_excludes_old_training_source(self):
        for split in ('val', 'test'):
            new, old, aug, policy = fixtures()
            policy['reviews'][0]['cross_dataset_matches'] = [{'split': split, 'leakage_group': 'protected'}]
            splits, _ = select_entries(new, old, aug, policy)
            self.assertNotIn('old:0', {e['source_id'] for e in splits['train']})

    def test_partial_audit_or_changed_source_is_rejected(self):
        new, old, aug, policy = fixtures()
        policy['reviews'].pop()
        with self.assertRaises(ValueError):
            select_entries(new, old, aug, policy)
        new, old, aug, policy = fixtures()
        policy['reviews'][0]['source_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            select_entries(new, old, aug, policy)

    def test_weather_derivatives_cannot_be_duplicated_or_dropped(self):
        new, old, aug, policy = fixtures()
        for invalid in (aug+[aug[0]], aug[1:]):
            with self.assertRaises(ValueError):
                select_entries(new, old, invalid, policy)

    def test_preflight_catches_changed_pixels_geometry_and_unhandled_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'a.jpg'
            im = Image.new('RGB', (30, 20))
            im.save(path)
            e = {'image_path': 'a.jpg', 'image_sha256': sha256_file(path), 'width': 30, 'height': 20,
                 'boxes': [{'xmin': 0, 'ymin': 0, 'xmax': 20, 'ymax': 10}]}
            preflight([e], root)
            with self.assertRaises(ValueError):
                preflight([{**e, 'width': 20}], root)
            with self.assertRaises(ValueError):
                preflight([{**e, 'boxes': [{**e['boxes'][0], 'xmax': 31}]}], root)
            exif = Image.Exif()
            exif[274] = 6
            im.save(path, exif=exif)
            with self.assertRaises(ValueError):
                preflight([e], root)
            with self.assertRaises(ValueError):
                preflight([{**e, 'image_sha256': sha256_file(path)}], root)


if __name__ == '__main__':
    unittest.main()
