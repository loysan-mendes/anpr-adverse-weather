"""Real image/annotation checks for the isolated Indian plate import."""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_utils import split_entries, provenance
from evaluate_pipeline import check_independent
from model_artifacts import sha256_file
from plate_validator import select_candidate
from prepare_indian_plates import (dataset_fingerprint, group_and_deduplicate,
                                   assign_group_splits, prepare, validate_build)


class PreparationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source'
        source.mkdir()
        policy={'source_url':'fixture','exclude_labels':{},'detection_only_labels':{},
                'missing_images':[],'filename_repairs':{},'dimension_repairs':{},
                'unannotated_variants':[]}
        for index in range(7):
            path=source/f'photo{index}.png'
            rng=random.Random(index)
            image=Image.new('RGB',(80,60))
            image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(80*60)])
            image.save(path)
            xml=(f'<annotation><filename>{path.name}</filename><size><width>80</width><height>60</height></size>'
                 f'<object><name>KA01AB{index:04}</name><bndbox><xmin>10</xmin><ymin>20</ymin>'
                 '<xmax>40</xmax><ymax>35</ymax></bndbox></object></annotation>')
            path.with_suffix('.xml').write_text(xml)
        policy['dataset_sha256']=dataset_fingerprint(list(source.iterdir()),source)
        policy_path=root/'policy.json'
        policy_path.write_text(json.dumps(policy))
        return source,policy_path

    def test_import_reads_text_preserves_source_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy=self.fixture(root)
            before=dataset_fingerprint(list(source.iterdir()),source)
            ml=root/'ml'
            with contextlib.redirect_stdout(io.StringIO()):
                build=prepare(source,ml/'prepared',policy,ml_dir=ml)
                repeated=prepare(source,ml/'prepared',policy,ml_dir=ml)
            self.assertEqual(build,repeated)
            self.assertEqual(before,dataset_fingerprint(list(source.iterdir()),source))
            entries=json.loads((build/'manifest.json').read_text())
            self.assertEqual({e['boxes'][0]['text'] for e in entries},{f'KA01AB{i:04}' for i in range(7)})
            self.assertTrue(all(e['full_image_labels_verified'] is False for e in entries))
            self.assertEqual(sum(validate_build(build,ml).values()),7)
            self.assertEqual({e['split'] for e in entries},{'train','val','test'})
            config=(ml/'prepared/data.yaml').read_text()
            self.assertIn("names: ['number_plate']",config)
            self.assertNotIn(str(root),config)
            self.assertTrue(all(ET.parse(p).findtext('object/name')=='number_plate'
                                for p in (build/'annotations').rglob('*.xml')))

    def test_changed_source_refuses_old_curation_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy=self.fixture(root)
            (source/'photo0.xml').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Source dataset changed'):
                prepare(source,root/'ml/prepared',policy,ml_dir=root/'ml')
            self.assertFalse((root/'ml/prepared/data.yaml').exists())

    def test_image_dimensions_and_stale_filename_are_repaired_only_when_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy_path=self.fixture(root)
            annotation=source/'photo0.xml'
            annotation.write_text(annotation.read_text().replace('photo0.png','stale.png').replace('<width>80</width>','<width>100</width>'))
            policy=json.loads(policy_path.read_text())
            policy['dataset_sha256']=dataset_fingerprint(list(source.iterdir()),source)
            policy['filename_repairs']={'photo0.xml':'photo0.png'}
            policy['dimension_repairs']={'photo0.xml':{'declared':[100,60],'actual':[80,60]}}
            policy_path.write_text(json.dumps(policy))
            with contextlib.redirect_stdout(io.StringIO()):
                build=prepare(source,root/'ml/prepared',policy_path,ml_dir=root/'ml')
            self.assertEqual(len(json.loads((build/'repairs.json').read_text())),2)
            entry=next(e for e in json.loads((build/'manifest.json').read_text()) if e['source_relative']=='photo0.png')
            self.assertEqual(entry['width'],80)
            self.assertEqual(entry['boxes'][0]['xmin'],10)

    def test_unreadable_text_stays_detection_only_and_is_not_ocr_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy_path=self.fixture(root)
            annotation=source/'photo0.xml'
            annotation.write_text(annotation.read_text().replace('KA01AB0000','blur'))
            policy=json.loads(policy_path.read_text())
            policy['detection_only_labels']={'BLUR':'unreadable'}
            policy['dataset_sha256']=dataset_fingerprint(list(source.iterdir()),source)
            policy_path.write_text(json.dumps(policy))
            with contextlib.redirect_stdout(io.StringIO()):
                build=prepare(source,root/'ml/prepared',policy_path,ml_dir=root/'ml')
            entries=json.loads((build/'manifest.json').read_text())
            ocr=json.loads((build/'ocr_manifest.json').read_text())
            self.assertEqual(len(entries),7)
            self.assertEqual(len(ocr),6)
            self.assertIsNone(next(e for e in entries if e['source_relative']=='photo0.png')['boxes'][0]['text'])

    def test_modified_labels_fail_build_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy=self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                build=prepare(source,root/'ml/prepared',policy,ml_dir=root/'ml')
            next((build/'labels').rglob('*.txt')).write_text('0 .5 .5 .2 .2\n')
            with self.assertRaisesRegex(ValueError,'box changed'):
                validate_build(build,root/'ml')

    def test_jpeg_trailer_is_normalized_losslessly_before_yolo_can_rewrite_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,policy_path=self.fixture(root)
            png=source/'photo0.png'
            jpeg=source/'photo0.jpg'
            with Image.open(png) as image:
                image.save(jpeg)
            # Trailing metadata after EOI is decodable but fails YOLO's trailer check.
            jpeg.write_bytes(jpeg.read_bytes()+b'trailing bytes')
            png.unlink()
            annotation=source/'photo0.xml'
            annotation.write_text(annotation.read_text().replace('photo0.png','photo0.jpg'))
            policy=json.loads(policy_path.read_text())
            policy['dataset_sha256']=dataset_fingerprint(list(source.iterdir()),source)
            policy_path.write_text(json.dumps(policy))
            with contextlib.redirect_stdout(io.StringIO()):
                build=prepare(source,root/'ml/prepared',policy_path,ml_dir=root/'ml')
            entry=next(e for e in json.loads((build/'manifest.json').read_text()) if e['source_relative']=='photo0.jpg')
            self.assertTrue(entry['image_path'].endswith('.png'))
            with Image.open(jpeg) as original, Image.open(root/'ml'/entry['image_path']) as repaired:
                self.assertEqual(original.convert('RGB').tobytes(),repaired.convert('RGB').tobytes())


class LeakageTests(unittest.TestCase):
    def record(self,index,text='KA01AB1234',sequence=None,pixels=None):
        return {'path':Path(f'{index}.jpg'), 'source_relative':sequence or f'google_images/{index}.jpg',
                'source_sha256':f'source{index}', 'pixel_sha256':pixels or f'pixels{index}',
                'dhash':(1 << (index*8))-1, 'boxes':[{'text':text,'raw_text':text}]}

    def test_same_plate_and_entire_sequence_stay_together(self):
        records=[self.record(1),self.record(2),
                 self.record(3,'MH12AB1234','video_images/video11_0001.jpg'),
                 self.record(4,'MH12AB1235','video_images/video11_0002.jpg'),
                 self.record(5,'TN01AB1234'),self.record(6,'TN01AB1235')]
        unique,_,_=group_and_deduplicate(records)
        result=assign_group_splits(unique)
        by={e['source_sha256']:e for e in result}
        self.assertEqual(by['source1']['split'],by['source2']['split'])
        self.assertEqual(by['source3']['split'],by['source4']['split'])
        self.assertNotEqual(by['source1']['source_id'],by['source2']['source_id'])

    def test_identical_pixels_are_deduplicated_and_alias_hashes_retained(self):
        records=[self.record(1,pixels='same'),self.record(2,pixels='same')]
        result,duplicates,_=group_and_deduplicate(records)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['duplicate_source_sha256'],['source1','source2'])
        self.assertEqual(len(duplicates),1)

    def test_uncertain_transcriptions_and_reviewed_aliases_also_share_a_group(self):
        records=[self.record(1,'KL01KLKL01'),self.record(2,'KL01KL0KL01'),
                 self.record(3,'TN01AB1234'),self.record(4,'TN01AB1235')]
        for record in records[:2]:
            record['boxes'][0].update(text=None,leakage_identity='KL01KLKL01')
        result,_,_=group_and_deduplicate(records)
        self.assertEqual(result[0]['leakage_group'],result[1]['leakage_group'])

    def test_explicit_related_sequences_cannot_cross_splits(self):
        entries=[{'image_path':'a.png','source_id':'a','leakage_group':'clip','split':'train'},
                 {'image_path':'b.png','source_id':'b','leakage_group':'clip','split':'val'}]
        with self.assertRaisesRegex(ValueError,'sequence crosses splits'):
            split_entries(entries)

    def test_source_identity_cannot_be_hidden_by_different_group_names(self):
        entries=[{'image_path':'a.png','source_id':'same','leakage_group':'one','split':'train'},
                 {'image_path':'b.png','source_id':'same','leakage_group':'two','split':'val'}]
        with self.assertRaisesRegex(ValueError,'Source photo crosses'):
            split_entries(entries)

    def test_pipeline_refuses_target_only_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'test.png').write_bytes(b'image')
            entry={'image_path':'test.png','boxes':[],'full_image_labels_verified':False}
            seen={'source_ids':set(),'source_sha256':set(),'image_sha256':set()}
            with self.assertRaisesRegex(ValueError,'Full-image labels'):
                check_independent([entry],seen,root)

    def test_training_provenance_includes_related_groups_and_duplicate_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'image.png').write_bytes(b'image')
            entry={'image_path':'image.png','source_id':'a','leakage_group':'sequence',
                   'source_sha256':'raw','duplicate_source_sha256':['copy'], 'pixel_sha256':'pixels', 'split':'train'}
            record=provenance([entry],root)
            self.assertIn('sequence',record['leakage_groups'])
            self.assertIn('copy',record['source_sha256'])
            self.assertIn('pixels',record['pixel_sha256'])


class LayoutTests(unittest.TestCase):
    def test_observed_extra_layouts_preserve_literal_text_for_review(self):
        for text in ('DL3CAY2231','DL8CAL4134','KL01CC50','TN02BL9','GJW115A1138'):
            with self.subTest(text=text):
                result=select_candidate([{'text':text,'conf':.99}])
                self.assertEqual(result['proposed_text'],text)
                self.assertEqual(result['status'],'uncertain')
                self.assertEqual(result['plate_text'],'')

    def test_extended_layout_does_not_make_display_text_a_registration(self):
        for text in ('CRETA','DUSTER','TERRANO','blur'):
            self.assertEqual(select_candidate([{'text':text,'conf':.99}])['status'],'unreadable')


if __name__=='__main__':
    unittest.main()
