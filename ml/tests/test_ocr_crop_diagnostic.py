from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_ocr_crops import crop_box, literal_evidence, score_readings, summarize
from plate_validator import select_candidate


def reading(text, confidence=.99):
    regions=[{'text':text,'conf':confidence}]
    return {**select_candidate(regions),'raw_ocr_regions':regions}


class OCRCropDiagnosticTests(unittest.TestCase):
    def test_review_proposals_are_not_counted_as_accepted_successes(self):
        result=score_readings([reading('PY036993')],'PY036993')
        self.assertEqual(result['outcome'],'correct_uncertain')
        self.assertTrue(result['literal_exact_available'])
        self.assertTrue(result['exact_proposal'])

    def test_literal_evidence_preserves_digits_and_handles_split_country_badge(self):
        regions=[{'text':t,'conf':.99} for t in ('IND','KA01','AB','1234')]
        self.assertTrue(literal_evidence([{'raw_ocr_regions':regions}],'KA01AB1234'))
        self.assertFalse(literal_evidence([{'raw_ocr_regions':[{'text':'PY03G993'}]}],'PY036993'))

    def test_matched_crop_denominator_excludes_detector_misses(self):
        good=score_readings([reading('KA01AB1234')],'KA01AB1234')
        rows=[{'predicted':good},{'predicted':None}]
        self.assertEqual(summarize(rows,'predicted')['crops'],1)
        self.assertEqual(summarize(rows,'predicted')['exact_accepted_fraction'],1)

    def test_wrong_accepted_text_is_not_a_review_or_correct_reading(self):
        result=score_readings([reading('KA01AB1235')],'KA01AB1234')
        self.assertEqual(result['outcome'],'wrong_accepted')
        self.assertFalse(result['exact_proposal'])

    def test_fractional_boxes_round_outward_and_clip_context_to_image(self):
        self.assertEqual(crop_box([.5,.5,19.2,9.1],20,10),[0,0,20,10])
        for invalid in ([0,0,0,10],[-1,0,10,10],[0,0,21,10],[0,0,float('nan'),10]):
            with self.assertRaises(ValueError):
                crop_box(invalid,20,10)


if __name__=='__main__':
    unittest.main()
