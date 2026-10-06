"""Verify target-only OCR diagnostics cannot mislabel omitted scene plates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_plate_pipeline import score_targets, summarize_targets


class TargetDiagnosticTests(unittest.TestCase):
    def entry(self, text='KA01AB1234'):
        return {'boxes':[dict(xmin=10,ymin=10,xmax=100,ymax=30,text=text)]}

    def prediction(self,text='KA01AB1234',status='accepted',box=None):
        return dict(box=box or [10,10,100,30],status=status,
                    plate_text=text if status=='accepted' else '',proposed_text=text,
                    readings=[dict(raw_ocr_regions=[dict(text=text)])])

    def test_unannotated_accepted_plate_is_not_called_wrong(self):
        result={'plates':[self.prediction(),self.prediction('MH02AB5678',box=[110,10,200,30])]}
        counts,_=score_targets(self.entry(),result)
        metrics=summarize_targets([counts])
        self.assertEqual(counts['wrong_accepted_targets'],0)
        self.assertEqual(counts['unmatched_accepted_predictions'],1)
        self.assertEqual(metrics['accepted_target_reading_precision'],1)

    def test_missed_target_stays_in_exact_recall_denominator(self):
        missed,_=score_targets(self.entry(),{'plates':[]})
        found,_=score_targets(self.entry(),{'plates':[self.prediction()]})
        metrics=summarize_targets([missed,found])
        self.assertEqual(metrics['targets'],2)
        self.assertEqual(metrics['missed_targets'],1)
        self.assertEqual(metrics['exact_accepted_target_recall'],.5)

    def test_correct_review_proposal_is_not_an_accepted_success(self):
        counts,details=score_targets(self.entry(),{'plates':[self.prediction(status='uncertain')]})
        self.assertEqual(counts['correct_accepted_targets'],0)
        self.assertEqual(counts['correct_uncertain_targets'],1)
        self.assertEqual(counts['exact_proposals'],1)
        self.assertEqual(details[0]['outcome'],'correct_uncertain')
        self.assertIsNone(summarize_targets([counts])['accepted_target_reading_precision'])

    def test_wrong_accepted_reading_is_reported(self):
        counts,_=score_targets(self.entry(),{'plates':[self.prediction('KA01AB1235')]})
        self.assertEqual(counts['wrong_accepted_targets'],1)
        self.assertEqual(summarize_targets([counts])['accepted_target_reading_precision'],0)

    def test_duplicate_predictions_cannot_both_match_one_target(self):
        counts,_=score_targets(self.entry(),{'plates':[self.prediction(),self.prediction()]})
        self.assertEqual(counts['matched_targets'],1)
        self.assertEqual(counts['correct_accepted_targets'],1)
        self.assertEqual(counts['unmatched_predictions'],1)

    def test_missing_transcription_fails(self):
        with self.assertRaisesRegex(ValueError,'transcription'):
            score_targets(self.entry(None),{'plates':[]})


if __name__ == '__main__':
    unittest.main()
