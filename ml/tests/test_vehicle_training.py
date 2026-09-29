import sys
import unittest
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_vehicle import score, iou
from train_vehicle import prepare


class EvaluationTests(unittest.TestCase):
    def test_wrong_class_is_false_positive_and_missed_truth(self):
        counts=score([('truck',.9,[0,0,100,100])],[('other',[0,0,100,100])])
        self.assertEqual(counts['truck']['fp'],1)
        self.assertEqual(counts['other']['fn'],1)

    def test_duplicate_predictions_cannot_claim_same_truth(self):
        counts=score([('car',.8,[0,0,100,100]),('car',.9,[0,0,100,100])],[('car',[0,0,100,100])])
        self.assertEqual(dict(counts['car']),dict(tp=1,fp=1,fn=0))

    def test_iou_and_missing_detection(self):
        self.assertEqual(iou([0,0,10,10],[10,10,20,20]),0)
        self.assertEqual(score([],[('bus',[0,0,10,10])])['bus']['fn'],1)

    def test_pairing_repair_refuses_unverified_dataset_release(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'revision'):
                prepare(Path(directory),Path(directory)/'out')


if __name__=='__main__':
    unittest.main()
