"""Unit tests for the end-to-end Video ANPR Tracking Pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
import cv2

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from track_and_read_video import VideoANPRTracker


class VideoTrackingPipelineTests(unittest.TestCase):
    def test_synthetic_video_passage_end_to_end(self):
        # Generate a small 6-frame synthetic video
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_file = tmp_path / "test_vehicle_passage.mp4"
            out_json = tmp_path / "passage_results.json"
            out_video = tmp_path / "annotated_passage.mp4"

            width, height, fps = 640, 480, 10
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

            for frame_num in range(6):
                frame = np.full((height, width, 3), 40, dtype=np.uint8)
                # Simulated moving car: bounding box [vx1, vy1, vx2, vy2]
                vx1 = 150 + frame_num * 10
                vy1 = 120 + frame_num * 5
                vx2 = vx1 + 280
                vy2 = vy1 + 220
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (180, 180, 180), -1)

                # Simulated license plate inside vehicle: [px1, py1, px2, py2]
                px1 = vx1 + 60
                py1 = vy2 - 50
                px2 = px1 + 140
                py2 = py1 + 35
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), -1)
                # Plate text simulation
                cv2.putText(frame, "KA01AB1234", (px1 + 8, py1 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                writer.write(frame)

            writer.release()

            self.assertTrue(video_file.is_file())

            # Initialize tracker (frame_stride=1 for short test video)
            tracker = VideoANPRTracker(frame_stride=1, max_ocr_per_track=3)
            result = tracker.process_video(
                video_file,
                out_json=out_json,
                out_video=out_video,
                conf_vehicle=0.1,  # low threshold for synthetic boxes
                conf_plate=0.1,
            )

            # Verify output structure and execution
            self.assertIn("video_metadata", result)
            self.assertEqual(result["video_metadata"]["total_frames"], 6)
            self.assertIn("passages", result)
            self.assertTrue(out_json.is_file())
            self.assertTrue(out_video.is_file())

            # Verify JSON serializability
            parsed_json = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed_json["video_metadata"]["total_frames"], 6)


if __name__ == "__main__":
    unittest.main()
