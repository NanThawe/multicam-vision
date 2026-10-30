"""Deterministic tests for visual error analysis; no model or network needed."""
import json
import tempfile
import unittest
from pathlib import Path

from src.error_analysis import (analyse_records, choose_examples, classify_frame,
                                read_predictions, run, size_group)


class ErrorAnalysisTest(unittest.TestCase):
    def test_height_buckets(self):
        self.assertEqual(size_group([0, 0, 10, 63]), "small")
        self.assertEqual(size_group([0, 0, 10, 64]), "medium")
        self.assertEqual(size_group([0, 0, 10, 160]), "large")

    def test_one_to_one_and_overlap_classification(self):
        gt = [{"gt_id": 1, "xyxy": [0, 0, 10, 10]},
              {"gt_id": 2, "xyxy": [50, 0, 60, 10]}]
        pred = [{"xyxy": [0, 0, 10, 10]}, {"xyxy": [0, 0, 10, 10]},
                {"xyxy": [70, 0, 80, 10]}]
        result = classify_frame(gt, pred)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(len(result["unmatched_gt"]), 1)
        self.assertEqual(len(result["unmatched_pred"]), 2)
        self.assertEqual(result["unmatched_overlap"]["overlap_gt_at_least_threshold"], 1)
        self.assertEqual(result["unmatched_overlap"]["little_or_no_gt_overlap"], 1)

    def make_fixture(self, folder):
        import cv2
        import numpy as np
        root = Path(folder)
        (root / "annotations_positions").mkdir()
        predictions = root / "predictions.jsonl"
        gt_views = [{"viewNum": i, "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 100}
                    for i in range(4)]
        (root / "annotations_positions" / "00000000.json").write_text(
            json.dumps([{"personID": 1, "views": gt_views}]), encoding="utf-8")
        with predictions.open("w", encoding="utf-8") as file:
            for camera in ("C1", "C2", "C3", "C4"):
                (root / "Image_subsets" / camera).mkdir(parents=True)
                cv2.imwrite(str(root / "Image_subsets" / camera / "00000000.png"),
                            np.zeros((108, 192, 3), dtype=np.uint8))
                tracks = [{"xyxy": [10, 10, 30, 100], "track_id": 5,
                           "global_id": 1, "confidence": .9}] if camera != "C4" else []
                file.write(json.dumps({"frame_id": 0, "camera": camera, "tracks": tracks}) + "\n")
        return predictions

    def test_load_requires_all_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text('{"frame_id":0,"camera":"C1","tracks":[]}\n')
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_predictions(path)

    def test_summary_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_fixture(tmp)
            preds = read_predictions(path)
            report, frames, _ = analyse_records(preds, tmp)
            self.assertEqual(report["overall"]["gt"], 4)
            self.assertEqual(report["overall"]["matched"], 3)
            self.assertEqual(report["overall"]["recall"], 0.75)
            self.assertEqual(report["per_camera"]["C4"]["missed"], 1)
            self.assertEqual(report["per_camera"]["C1"]["gt_by_height"]["medium"]["matched"], 1)
            self.assertEqual(len(frames), 4)

    def test_writes_png_overlays_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_fixture(tmp)
            out = Path(tmp) / "outputs"
            report = run(path, tmp, out, examples=1)
            self.assertEqual(report["overall"]["matched"], 3)
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "per_frame.csv").is_file())
            for cam in ("C1", "C2", "C3", "C4"):
                self.assertTrue((out / "examples" / f"{cam}_frame_00000000.jpg").is_file())

    def test_example_selection_prefers_misses(self):
        rows = [{"camera": "C1", "frame_id": 0, "gt": 5, "missed": 5, "unmatched": 0},
                {"camera": "C1", "frame_id": 5, "gt": 1, "missed": 0, "unmatched": 3},
                {"camera": "C1", "frame_id": 10, "gt": 2, "missed": 2, "unmatched": 0}]
        self.assertEqual(choose_examples(rows, "C1", 2), [0, 5])


if __name__ == "__main__":
    unittest.main()
