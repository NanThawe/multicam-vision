"""Small deterministic checks of WILDTRACK evaluation semantics."""
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate import available_frames, compute_metrics, identity_f1, iou, parse_gt, export_trackeval


class EvaluationTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)

    def test_annotation_camera_mapping_and_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations_positions" / "00000000.json"
            path.parent.mkdir()
            path.write_text(json.dumps([{"personID": 5, "views": [
                {"viewNum": 0, "xmin": -2, "ymin": 2, "xmax": 9, "ymax": 12},
                {"viewNum": 1, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
            ]}]), encoding="utf-8")
            result = parse_gt(tmp, 0, width=10, height=10)
            self.assertEqual(result["C1"], [{"gt_id": 5, "xyxy": [0, 2, 9, 10]}])
            self.assertEqual(result["C2"], [])

    def test_only_common_frames_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cam in ("C1", "C2", "C3", "C4"):
                folder = root / "Image_subsets" / cam
                folder.mkdir(parents=True)
                (folder / "00000000.png").touch()
            (root / "annotations_positions").mkdir()
            (root / "annotations_positions" / "00000000.json").write_text("[]")
            self.assertEqual(available_frames(root, 20), [0])

    def test_cross_camera_correct_same_global_id(self):
        gt = {0: {"C1": [{"gt_id": 8, "xyxy": [0, 0, 10, 10]}],
                  "C2": [{"gt_id": 8, "xyxy": [0, 0, 10, 10]}]}}
        pred = {0: {"C1": [{"track_id": 1, "global_id": 3, "xyxy": [0, 0, 10, 10]}],
                    "C2": [{"track_id": 6, "global_id": 3, "xyxy": [0, 0, 10, 10]}]}}
        stats = compute_metrics(gt, pred)["cross_camera"]
        self.assertEqual(stats["pairwise_precision"], 1)
        self.assertEqual(stats["pairwise_recall"], 1)

    def test_cross_camera_split_global_id(self):
        gt = {0: {"C1": [{"gt_id": 8, "xyxy": [0, 0, 10, 10]}],
                  "C2": [{"gt_id": 8, "xyxy": [0, 0, 10, 10]}]}}
        pred = {0: {"C1": [{"track_id": 1, "global_id": 3, "xyxy": [0, 0, 10, 10]}],
                    "C2": [{"track_id": 6, "global_id": 7, "xyxy": [0, 0, 10, 10]}]}}
        stats = compute_metrics(gt, pred)["cross_camera"]
        self.assertEqual(stats["pairwise_recall"], 0)
        self.assertIsNone(stats["pairwise_precision"])

    def test_cross_camera_false_positive(self):
        gt = {0: {"C1": [{"gt_id": 8, "xyxy": [0, 0, 10, 10]}],
                  "C2": [{"gt_id": 9, "xyxy": [0, 0, 10, 10]}]}}
        pred = {0: {"C1": [{"track_id": 1, "global_id": 3, "xyxy": [0, 0, 10, 10]}],
                    "C2": [{"track_id": 6, "global_id": 3, "xyxy": [0, 0, 10, 10]}]}}
        stats = compute_metrics(gt, pred)["cross_camera"]
        self.assertEqual(stats["predicted_same_identity_pairs"], 1)
        self.assertEqual(stats["pairwise_precision"], 0)

    def test_identity_f1(self):
        self.assertEqual(identity_f1({(1, 2): 2}, 2, 2), 1)
        self.assertEqual(identity_f1({}, 2, 2), 0)

    def test_trackeval_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = {0: {"C1": [{"gt_id": 8, "xyxy": [0, 1, 10, 20]}]}}
            pred = {0: {"C1": [{"track_id": 2, "confidence": .8,
                                 "xyxy": [0, 1, 10, 20]}]}}
            export_trackeval(gt, pred, tmp)
            root = Path(tmp)
            gt_file = root / "gt/mot_challenge/WILDTRACK-EVAL-train/C1/gt/gt.txt"
            pr_file = root / "trackers/mot_challenge/WILDTRACK-EVAL-train/multicam/data/C1.txt"
            self.assertEqual(gt_file.read_text().strip().split(",")[:2], ["1", "9"])
            self.assertEqual(pr_file.read_text().strip().split(",")[:2], ["1", "3"])

    def test_rejects_frame_misalignment(self):
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            compute_metrics({0: {}}, {5: {}})


if __name__ == "__main__":
    unittest.main()
