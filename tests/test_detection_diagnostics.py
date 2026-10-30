"""Step 14.2 unit tests; no neural network weights or online dataset required."""
import json
import tempfile
import unittest
from pathlib import Path

from src.detection_diagnostics import (
    annotation_audit, apply_roi, box_stats, compare_predictions, inside_polygon,
    load_roi, validate_box, evaluate_setting,
)


class DetectionDiagnosticsTests(unittest.TestCase):
    def test_box_geometry(self):
        self.assertTrue(validate_box([0, 0, 100, 100], 1920, 1080))
        self.assertFalse(validate_box([-1, 0, 100, 100], 1920, 1080))
        self.assertFalse(validate_box([3, 3, 2, 2], 1920, 1080))

    def test_roi_polygon(self):
        square = [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]
        self.assertTrue(inside_polygon((.5, .5), square))
        self.assertTrue(inside_polygon((.1, .5), square))
        self.assertFalse(inside_polygon((.95, .5), square))

    def test_roi_applies_to_bottom_centre(self):
        square = [[0, 0], [.5, 0], [.5, 1], [0, 1]]
        people = [{"xyxy": [1, 1, 20, 50]}, {"xyxy": [70, 1, 90, 50]}]
        kept, ignored = apply_roi(people, square, 100, 100)
        self.assertEqual(len(kept), 1)
        self.assertEqual(ignored, 1)

    def test_separate_yolo_and_tracker_matches(self):
        gt = [{"gt_id": 1, "xyxy": [0, 0, 20, 20]},
              {"gt_id": 2, "xyxy": [40, 0, 60, 20]}]
        yolo = [{"xyxy": [0, 0, 20, 20]}, {"xyxy": [40, 0, 60, 20]}]
        tracks = [{"xyxy": [0, 0, 20, 20]}]
        d, t, diff, _ = compare_predictions(gt, yolo, tracks, image_size=(100, 100))
        self.assertEqual(d["matched"], 2)
        self.assertEqual(t["matched"], 1)
        self.assertEqual(diff["gt_matched_only_by_yolo"], 1)

    def test_no_ground_truth_precision_not_assumed_true(self):
        stats, _ = box_stats([], [{"xyxy": [0, 0, 10, 10]}])
        self.assertEqual(stats["precision_proxy"], 0.0)
        self.assertIsNone(stats["recall"])

    def test_roi_validation(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / 'roi.json'
            f.write_text(json.dumps({"C1": [[0, 0], [1, 0], [1, 1]]}))
            with self.assertRaisesRegex(ValueError, 'C1, C2'):
                load_roi(f)

    def test_mock_batched_inference_writes_diagnostics(self):
        import cv2
        import numpy as np
        from unittest.mock import patch

        class Tensor:
            def __init__(self, values):
                self.values = values
            def cpu(self):
                return self
            def tolist(self):
                return self.values
        class Result:
            def __init__(self):
                self.boxes = type("Boxes", (), {
                    "xyxy": Tensor([[10, 10, 30, 50]]), "conf": Tensor([0.9])})()
        class Detector:
            def predict(self, source, **kwargs):
                return [Result() for _ in source]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / 'annotations_positions').mkdir()
            (root / 'annotations_positions' / '00000000.json').write_text(json.dumps([
                {"personID": 2, "views": [
                    {"viewNum": i, "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 50}
                    for i in range(4)]}]))
            for cam in ('C1', 'C2', 'C3', 'C4'):
                d = root / 'Image_subsets' / cam
                d.mkdir(parents=True)
                cv2.imwrite(str(d / '00000000.png'), np.zeros((100, 150, 3), dtype=np.uint8))
            tracks = {0: {cam: [] for cam in ('C1', 'C2', 'C3', 'C4')}}
            with patch('torch.get_num_threads', return_value=4):
                report = evaluate_setting(Detector(), root, [0], 640, .1, .7, .5,
                                          None, tracks, root / 'out', examples=1)
            self.assertEqual(report['overall']['raw_yolo']['matched'], 4)
            self.assertEqual(report['overall']['post_tracker']['matched'], 0)
            self.assertEqual(report['per_camera']['C1']['difference']['gt_matched_only_by_yolo'], 1)
            self.assertTrue((root / 'out' / 'examples' / 'C1_00000000_size640.jpg').exists())

    def test_geometry_audit_fixture(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t)
            annotation_dir = folder / 'annotations_positions'
            annotation_dir.mkdir()
            ann = [{"personID": 2, "views": [
                {"viewNum": i, "xmin": 10, "ymin": 10, "xmax": 30, "ymax": 50}
                for i in range(4)]}]
            (annotation_dir / '00000000.json').write_text(json.dumps(ann))
            for cam in ('C1', 'C2', 'C3', 'C4'):
                d = folder / 'Image_subsets' / cam
                d.mkdir(parents=True)
                cv2.imwrite(str(d / '00000000.png'), np.zeros((100, 150, 3), dtype=np.uint8))
            report = annotation_audit(folder, [0])
            self.assertFalse(report['geometry_problems'])
            self.assertEqual(report['per_camera']['C1']['gt_boxes'], 1)
            self.assertEqual(report['image_dimensions']['C1'], [150, 100])


if __name__ == '__main__':
    unittest.main()
