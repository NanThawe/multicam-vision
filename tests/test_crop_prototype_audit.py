"""Synthetic Step 14.24 tests. These do not replace Windows/WILDTRACK validation."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import crop_prototype_audit as d
from src.association_ablation import Policy


def box(x1, y1, x2, y2):
    return {"xyxy": (float(x1), float(y1), float(x2), float(y2)), "gt_id": 7}


class GeometryTests(unittest.TestCase):
    def test_geometry_original_vs_new_border(self):
        gt = {0: {"C1": [box(10, 15, 110, 200), box(0, 200, 50, 300)],
                  "C2": [box(1500, 20, 1910, 200)], "C3": [], "C4": []}}
        out, examples = d.border_geometry(gt, paddings=(0.1,), width=1920, height=1080)
        self.assertEqual(out["C1"]["original_touch_any"], 1)
        self.assertEqual(out["C1"]["variants"]["pad_10"]["touch_any"], 2)
        self.assertEqual(out["C1"]["variants"]["pad_10"]["new_touch_any"], 1)
        self.assertEqual(out["C2"]["variants"]["pad_10"]["new_touch_by_edge"]["right"], 1)
        self.assertTrue(any(x["camera"] == "C2" for x in examples))

    def test_edge_directions(self):
        self.assertEqual(d._edges((0, 0, 1920, 1080)),
                         {"left": True, "top": True, "right": True, "bottom": True})
        self.assertFalse(any(d._edges((10, 20, 100, 200)).values()))

    def test_geometry_negative_or_invalid_box(self):
        gt = {0: {"C1": [box(-1, 0, 10, 10)]}}
        with self.assertRaises(ValueError): d.border_geometry(gt)
        gt = {0: {"C1": [box(1, 1, 1930, 10)]}}
        with self.assertRaises(ValueError): d.border_geometry(gt)

    def test_zero_records_are_safe(self):
        out, examples = d.border_geometry({0: {}})
        self.assertEqual(out["C2"]["crops"], 0)
        self.assertEqual(out["C2"]["original_margin_px"]["left"]["p50"], None)
        self.assertEqual(examples, [])

    def test_raw_annotation_clipped_matches_cached(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "annotations_positions").mkdir()
            source = [{"personID": 7, "views": [{"viewNum": 1, "xmin": -5, "ymin": 30,
                                                   "xmax": 75, "ymax": 1090}]}]
            (root / "annotations_positions" / "00000000.json").write_text(json.dumps(source))
            gt = {0: {"C2": [box(0, 30, 75, 1080)]}}
            read = lambda _: np.empty((1080, 1920, 3), dtype=np.uint8)
            a = d.inspect_raw_annotations(root, gt, read)
            self.assertEqual(a["eligible_box_verified_count"]["C2"], 1)
            self.assertEqual(a["raw_annotation_box_audit"]["C2"]["source_outside_left"], 1)
            self.assertEqual(a["raw_annotation_box_audit"]["C2"]["source_outside_bottom"], 1)
            self.assertEqual(a["sample_image_dimensions_wh"]["C2"], [1920, 1080])

    def test_raw_annotation_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "annotations_positions").mkdir()
            (root / "annotations_positions" / "00000000.json").write_text(
                json.dumps([{"personID": 7, "views": [{"viewNum": 0, "xmin": 20,
                    "ymin": 30, "xmax": 75, "ymax": 80}]}]))
            gt = {0: {"C1": [box(21, 30, 75, 80)]}}
            read = lambda _: np.empty((1080, 1920, 3), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                d.inspect_raw_annotations(root, gt, read)

    def test_wrong_image_dimensions_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                d.inspect_raw_annotations(root, {0: {"C1": []}},
                    lambda _: np.empty((720, 1280, 3), dtype=np.uint8))

    def test_missing_image_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                d.inspect_raw_annotations(td, {0: {"C1": []}}, lambda _: None)


class ReplayTests(unittest.TestCase):
    @staticmethod
    def frames():
        a = np.array([1., 0., 0.], np.float32)
        b = np.array([.999, .0447, 0.], np.float32)
        b /= np.linalg.norm(b)
        out = {}
        for fid in (0, 5, 10, 15, 20):
            out[fid] = {"C1": [(116, (10., 10., 40., 70.), a)],
                        "C2": [(115, (80., 10., 110., 70.), b)],
                        "C3": [], "C4": []}
        return out

    def test_prototype_history_catches_wrong_merge(self):
        policy = Policy("toy", threshold=.8, margin=.05,
                        prototype="camera_balanced", assignment="batch_hungarian")
        matcher, preds, labels = d.replay_history(self.frames(), policy, 1)
        self.assertEqual(len(matcher.history) > 0, True)
        self.assertTrue(any(x["action"] == "attach" and x["distinct_origin_gt_people"] == 2
                            for x in matcher.history))
        self.assertTrue(any(r["wrong_gt_merge_at_assignment"] == 1 for r in matcher.audit))
        self.assertTrue(d.false_pair_timeline(preds, labels))
        self.assertEqual(len(matcher.globals), 1)

    def test_trace_summarizes_past_and_contamination(self):
        policy = Policy("toy", threshold=.8, margin=.05,
                        prototype="camera_balanced", assignment="batch_hungarian")
        matcher, _, _ = d.replay_history(self.frames(), policy, 1)
        report = d.trace_summary(matcher.history, 10, 1)
        self.assertEqual(report["focus_global_id"], 1)
        self.assertEqual(report["first_heterogeneous_prototype_event"]["action"], "attach")
        self.assertEqual(report["first_heterogeneous_prototype_event"]["frame_id"], 10)
        self.assertGreater(report["prototype_history_events"], 1)

    def test_focus_does_not_affect_replay(self):
        policy = Policy("toy", threshold=.8, margin=.05,
                        prototype="camera_balanced", assignment="batch_hungarian")
        one, a, _ = d.replay_history(self.frames(), policy, 1)
        missing, b, _ = d.replay_history(self.frames(), policy, 999)
        self.assertEqual(a, b)
        self.assertEqual(missing.history, [])
        self.assertTrue(one.history)

    def test_fresh_embedding_invariance(self):
        policy = Policy("toy", threshold=.8, margin=.05,
                        prototype="camera_balanced", assignment="batch_hungarian")
        matcher, _, _ = d.replay_history(self.frames(), policy, 1)
        updates = [x for x in matcher.history if x["action"] == "update"]
        self.assertTrue(updates)
        self.assertTrue(all(x["prototype_norm"] is not None for x in updates))
        self.assertTrue(any(x["reuses_cached_feature"] is True for x in updates))

    def test_csv_writer(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.csv"
            d.write_csv(p, [{"a": 2, "b": 7}], ["a"])
            self.assertEqual(p.read_text().splitlines(), ["a", "2"])

    def test_cosine_format(self):
        self.assertIsNone(d._cosine(None, None) if False else d._rounded(None))
        self.assertAlmostEqual(d._cosine(np.array([1., 0.]), np.array([.5, 0.])), .5)


if __name__ == "__main__":
    unittest.main()
