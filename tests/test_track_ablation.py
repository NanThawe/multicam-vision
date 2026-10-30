"""Lightweight ablation tests. No network or real YOLO model required."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.track_ablation import (
    SETTINGS, cache_spec, read_cache, run_ablation, run_setting,
    save_cache, tracker_config, validate_records, CAMERAS,
)


class FakeTracker:
    """Mimics a threshold-dependent track output for controlled pipeline tests."""

    def __init__(self, cameras, config):
        self.cameras = cameras
        self.config = config

    def update(self, frames, detections):
        return {
            c: [{"xyxy": d["xyxy"], "track_id": 7, "confidence": d["confidence"]}
                for d in detections[c] if d["confidence"] >= self.config.new_track_thresh]
            for c in self.cameras
        }


class AblationTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)
        self.data = self.root / "data"
        self.out = self.root / "out"
        for cam in CAMERAS:
            folder = self.data / "Image_subsets" / cam
            folder.mkdir(parents=True)
        (self.data / "annotations_positions").mkdir()
        self.ids = [0, 5]
        self.records = []
        for fid in self.ids:
            name = f"{fid:08d}"
            truth = [{"personID": 3, "views": [
                {"viewNum": i, "xmin": 2, "ymin": 3, "xmax": 12, "ymax": 18}
                for i in range(4)]}]
            (self.data / "annotations_positions" / f"{name}.json").write_text(
                json.dumps(truth), encoding="utf-8")
            for cam in CAMERAS:
                assert cv2.imwrite(str(self.data / "Image_subsets" / cam / f"{name}.png"),
                                   np.zeros((24, 24, 3), dtype=np.uint8))
            self.records.append({"frame_id": fid, "detections": {
                cam: [{"xyxy": [2, 3, 12, 18], "confidence": 0.20, "class_id": 0}]
                for cam in CAMERAS
            }})

    def test_baseline_thresholds_match_original_tracker(self):
        c = tracker_config(SETTINGS["baseline"])
        self.assertEqual((c.track_high_thresh, c.track_low_thresh, c.new_track_thresh),
                         (0.25, 0.1, 0.35))
        self.assertEqual((c.track_buffer, c.match_thresh, c.fuse_score), (10, 0.8, True))

    def test_variant_only_modifies_two_thresholds(self):
        for c in SETTINGS.values():
            v = vars(tracker_config(c))
            self.assertEqual(set(v), {"track_high_thresh", "track_low_thresh",
                                      "new_track_thresh", "track_buffer", "match_thresh", "fuse_score"})
            self.assertEqual(v["track_buffer"], 10)
            self.assertEqual(v["track_low_thresh"], .1)

    def test_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            tracker_config({"track_high_thresh": 0.25, "new_track_thresh": 0.20})

    def test_cache_round_trip(self):
        spec = {"frame_ids": self.ids, "weights_sha256": "placeholder"}
        save_cache(self.out, spec, self.records)
        self.assertEqual(read_cache(self.out, spec), self.records)

    def test_cache_mismatch_refuses_stale_records(self):
        save_cache(self.out, {"frame_ids": self.ids}, self.records)
        self.assertIsNone(read_cache(self.out, {"frame_ids": [0]}))

    def test_cache_checks_missing_camera(self):
        broken = [{"frame_id": 0, "detections": {"C1": []}}]
        with self.assertRaises(ValueError):
            validate_records(broken, [0])

    def test_cache_checks_duplicate_or_wrong_frame_ids(self):
        with self.assertRaises(ValueError):
            validate_records(self.records, [0, 0])

    def test_ground_truth_scoring_and_separate_tracker_state(self):
        baseline, _, _ = run_setting(self.records, self.data, SETTINGS["baseline"], FakeTracker)
        relaxed, _, _ = run_setting(self.records, self.data, SETTINGS["permissive"], FakeTracker)
        self.assertEqual(baseline["overall"]["matched_boxes"], 0)
        self.assertEqual(relaxed["overall"]["matched_boxes"], 8)
        self.assertEqual(relaxed["overall"]["observed_local_id_changes"], 0)
        self.assertEqual(relaxed["per_camera"]["C1"]["matched_boxes"], 2)

    def test_comparison_outputs_and_deltas(self):
        result = run_ablation(self.data, self.out, self.ids, self.records,
                              export=False, tracker_factory=FakeTracker)
        self.assertEqual(result["settings"]["permissive"]["versus_baseline"]["matched_boxes_difference"], 8)
        self.assertTrue((self.out / "comparison.json").exists())
        self.assertTrue((self.out / "moderate" / "metrics.json").exists())
        self.assertEqual(result["settings"]["moderate"]["overall"]["matched_boxes"], 0)

    def test_cache_spec_has_weights_hash(self):
        weights = self.root / "weights.pt"
        weights.write_bytes(b"test-model")
        s = cache_spec(self.ids, self.data, weights)
        self.assertEqual(len(s["weights_sha256"]), 64)
        self.assertEqual(s["imgsz"], 640)


if __name__ == "__main__":
    unittest.main()
