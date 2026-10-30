"""Step 14.4 unit tests. No videos, GPUs or Ultralytics required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.track_lifecycle import (
    confidence_counts, state_ids, transition_counts, output_indices,
    LifecycleMeter, summarise, write_results,
)


def track(idx, active=True):
    return SimpleNamespace(track_id=idx, is_activated=active)


class FakeTracker:
    def __init__(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []


class LifecycleTests(unittest.TestCase):
    def test_confidence_bands_are_disjoint(self):
        r = confidence_counts([{"confidence": c} for c in (.05, .10, .12, .24, .25, .4)])
        self.assertEqual(r["detections"], 6)
        self.assertEqual(r["high_score_detections"], 2)
        self.assertEqual(r["low_score_detections"], 2)
        self.assertEqual(r["at_or_below_low_threshold"], 2)
        self.assertEqual(r["below_new_track_threshold"], 5)

    def test_state_distinguishes_active_and_unconfirmed(self):
        t = FakeTracker()
        t.tracked_stracks = [track(1), track(2, False)]
        t.lost_stracks = [track(3)]
        t.removed_stracks = [track(4)]
        self.assertEqual(state_ids(t), {
            "active": {1}, "unconfirmed": {2}, "lost": {3}, "removed": {4}
        })

    def test_transition_tracks_new_lost_and_reactivated(self):
        before = {"active": {1}, "unconfirmed": set(), "lost": {2}, "removed": set()}
        after = {"active": {2, 3}, "unconfirmed": set(), "lost": {1}, "removed": {4}}
        result = transition_counts(before, after, {1, 2}, set())
        self.assertEqual((result["new_track_ids"], result["became_lost"],
                          result["reactivated"], result["newly_removed"]), (1, 1, 1, 1))

    def test_original_detection_indices_available(self):
        rows = [[1, 2, 3, 4, 7, .9, 0, 0], [1, 2, 3, 4, 8, .2, 0, 2]]
        self.assertEqual(output_indices(rows, 3), {0, 2})
        self.assertIsNone(output_indices([rows[0][:7]], 3))
        self.assertIsNone(output_indices([[1,2,3,4,7,.9,0,99]], 3))

    def test_meter_counts_returned_detections_by_confidence(self):
        t = FakeTracker()
        m = LifecycleMeter(t)
        t.tracked_stracks = [track(5)]
        detections = [
            {"confidence": .5}, {"confidence": .2}, {"confidence": .9}]
        rows = [[1,2,3,4,5,.5,0,0], [2,3,4,5,6,.2,0,1]]
        r, output = m.record(5, detections, rows)
        self.assertEqual(r["new_track_ids"], 1)
        self.assertEqual(r["returned_high_detections"], 1)
        self.assertEqual(r["returned_low_detections"], 1)
        self.assertEqual(r["high_not_in_output"], 1)
        self.assertEqual(len(output), 2)

    def test_meter_does_not_invent_missing_indices(self):
        t = FakeTracker()
        m = LifecycleMeter(t)
        stats, _ = m.record(0, [{"confidence": .5}], [[1,2,3,4,1,.5,0]])
        self.assertFalse(stats["detection_indices_available"])
        self.assertNotIn("high_not_in_output", stats)

    def test_meter_tracks_loss_and_recovery(self):
        t = FakeTracker()
        m = LifecycleMeter(t)
        t.tracked_stracks = [track(1)]
        m.record(0, [], [])
        t.tracked_stracks = []
        t.lost_stracks = [track(1)]
        lost, _ = m.record(5, [], [])
        self.assertEqual(lost["became_lost"], 1)
        t.tracked_stracks = [track(1)]
        t.lost_stracks = []
        returned, _ = m.record(10, [], [])
        self.assertEqual(returned["reactivated"], 1)
        self.assertEqual(returned["new_track_ids"], 0)

    def test_summary_and_output_files(self):
        base = {"detections": 4, "high_score_detections": 3,
                "low_score_detections": 1, "at_or_below_low_threshold": 0,
                "below_new_track_threshold": 1, "output_tracks": 2,
                "new_track_ids": 1, "became_lost": 0, "reactivated": 0,
                "newly_removed": 0, "unconfirmed_tracks": 0,
                "lost_tracks": 0, "detection_indices_available": False,
                "gt_boxes": 3, "matched_boxes": 2, "frame_id": 0}
        rows = [{**base, "camera": c} for c in ("C1", "C2", "C3", "C4")]
        s = summarise(rows, "annotated_2fps", True)
        self.assertEqual(s["aggregate"]["detections"], 16)
        self.assertEqual(s["aggregate"]["matched_gt_recall"], .6667)
        self.assertNotIn("high_not_in_output", s["per_camera"]["C1"])
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root)
            write_results(dest, {"annotated": {"test": s}}, rows)
            self.assertEqual(json.loads((dest / "summary.json").read_text())["annotated"]["test"]["aggregate"]["detections"], 16)
            self.assertTrue((dest / "per_frame.csv").is_file())


if __name__ == "__main__":
    unittest.main()
