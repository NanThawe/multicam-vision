"""Step 14.5 unit tests, no actual model weights or cameras required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.track_confirmation import (
    same_frame_unconfirmed, confirmed_from_rows, validated_detection_indices,
    frame_statistics, validate_cache, run_history, summarise, compare,
)


def det(box, conf=.9):
    return {"xyxy": box, "confidence": conf, "class_id": 0}


def person(tid, box, active=False, frame=1, idx=0, conf=.9):
    return SimpleNamespace(track_id=tid, xyxy=box, score=conf, idx=idx,
                           is_activated=active, frame_id=frame)


class FakeTracker:
    def __init__(self):
        self.frame_id = 0
        self.tracked_stracks = []


def update(tracker, frame, detections):
    tracker.frame_id += 1
    if not detections:
        tracker.tracked_stracks = []
        return []
    d = detections[0]
    # Synthetic tracker requires two additional updates before confirmation.
    activated = tracker.frame_id >= 3
    track = person(1, d["xyxy"], active=activated, frame=tracker.frame_id,
                   idx=0, conf=d["confidence"])
    tracker.tracked_stracks = [track]
    if activated:
        return [[*d["xyxy"], 1, d["confidence"], 0, 0]]
    return []


class TrackConfirmationTests(unittest.TestCase):
    def test_pending_only_on_current_frame(self):
        t = FakeTracker()
        t.frame_id = 3
        t.tracked_stracks = [person(1, [0,0,10,10], False, 3),
                             person(2, [0,0,10,10], True, 3),
                             person(3, [0,0,10,10], False, 2)]
        self.assertEqual([x["track_id"] for x in same_frame_unconfirmed(t)], [1])

    def test_rows_conversion(self):
        self.assertEqual(confirmed_from_rows([[1,2,3,4,9,.8,0,2]])[0]["idx"], 2)

    def test_original_detection_index_is_validated(self):
        ds = [det([0,0,10,10], .7), det([20,20,30,30], .9)]
        x = [{"xyxy": [20,20,30,30], "confidence": .9, "idx": 1}]
        self.assertEqual(validated_detection_indices(x, ds), {1})
        x[0]["idx"] = 0
        self.assertIsNone(validated_detection_indices(x, ds))

    def test_unconfirmed_increases_diagnostic_matching(self):
        ds = [det([0,0,20,20])]
        report = frame_statistics(0, "C1", ds, [{"xyxy": [0,0,20,20]}], [],
                                  [{"track_id": 1, "xyxy": [0,0,20,20],
                                    "confidence": .9, "idx": 0}])
        self.assertEqual(report["matched_confirmed"], 0)
        self.assertEqual(report["matched_with_unconfirmed"], 1)
        self.assertEqual(report["high_unaccounted"], 0)

    def test_unaccounted_is_not_claimed_as_a_rejection_cause(self):
        ds = [det([0,0,10,10]), det([20,20,30,30])]
        r = frame_statistics(0, "C1", ds, [], [], [], score=False)
        self.assertEqual(r["high_unaccounted"], 2)
        self.assertNotIn("matched_confirmed", r)

    def test_no_unverified_indices_claimed(self):
        r = frame_statistics(0, "C1", [det([0,0,10,10])], [],
                             [{"track_id": 1, "xyxy": [0,0,10,10],
                               "confidence": .9, "idx": None}], [], score=False)
        self.assertFalse(r["detection_indices_verified"])
        self.assertNotIn("high_unaccounted", r)

    def test_cache_must_be_chronological(self):
        records = [{"frame_id": fid, "detections": {c: [] for c in ('C1','C2','C3','C4')}}
                   for fid in (5,0)]
        with self.assertRaises(ValueError):
            validate_cache(records)

    def test_same_frame_history_experiment(self):
        box = [0,0,20,20]
        records = [{"frame_id": i * 5,
                    "detections": {c: [det(box)] for c in ('C1','C2','C3','C4')}}
                   for i in range(4)]
        truth = lambda fid, frame: {c: [{"xyxy": box}] for c in ('C1','C2','C3','C4')}
        images = lambda data, fid: {c: SimpleNamespace(shape=(50,50,3)) for c in ('C1','C2','C3','C4')}
        report, dense, sparse = compare(records, Path('.'), tracker_factory=FakeTracker,
                                        gt_loader=truth, image_loader=images, updater=update)
        self.assertEqual(report["full_2fps"]["scored_frame_sets"], 4)
        d = report["same_frame_history_test"]["2fps_history"]
        s = report["same_frame_history_test"]["1fps_history"]
        self.assertEqual(d["scored_frame_sets"], s["scored_frame_sets"])
        self.assertEqual(d["aggregate"]["gt"], s["aggregate"]["gt"])
        # Both modes have a new unconfirmed observation on their first input frame.
        self.assertGreater(d["aggregate"]["matched_confirmed"], s["aggregate"]["matched_confirmed"])

    def test_summary_tracks_index_availability(self):
        r = frame_statistics(0, "C1", [det([0,0,10,10])], [{"xyxy": [0,0,10,10]}], [], [], score=True)
        s = summarise([r])
        self.assertEqual(s["index_integrity"]["verified_observations"], 1)
        self.assertEqual(s["aggregate"]["gt"], 1)


if __name__ == '__main__':
    unittest.main()
