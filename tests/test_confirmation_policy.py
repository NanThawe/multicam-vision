"""Step 14.6 controlled experiment tests. No Ultralytics or model weights required."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.confirmation_policy import (
    normalise_confirmed, provisional_candidates, promote_current_candidates,
    score_observations, run_pass, compare_policies, summarise_policy, POLICIES,
)


BOX = [5.0, 5.0, 25.0, 25.0]
CAMERAS = ("C1", "C2", "C3", "C4")


def det(box=BOX):
    return {"xyxy": list(box), "confidence": 0.95, "class_id": 0}


def fake_images(_data, _fid):
    return {cam: SimpleNamespace(shape=(100, 100, 3)) for cam in CAMERAS}


def fake_gt(_fid, _image):
    return {cam: [{"gt_id": 100, "xyxy": list(BOX)}] for cam in CAMERAS}


class FakeTracker:
    def __init__(self):
        self.frame_id = 0
        self.tracked_stracks = []
        self._next_id = 10


def fake_update(tracker, _frame, detections):
    tracker.frame_id += 1
    if not detections:
        tracker.tracked_stracks = []
        return []
    old = tracker.tracked_stracks[0] if tracker.tracked_stracks else None
    if old and old.is_activated:
        old.frame_id = tracker.frame_id
        return [[*BOX, old.track_id, old.score, 0, 0]]
    # A newly observed unconfirmed track is not in normal update() output.
    tracker._next_id += 1
    tracker.tracked_stracks = [SimpleNamespace(
        xyxy=list(BOX), score=0.95, track_id=tracker._next_id, idx=0,
        is_activated=False, frame_id=tracker.frame_id)]
    return []


def records(n=3):
    return [{"frame_id": i * 5, "detections": {cam: [det()] for cam in CAMERAS}}
            for i in range(n)]


class ConfirmationPolicyTests(unittest.TestCase):
    def test_provisional_cannot_expose_identity(self):
        tracker = FakeTracker()
        fake_update(tracker, None, [det()])
        p = provisional_candidates(tracker)
        self.assertEqual(p[0]["status"], "provisional")
        self.assertIsNone(p[0]["track_id"])

    def test_provisional_only_current_frame(self):
        tracker = FakeTracker()
        fake_update(tracker, None, [det()])
        tracker.frame_id += 1
        self.assertEqual(provisional_candidates(tracker), [])

    def test_promote_mutates_only_current_pending(self):
        tracker = FakeTracker()
        fake_update(tracker, None, [det()])
        previous = tracker.tracked_stracks[0]
        stale = SimpleNamespace(xyxy=BOX, score=.8, track_id=88,
                                is_activated=False, frame_id=0)
        tracker.tracked_stracks.append(stale)
        out = promote_current_candidates(tracker)
        self.assertEqual(len(out), 1)
        self.assertTrue(previous.is_activated)
        self.assertFalse(stale.is_activated)
        self.assertEqual(out[0]["status"], "experimental_early_confirmed")
        self.assertEqual(promote_current_candidates(tracker), [])

    def test_rows_normalized_and_score(self):
        c = normalise_confirmed([[*BOX, 7, .9, 0, 0]])
        self.assertEqual(c[0]["track_id"], 7)
        self.assertEqual(c[0]["status"], "confirmed")
        r, matches = score_observations([{"xyxy": BOX}], c, c)
        self.assertEqual(r["matched_boxes"], 1)
        self.assertEqual(matches, [(0, 0)])

    def test_extra_matches_from_provisional_are_not_identity_matches(self):
        gt = [{"xyxy": BOX}]
        p = [{"xyxy": BOX, "confidence": .9,
              "track_id": None, "status": "provisional"}]
        r, identities = score_observations(gt, [], p)
        self.assertEqual(r["extra_matches_beyond_confirmed"], 1)
        self.assertEqual(r["provisional_matches"], 1)
        self.assertEqual(identities, [])

    def test_provisional_does_not_replace_confirmed_identity_match(self):
        gt = [{"xyxy": BOX}]
        c = [{"xyxy": [6, 6, 26, 26], "confidence": .9,
              "track_id": 8, "status": "confirmed"}]
        p = [{"xyxy": BOX, "confidence": .95, "track_id": None,
              "status": "provisional"}]
        r, identities = score_observations(gt, c, c + p)
        self.assertEqual(r["matched_boxes"], 1)
        self.assertEqual(identities, [(0, 0)])  # Preserve baseline ID matching

    def test_compare_uses_same_cache_but_isolated_tracker_instances(self):
        report, frame_rows, previews = compare_policies(
            records(3), Path("."), tracker_factory=FakeTracker,
            image_loader=fake_images, gt_loader=fake_gt, updater=fake_update)
        self.assertEqual(set(report["policies"]), set(POLICIES))
        base = report["policies"]["confirmed_only"]["overall"]
        visible = report["policies"]["provisional_visibility"]["overall"]
        early = report["policies"]["early_promotion"]["overall"]
        self.assertEqual(base["gt_boxes"], 12)
        self.assertEqual(visible["confirmed_matches"], base["matched_boxes"])
        self.assertGreater(visible["matched_boxes"], base["matched_boxes"])
        self.assertGreater(early["matched_boxes"], base["matched_boxes"])
        self.assertEqual(len(frame_rows), 3 * 4 * 3)
        self.assertEqual(len(previews), len(frame_rows))
        for p in previews:
            if p["policy"] == "provisional_visibility":
                for obs in p["observations"]:
                    if obs["status"] == "provisional":
                        self.assertIsNone(obs["track_id"])

    def test_standard_pass_does_not_promote(self):
        row, _ = run_pass(records(2), Path("."), False,
                          tracker_factory=FakeTracker, image_loader=fake_images,
                          gt_loader=fake_gt, updater=fake_update)
        base = [r for r in row if r["policy"] == "confirmed_only"]
        self.assertEqual(sum(r["matched_boxes"] for r in base), 0)

    def test_empty_summary_rejected(self):
        with self.assertRaises(ValueError):
            summarise_policy([])

    def test_cache_order_validated(self):
        bad = list(reversed(records(3)))
        with self.assertRaises(ValueError):
            compare_policies(bad, Path("."), tracker_factory=FakeTracker,
                             image_loader=fake_images, gt_loader=fake_gt, updater=fake_update)


if __name__ == "__main__":
    unittest.main()
