"""Synthetic scheduling tests for ReID embedding reuse (no BoxMOT weights needed)."""

import unittest

import numpy as np

from src.reid import CrossCameraMatcher
from src.benchmark import run_benchmark


class StubEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, camera, frame_id, image, people):
        self.calls.append((camera, frame_id, [p["track_id"] for p in people]))
        # Deliberately identical for same person across cameras.
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in people]


def person(track_id=1, width=32, height=64):
    return {"track_id": track_id, "xyxy": [0, 0, width, height],
            "confidence": 0.9}


class ReIDScheduleTests(unittest.TestCase):
    def test_bootstrap_and_refresh_every_five(self):
        encoder = StubEncoder()
        matcher = CrossCameraMatcher(encoder, refresh_every=5)
        ids = []
        for i in range(10):
            people, _ = matcher.update(i * 12, {"C1": None}, {"C1": [person()]})
            ids.append(people["C1"][0]["global_id"])

        self.assertEqual(ids[:2], [None, None])
        self.assertEqual(ids[2:], [1] * 8)
        self.assertEqual([frame for _, frame, _ in encoder.calls], [0, 12, 24, 84])
        self.assertEqual(matcher.stats, {
            "eligible_crops": 10, "encoded_crops": 4,
            "reused_crops": 6, "encoder_calls": 4,
        })

    def test_new_track_bootstraps_while_old_track_is_cached(self):
        encoder = StubEncoder()
        matcher = CrossCameraMatcher(encoder, refresh_every=5)
        for i in range(5):
            tracks = {"C1": [person(1)]}
            if i >= 3:
                tracks["C1"].append(person(2))
            matcher.update(i * 12, {"C1": None}, tracks)
        self.assertEqual(encoder.calls[3][2], [2])
        self.assertEqual(encoder.calls[4][2], [2])
        self.assertEqual(matcher.local[("C1", 2)]["count"], 2)

    def test_original_frequency_one_encodes_every_frame(self):
        encoder = StubEncoder()
        matcher = CrossCameraMatcher(encoder, refresh_every=1)
        for i in range(10):
            matcher.update(i, {"C1": None}, {"C1": [person()]})
        self.assertEqual(matcher.stats["encoded_crops"], 10)
        self.assertEqual(matcher.stats["reused_crops"], 0)

    def test_same_person_across_cameras_keeps_distinct_local_keys(self):
        encoder = StubEncoder()
        matcher = CrossCameraMatcher(encoder, refresh_every=5)
        for i in range(3):
            tracks, _ = matcher.update(i, {"C1": None, "C2": None},
                                       {"C1": [person()], "C2": [person()]})
        self.assertEqual(tracks["C1"][0]["global_id"],
                         tracks["C2"][0]["global_id"])
        self.assertEqual(len(matcher.local), 2)

    def test_invalid_bbox_and_invalid_refresh(self):
        with self.assertRaises(ValueError):
            CrossCameraMatcher(StubEncoder(), refresh_every=0)
        encoder = StubEncoder()
        matcher = CrossCameraMatcher(encoder)
        output, decisions = matcher.update(0, {"C1": None},
                                           {"C1": [person(width=4)]})
        self.assertIsNone(output["C1"][0]["global_id"])
        self.assertEqual(decisions, [])
        self.assertEqual(encoder.calls, [])

    def test_benchmark_reports_embedding_reuse_only_for_measured_batches(self):
        class Detector:
            def detect(self, frames):
                return {"C1": []}

        class Tracker:
            def update(self, frames, detections):
                return {"C1": [person()]}

        class Events:
            def update(self, frame_id, camera, tracks):
                return []

        matcher = CrossCameraMatcher(StubEncoder(), refresh_every=5)
        frames = iter((i * 12, {"C1": None}) for i in range(10))
        report, _ = run_benchmark(frames, Detector(), Tracker(), Events(),
                                  matcher=matcher, warmup=3, batches=7)
        self.assertEqual(report["reid_workload"]["eligible_crops"], 7)
        self.assertEqual(report["reid_workload"]["encoded_crops"], 1)
        self.assertEqual(report["reid_workload"]["reused_crops"], 6)
        self.assertEqual(report["reid_workload"]["encoder_calls"], 1)


if __name__ == "__main__":
    unittest.main()
