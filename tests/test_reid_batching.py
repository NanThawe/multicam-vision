"""Synthetic equivalence tests for cross-camera ReID batching."""

import unittest

import numpy as np

from src.reid import CrossCameraMatcher


def person(track_id):
    return {"track_id": track_id, "xyxy": [10, 20, 60, 140], "confidence": 0.9}


class FakeEncoder:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls = 0

    def _vectors(self, people):
        return np.asarray([[1.0, 0.0] if p["track_id"] == 1 else [0.0, 1.0]
                           for p in people], dtype=np.float32)

    def encode(self, camera, frame_id, image, people):
        self.single_calls += 1
        return self._vectors(people)

    def encode_batch(self, requests):
        self.batch_calls += 1
        return [self._vectors(people) for _, _, _, people in requests]


class BatchReIDTests(unittest.TestCase):
    def test_same_id_assignments_with_and_without_batching(self):
        baseline_encoder, batched_encoder = FakeEncoder(), FakeEncoder()
        baseline = CrossCameraMatcher(baseline_encoder, refresh_every=5)
        batched = CrossCameraMatcher(batched_encoder, refresh_every=5,
                                     batch_across_cameras=True)
        for i in range(12):
            tracks = {"C1": [person(1), person(2)], "C2": [person(1)],
                      "C3": [person(2)], "C4": []}
            frames = {camera: None for camera in tracks}
            reference, ref_decisions = baseline.update(i * 12, frames, tracks)
            candidate, cand_decisions = batched.update(i * 12, frames, tracks)
            self.assertEqual(reference, candidate)
            self.assertEqual(ref_decisions, cand_decisions)
        self.assertEqual(baseline.stats["encoded_crops"], batched.stats["encoded_crops"])
        self.assertEqual(baseline.stats["reused_crops"], batched.stats["reused_crops"])
        self.assertGreater(baseline.stats["encoder_calls"], batched.stats["encoder_calls"])
        self.assertEqual(baseline_encoder.batch_calls, 0)
        self.assertEqual(batched_encoder.single_calls, 0)

    def test_empty_batch_avoids_encoder_call(self):
        encoder = FakeEncoder()
        matcher = CrossCameraMatcher(encoder, batch_across_cameras=True)
        tracks, decisions = matcher.update(0, {"C1": None, "C2": None},
                                           {"C1": [], "C2": []})
        self.assertEqual(tracks, {"C1": [], "C2": []})
        self.assertEqual(decisions, [])
        self.assertEqual(matcher.stats["encoder_calls"], 0)
        self.assertEqual(encoder.batch_calls, 0)

    def test_invalid_batch_shape_fails_loudly(self):
        class IncorrectEncoder(FakeEncoder):
            def encode_batch(self, requests):
                return []

        matcher = CrossCameraMatcher(IncorrectEncoder(), batch_across_cameras=True)
        with self.assertRaisesRegex(ValueError, "Wrong number"):
            matcher.update(0, {"C1": None}, {"C1": [person(1)]})

    def test_missing_batch_method_fails_loudly(self):
        class SingleOnly:
            def encode(self, camera, frame_id, image, people):
                return [np.array([1, 0], dtype=np.float32)]

        matcher = CrossCameraMatcher(SingleOnly(), batch_across_cameras=True)
        with self.assertRaisesRegex(TypeError, "encode_batch"):
            matcher.update(0, {"C1": None}, {"C1": [person(1)]})


if __name__ == "__main__":
    unittest.main()

class EncoderProfileTests(unittest.TestCase):
    def test_benchmark_excludes_warmup_from_encoder_profile(self):
        from src.benchmark import run_benchmark

        class ProfileEncoder(FakeEncoder):
            def __init__(self):
                super().__init__()
                self.timings_ms = {"input_prep": 0., "model_encode": 0.,
                                   "output_convert": 0.}

            def encode_batch(self, requests):
                self.timings_ms["input_prep"] += 2.0
                self.timings_ms["model_encode"] += 10.0
                self.timings_ms["output_convert"] += 1.0
                return super().encode_batch(requests)

        class Detector:
            def detect(self, frames):
                return {"C1": []}

        class Tracker:
            def update(self, frames, detections):
                return {"C1": [person(1)]}

        class Events:
            def update(self, frame_id, camera, tracks):
                return []

        encoder = ProfileEncoder()
        matcher = CrossCameraMatcher(encoder, refresh_every=1, batch_across_cameras=True)
        frames = iter((i, {"C1": None}) for i in range(5))
        report, _ = run_benchmark(frames, Detector(), Tracker(), Events(),
                                  matcher=matcher, warmup=2, batches=3)
        self.assertEqual(report["reid_encoder_profile"]["total_ms"], {
            "input_prep": 6.0,
            "model_encode": 30.0,
            "output_convert": 3.0,
        })
