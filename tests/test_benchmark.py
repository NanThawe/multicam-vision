"""Fast benchmarks tests; no videos, models or GPU needed."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import latency_stats, run_benchmark, save_results


class FakeDetector:
    def detect(self, frames):
        return {camera: [] for camera in frames}


class FakeTracker:
    def update(self, frames, detections):
        return {camera: [] for camera in frames}


class FakeEvents:
    def update(self, frame_id, camera, tracks):
        return []


class FakeMatcher:
    def update(self, frame_id, frames, tracks):
        return tracks, []


def frames(n):
    return iter((i * 12, {f"C{k}": None for k in range(1, 5)}) for i in range(n))


class BenchmarkTests(unittest.TestCase):
    def test_latency_percentiles(self):
        self.assertEqual(latency_stats([1, 2, 3, 4, 5]),
                         {"mean": 3, "p50": 3, "p95": 4.8})

    def test_four_streams_and_intentional_skips(self):
        report, samples = run_benchmark(
            frames(5), FakeDetector(), FakeTracker(), FakeEvents(),
            batches=3, warmup=2, rss_mb=lambda: 100.0,
        )
        self.assertEqual(report["measured_batches"], 3)
        self.assertEqual(report["warmup_batches"], 2)
        self.assertEqual(len(report["fps_per_camera"]), 4)
        self.assertEqual([s["frame_id"] for s in samples], [24, 36, 48])
        self.assertEqual(report["sampling"]["intentionally_skipped_source_frames_per_camera"], 22)
        self.assertIsNone(report["sampling"]["live_dropped_frames"])
        self.assertEqual(report["memory"]["max_sampled_process_rss_mb"], 100.0)
        self.assertAlmostEqual(report["aggregate_fps"],
                               4 * next(iter(report["fps_per_camera"].values())),
                               delta=0.005)

    def test_reid_path_and_short_video(self):
        report, samples = run_benchmark(
            frames(2), FakeDetector(), FakeTracker(), FakeEvents(),
            matcher=FakeMatcher(), batches=10, warmup=1,
        )
        self.assertEqual(report["measured_batches"], 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["reid_ms"] >= 0, True)

    def test_empty_and_invalid_batches(self):
        with self.assertRaises(ValueError):
            run_benchmark(frames(0), FakeDetector(), FakeTracker(), FakeEvents(),
                          warmup=0)
        with self.assertRaises(ValueError):
            run_benchmark(frames(2), FakeDetector(), FakeTracker(), FakeEvents(),
                          batches=0)

    def test_exports_json_and_csv(self):
        report, samples = run_benchmark(
            frames(3), FakeDetector(), FakeTracker(), FakeEvents(),
            warmup=0, batches=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            json_file, csv_file = save_results(report, samples, folder)
            self.assertEqual(json.loads(json_file.read_text())["measured_batches"], 2)
            with Path(csv_file).open(newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)


if __name__ == "__main__":
    unittest.main()
