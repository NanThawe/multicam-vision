"""Step 14.7 tests. No RTSP server, YOLO, or model weights needed."""

import copy
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.provisional_overlay import (collect_provisional, intersection_over_union,
                                     render_grid, CAMERAS)
from src.live_rtsp import summarize
import src.live_rtsp as live_rtsp


def candidate(box=(20, 30, 90, 170), *, current=5, activated=False, score=0.63):
    return SimpleNamespace(xyxy=list(box), frame_id=current,
                           is_activated=activated, score=score, track_id=891)


def make_state(camera_with_candidates="C1", values=None):
    trackers = {c: SimpleNamespace(frame_id=5, tracked_stracks=[]) for c in CAMERAS}
    trackers[camera_with_candidates].tracked_stracks = values or []
    confirmed = {c: [] for c in CAMERAS}
    return trackers, confirmed


class ProvisionalOverlayTests(unittest.TestCase):
    def test_same_update_only_no_public_identity(self):
        trackers, confirmed = make_state(values=[candidate()])
        result = collect_provisional(trackers, confirmed)
        self.assertEqual(result["C1"], [{"xyxy": [20., 30., 90., 170.],
                                         "confidence": 0.63,
                                         "track_id": None, "status": "provisional"}])
        self.assertEqual(result["C2"], [])
        self.assertNotIn("global_id", result["C1"][0])

    def test_activated_and_stale_tracks_never_returned(self):
        trackers, confirmed = make_state(values=[candidate(activated=True),
                                                  candidate(current=4)])
        self.assertFalse(collect_provisional(trackers, confirmed)["C1"])

    def test_no_tracker_mutation_and_no_confirmed_mutation(self):
        tracks = [candidate()]
        trackers, confirmed = make_state(values=tracks)
        confirmed["C1"] = [{"xyxy": [200, 200, 240, 260], "track_id": 41,
                            "confidence": 0.99, "global_id": 11}]
        saved = copy.deepcopy(confirmed)
        output = collect_provisional(trackers, confirmed)
        self.assertEqual(confirmed, saved)
        self.assertFalse(tracks[0].is_activated)
        self.assertEqual(tracks[0].track_id, 891)
        self.assertIsNone(output["C1"][0]["track_id"])

    def test_suppresses_overlap_with_confirmed_in_preview_only(self):
        trackers, confirmed = make_state(values=[candidate()])
        confirmed["C1"] = [{"xyxy": [20, 30, 90, 170], "track_id": 3}]
        self.assertEqual(collect_provisional(trackers, confirmed)["C1"], [])
        self.assertFalse(trackers["C1"].tracked_stracks[0].is_activated)

    def test_invalid_candidates_ignored(self):
        trackers, confirmed = make_state(values=[candidate(box=(20, 30, 5, 100)),
                                                  candidate(box=(20, 30, float('nan'), 100)),
                                                  candidate(score=float('inf'))])
        self.assertEqual(collect_provisional(trackers, confirmed)["C1"], [])

    def test_wrong_camera_keys_and_duplicate_threshold_rejected(self):
        trackers, confirmed = make_state()
        with self.assertRaises(ValueError):
            collect_provisional(trackers, {"C1": []})
        with self.assertRaises(ValueError):
            collect_provisional(trackers, confirmed, duplicate_iou=1.1)

    def test_missing_lifecycle_interface_rejected(self):
        trackers, confirmed = make_state()
        trackers["C1"] = SimpleNamespace(tracked_stracks=[])
        with self.assertRaisesRegex(RuntimeError, "lifecycle"):
            collect_provisional(trackers, confirmed)

    def test_iou_edge_cases(self):
        self.assertEqual(intersection_over_union([0, 0, 10, 10], [10, 0, 20, 10]), 0)
        self.assertAlmostEqual(intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(intersection_over_union([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_preview_is_four_tiles_with_distinct_box_colours(self):
        frames = {c: np.zeros((360, 640, 3), dtype=np.uint8) for c in CAMERAS}
        original = {c: frame.copy() for c, frame in frames.items()}
        confirmed = {c: [] for c in CAMERAS}
        provisional = {c: [] for c in CAMERAS}
        confirmed["C1"].append({"xyxy": [50, 90, 100, 150], "track_id": 8, "global_id": 3})
        provisional["C2"].append({"xyxy": [50, 90, 100, 150],
                                  "confidence": .5, "track_id": None,
                                  "status": "provisional"})
        frame = render_grid(frames, confirmed, provisional)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertTrue(np.array_equal(frame[90, 50], np.array([70, 220, 70])))
        self.assertTrue(np.array_equal(frame[90, 640 + 50], np.array([0, 185, 255])))
        for cam in CAMERAS:
            self.assertTrue(np.array_equal(frames[cam], original[cam]))

    def test_provisional_with_identity_rejected_on_draw(self):
        frames = {c: np.zeros((360, 640, 3), dtype=np.uint8) for c in CAMERAS}
        confirmed = {c: [] for c in CAMERAS}
        provisional = {c: [] for c in CAMERAS}
        provisional["C1"].append({"xyxy": [5, 50, 40, 150],
                                  "track_id": 1, "status": "provisional"})
        with self.assertRaisesRegex(ValueError, "null IDs"):
            render_grid(frames, confirmed, provisional)

    def test_mock_four_camera_runtime_does_not_route_provisional_to_analytics(self):
        """Exercise live integration, including encoding, without actual RTSP/models."""
        inspected = {"reid": 0, "events": 0}
        frames = {c: np.zeros((180, 320, 3), dtype=np.uint8) for c in CAMERAS}
        sensor = SimpleNamespace(frame_id=2, tracked_stracks=[candidate(current=2)])

        class FakeDetector:
            def __init__(self, **kwargs):
                pass
            def detect(self, images):
                return {c: [{"xyxy": [10, 30, 45, 130], "confidence": .8,
                             "class_id": 0}] for c in CAMERAS}

        class FakeTracker:
            def __init__(self):
                self.trackers = {c: SimpleNamespace(frame_id=2, tracked_stracks=[])
                                 for c in CAMERAS}
                self.trackers["C1"] = sensor
            def update(self, images, detections):
                return {c: [{"xyxy": [10, 30, 45, 130], "confidence": .8,
                             "track_id": 12}] for c in CAMERAS}

        class FakeEncoder:
            def __init__(self, **kwargs):
                pass

        class FakeMatcher:
            def __init__(self, *args, **kwargs):
                pass
            def update(self, fid, images, tracks):
                inspected["reid"] += 1
                assert all(set(p) == {"xyxy", "confidence", "track_id"}
                           for c in CAMERAS for p in tracks[c])
                return ({c: [{**p, "global_id": 32} for p in tracks[c]]
                         for c in CAMERAS}, [])

        class FakeEvents:
            def __init__(self, **kwargs):
                pass
            def update(self, fid, camera, tracks):
                inspected["events"] += 1
                assert all(p["track_id"] == 12 for p in tracks)
                return []

        class FakeSlot:
            def counters(self):
                return (10, 0)

        class FakeReader:
            def __init__(self, camera, url):
                self.camera, self.url, self.slot = camera, url, FakeSlot()
            def start(self, **kwargs):
                pass
            def stop(self):
                pass

        def fake_batch(readers, **kwargs):
            return {c: SimpleNamespace(image=frames[c], received_at=time.perf_counter())
                    for c in CAMERAS}

        module_mapping = {}
        for name, attributes in {
            "src.detector": {"PersonDetector": FakeDetector},
            "src.tracker": {"MultiCameraTracker": FakeTracker},
            "src.events": {"EventEngine": FakeEvents},
            "src.reid": {"AppearanceEncoder": FakeEncoder,
                         "CrossCameraMatcher": FakeMatcher},
            "ultralytics": {}, "ultralytics.utils": {},
            "ultralytics.utils.torch_utils": {"NUM_THREADS": 4},
        }.items():
            module = ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            module_mapping[name] = module
        module_mapping["ultralytics"].utils = module_mapping["ultralytics.utils"]
        module_mapping["ultralytics.utils"].torch_utils = module_mapping["ultralytics.utils.torch_utils"]

        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(sys.modules, module_mapping), \
             patch.object(live_rtsp, "RTSPReader", FakeReader), \
             patch.object(live_rtsp, "wait_for_batch", fake_batch), \
             patch("src.benchmark.configure_cpu_threads"), \
             patch("src.benchmark.check_intraop_threads"):
            args = SimpleNamespace(threads=4, mqtt=False, no_reid=False,
                                   rtsp_base="rtsp://127.0.0.1:8554",
                                   output_dir=Path(folder), provisional_preview=True,
                                   warmup_batches=0, seconds=60, max_batches=2,
                                   startup_timeout=1, timeout=1)
            report = live_rtsp.run_stream(args)
            self.assertTrue(report["complete"])
            self.assertEqual(inspected, {"reid": 2, "events": 8})
            self.assertEqual(report["provisional_preview"]["measured_provisional_observations"], 2)
            self.assertEqual(report["provisional_preview"]["measured_confirmed_observations"], 8)
            self.assertTrue((Path(folder) / "provisional_preview.mp4").is_file())
            video = cv2.VideoCapture(str(Path(folder) / "provisional_preview.mp4"))
            try:
                self.assertTrue(video.isOpened())
                self.assertTrue(video.read()[0])
            finally:
                video.release()
            with (Path(folder) / "provisional_counts.csv").open(newline="") as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 8)
            self.assertEqual(sum(int(r["provisional"]) for r in records), 2)
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text())
                             ["provisional_preview"]["measured_provisional_observations"], 2)

    def test_report_unchanged_without_overlay(self):
        cams = {c: (0, 0) for c in CAMERAS}
        samples = [{"frame_age_ms": {c: 4 for c in CAMERAS}, "processing_ms": 130}]
        report = summarize(samples, cams, {c: (1, 0) for c in CAMERAS}, .25,
                           event_count=0, warmup_batches=0)
        self.assertNotIn("provisional_preview", report)
        self.assertEqual(report["aggregate_processed_fps"], 16)


if __name__ == '__main__':
    unittest.main()
