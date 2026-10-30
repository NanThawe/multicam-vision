"""Smoke-test the event preview without running detection or ReID."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.event_preview import draw_tile, load_records, render_preview


class EventPreviewTest(unittest.TestCase):
    def test_marker_is_drawn_at_saved_event_footpoint(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        event = {"event_type": "zone_entry", "footpoint": [0.5, 0.5]}
        result = draw_tile(frame, "C1", [], [event])
        self.assertEqual(result.shape, (360, 640, 3))
        self.assertEqual(tuple(result[180, 320]), (70, 220, 70))

    def test_reject_event_without_corresponding_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            tracks_file, events_file = folder / "tracks.jsonl", folder / "events.jsonl"
            tracks_file.write_text(json.dumps({"frame_id": 0, "camera": "C1", "tracks": []}) + "\n")
            events_file.write_text(json.dumps({
                "frame_id": 0, "camera": "C1", "track_id": 9, "event_type": "zone_entry"
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "no matching tracked person"):
                load_records(tracks_file, events_file)

    def test_four_camera_video_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            videos = {}
            for camera in ("C1", "C2", "C3", "C4"):
                path = folder / f"{camera}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         10, (320, 180))
                self.assertTrue(writer.isOpened())
                for _ in range(8):
                    writer.write(np.full((180, 320, 3), 60, dtype=np.uint8))
                writer.release()
                videos[camera] = path

            tracks_file, events_file = folder / "tracks.jsonl", folder / "events.jsonl"
            with tracks_file.open("w") as out:
                for frame_id in (0, 2, 4, 6):
                    for camera in videos:
                        out.write(json.dumps({
                            "frame_id": frame_id, "camera": camera,
                            "tracks": [{"track_id": 1, "global_id": 7,
                                        "xyxy": [100, 40, 140, 150]}],
                        }) + "\n")
            event = {"event_id": "C1:1:2:zone_entry", "frame_id": 2, "camera": "C1",
                     "track_id": 1, "event_type": "zone_entry",
                     "footpoint": [0.375, 150 / 180]}
            events_file.write_text(json.dumps(event) + "\n")
            output = folder / "events" / "preview.mp4"
            self.assertEqual(render_preview(tracks_file, events_file, output,
                                            videos=videos, sample_fps=5), output)
            self.assertTrue(output.is_file())
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                count = 0
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    self.assertEqual(image.shape, (720, 1280, 3))
                    count += 1
                self.assertEqual(count, 4)
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
