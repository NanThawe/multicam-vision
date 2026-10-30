"""Track people independently in four cameras using ByteTrack."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from .detector import PersonDetector
from .framesource import iter_frames

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tracking"
CAMERAS = ("C1", "C2", "C3", "C4")
SAMPLE_FPS = 5
MAX_BATCHES = 30  # About six seconds of sampled video; set to None for all frames.


class MultiCameraTracker:
    def __init__(self, cameras=CAMERAS, config=None):
        # One independent tracker state and ID space per camera.
        if config is None:
            config = SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.35,
                track_buffer=10,  # 10 processed frames (about 2 seconds at 5 FPS).
                match_thresh=0.8,
                fuse_score=True,
            )
        self.trackers = {camera: BYTETracker(config) for camera in cameras}

    def update(self, frames, detections):
        """Return {camera: [{track_id, xyxy, confidence}, ...]}."""
        tracked = {}
        for camera, tracker in self.trackers.items():
            # The detector gives xyxy/conf/class; Boxes adapts it to ByteTrack.
            rows = [d["xyxy"] + [d["confidence"], d["class_id"]]
                    for d in detections[camera]]
            boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6),
                          orig_shape=frames[camera].shape[:2])
            tracks = tracker.update(boxes, img=frames[camera])
            tracked[camera] = [
                {
                    "track_id": int(row[4]),
                    "xyxy": [round(float(x), 1) for x in row[:4]],
                    "confidence": round(float(row[5]), 3),
                }
                for row in tracks
            ]
        return tracked


def draw_grid(frames, tracks, cameras=CAMERAS):
    """Draw four 640x360 camera tiles as one 1280x720 preview frame."""
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i, camera in enumerate(cameras):
        image = frames[camera]
        tile = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        sx, sy = 640 / image.shape[1], 360 / image.shape[0]
        for person in tracks[camera]:
            x1, y1, x2, y2 = person["xyxy"]
            top_left = (round(x1 * sx), round(y1 * sy))
            bottom_right = (round(x2 * sx), round(y2 * sy))
            cv2.rectangle(tile, top_left, bottom_right, (0, 255, 0), 2)
            label = f"L{person['track_id']}"
            if "global_id" in person:
                label += f" G{person['global_id']}" if person["global_id"] is not None else " G?"
            cv2.putText(tile, label,
                        (top_left[0], max(18, top_left[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.rectangle(tile, (0, 0), (90, 32), (0, 0, 0), -1)
        cv2.putText(tile, camera, (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y, x = divmod(i, 2)
        canvas[y * 360:(y + 1) * 360, x * 640:(x + 1) * 640] = tile
    return canvas


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / "preview.mp4"
    json_path = OUTPUT_DIR / "tracks.jsonl"

    detector = PersonDetector()
    tracker = MultiCameraTracker()
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             SAMPLE_FPS, (1280, 720))
    if not writer.isOpened():
        raise OSError(f"Cannot write preview video: {video_path}")

    reader = iter_frames(sample_fps=SAMPLE_FPS)
    batches = 0
    try:
        with json_path.open("w", encoding="utf-8") as out:
            for frame_id, frames in reader:
                detections = detector.detect(frames)
                tracks = tracker.update(frames, detections)
                writer.write(draw_grid(frames, tracks))

                for camera, people in tracks.items():
                    out.write(json.dumps({
                        "frame_id": frame_id,
                        "camera": camera,
                        "tracks": people,
                    }) + "\n")

                batches += 1
                print(f"Frame {frame_id}: " + ", ".join(
                    f"{cam}={len(people)} tracks" for cam, people in tracks.items()))
                if MAX_BATCHES is not None and batches >= MAX_BATCHES:
                    break
    finally:
        reader.close()
        writer.release()

    print(f"Processed {batches} four-camera batches")
    print(f"Saved {video_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
