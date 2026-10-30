"""Overlay saved multi-camera tracks and events on the original videos.

Run from the project root: python -m src.event_preview
This is a visual diagnostic: no detector, tracker, or ReID model is loaded.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .events import RULES
from .framesource import VIDEOS, iter_frames

ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = ROOT / "outputs" / "reid" / "global_tracks.jsonl"
EVENTS_PATH = ROOT / "outputs" / "events" / "events.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "events" / "preview.mp4"
SAMPLE_FPS = 5  # Must match the FPS used by src.reid.
TILE_W, TILE_H = 640, 360
CAMERAS = ("C1", "C2", "C3", "C4")
EVENT_COLORS = {"zone_entry": (70, 220, 70),
                "zone_exit": (0, 170, 255),
                "line_crossing": (220, 80, 255)}  # OpenCV BGR


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc


def load_records(tracks_path=TRACKS_PATH, events_path=EVENTS_PATH):
    """Index tracking observations and event markers by original frame number."""
    tracks = defaultdict(dict)
    for row in read_jsonl(tracks_path):
        frame_id, camera = int(row["frame_id"]), row["camera"]
        if camera in tracks[frame_id]:
            raise ValueError(f"Duplicate track record at frame {frame_id}, {camera}")
        tracks[frame_id][camera] = row["tracks"]
    if not tracks:
        raise ValueError("Tracking file contains no frame records")

    events = defaultdict(lambda: defaultdict(list))
    for event in read_jsonl(events_path):
        frame_id, camera = int(event["frame_id"]), event["camera"]
        people = tracks.get(frame_id, {}).get(camera)
        if people is None or not any(
            int(person["track_id"]) == int(event["track_id"]) for person in people
        ):
            raise ValueError(
                f"Event {event.get('event_id')} has no matching tracked person. "
                "Regenerate events from the current global_tracks.jsonl."
            )
        events[frame_id][camera].append(event)
    return tracks, events


def draw_tile(frame, camera, people, events, notices=(), event_total=0, rules=RULES):
    """Draw zone, counting line, tracks and event markers on a 640x360 tile."""
    original_h, original_w = frame.shape[:2]
    tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

    def xy(point):
        return (round(point[0] * TILE_W), round(point[1] * TILE_H))

    if camera in rules:
        zone = rules[camera]["zone"]
        upper_left, lower_right = xy(zone[:2]), xy(zone[2:])
        overlay = tile.copy()
        cv2.rectangle(overlay, upper_left, lower_right, (30, 180, 30), -1)
        cv2.addWeighted(overlay, 0.10, tile, 0.90, 0, tile)
        cv2.rectangle(tile, upper_left, lower_right, (40, 220, 40), 2)
        start, end = (xy(point) for point in rules[camera]["line"])
        cv2.line(tile, start, end, (255, 220, 0), 2)
        cv2.circle(tile, start, 4, (255, 220, 0), -1)
        cv2.circle(tile, end, 4, (255, 220, 0), -1)

    for person in people:
        x1, y1, x2, y2 = person["xyxy"]
        left_top = (round(x1 * TILE_W / original_w), round(y1 * TILE_H / original_h))
        right_bottom = (round(x2 * TILE_W / original_w), round(y2 * TILE_H / original_h))
        colour = (100, 255, 100)
        cv2.rectangle(tile, left_top, right_bottom, colour, 1)
        label = f"L{person['track_id']}"
        if person.get("global_id") is not None:
            label += f" G{person['global_id']}"
        cv2.putText(tile, label, (left_top[0], max(15, left_top[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        cv2.circle(tile, ((left_top[0] + right_bottom[0]) // 2, right_bottom[1]),
                   3, colour, -1)

    for event in events:
        p = xy(event["footpoint"])
        colour = EVENT_COLORS.get(event["event_type"], (255, 255, 255))
        cv2.circle(tile, p, 11, colour, 2)
        cv2.circle(tile, p, 4, colour, -1)

    cv2.rectangle(tile, (0, 0), (TILE_W, 32), (0, 0, 0), -1)
    cv2.putText(tile, f"{camera}  |  events: {event_total}", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for index, message in enumerate(notices[:2]):
        y = TILE_H - 12 - (len(notices[:2]) - 1 - index) * 24
        cv2.rectangle(tile, (0, y - 19), (TILE_W, y + 5), (0, 0, 0), -1)
        cv2.putText(tile, message, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.51, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_grid(frames, tracks, events, notices, totals):
    canvas = np.zeros((2 * TILE_H, 2 * TILE_W, 3), dtype=np.uint8)
    for index, camera in enumerate(CAMERAS):
        tile = draw_tile(frames[camera], camera, tracks[camera],
                         events.get(camera, []), notices.get(camera, []), totals[camera])
        row, col = divmod(index, 2)
        canvas[row * TILE_H:(row + 1) * TILE_H,
               col * TILE_W:(col + 1) * TILE_W] = tile
    return canvas


def render_preview(tracks_path=TRACKS_PATH, events_path=EVENTS_PATH,
                   output_path=OUTPUT_PATH, videos=VIDEOS, sample_fps=SAMPLE_FPS):
    """Render only original frames present in the saved ReID track log."""
    tracks, events = load_records(tracks_path, events_path)
    missing_cameras = {
        (frame_id, camera) for frame_id, rows in tracks.items()
        for camera in CAMERAS if camera not in rows
    }
    if missing_cameras:
        raise ValueError(f"Missing camera records: {sorted(missing_cameras)[:5]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             sample_fps, (2 * TILE_W, 2 * TILE_H))
    if not writer.isOpened():
        raise OSError(f"Cannot create output video: {output_path}")

    totals = Counter()
    notices = {}  # camera -> (last event labels, expiry preview-frame number)
    seen = set()
    last_frame = max(tracks)
    reader = iter_frames(videos=videos, sample_fps=sample_fps)
    try:
        for frame_id, frames in reader:
            if frame_id > last_frame:
                break
            if frame_id not in tracks:
                continue

            current = events.get(frame_id, {})
            for camera, camera_events in current.items():
                people = {int(p["track_id"]): p for p in tracks[frame_id][camera]}
                h, w = frames[camera].shape[:2]
                for event in camera_events:
                    x1, _, x2, y2 = people[int(event["track_id"])]["xyxy"]
                    expected = ((x1 + x2) / (2 * w), y2 / h)
                    if any(abs(a - b) > 0.01 for a, b in zip(expected, event["footpoint"])):
                        raise ValueError(
                            f"Event footpoint differs from track: {event['event_id']}. "
                            "Check whether both JSONL files came from the same run."
                        )
            for camera, camera_events in current.items():
                totals[camera] += len(camera_events)
                if camera_events:
                    labels = [f"{event['event_type']}  L{event['track_id']}"
                              for event in camera_events]
                    notices[camera] = (labels, len(seen) + 3)

            active = {camera: labels for camera, (labels, expires) in notices.items()
                      if len(seen) < expires}
            writer.write(make_grid(frames, tracks[frame_id], current, active, totals))
            seen.add(frame_id)
    finally:
        reader.close()
        writer.release()

    if seen != set(tracks):
        output_path.unlink(missing_ok=True)
        missing = sorted(set(tracks) - seen)
        raise ValueError(
            f"Video frames and tracking log do not align. Missing frames: {missing[:8]}. "
            "Check SAMPLE_FPS and ensure the same recordings were used for ReID."
        )
    print(f"Saved {output_path} ({len(seen)} four-camera frames)")
    print("Displayed events: " + ", ".join(f"{c}={totals[c]}" for c in CAMERAS))
    return output_path


if __name__ == "__main__":
    render_preview()
