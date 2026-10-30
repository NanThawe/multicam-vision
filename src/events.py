"""Generate zone and line-crossing events from saved multi-camera tracks.

Run from the project root: python -m src.events
No model inference is run here. Camera zones are illustrative, not calibrated.
"""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "reid" / "global_tracks.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "events"

# WILDTRACK source videos used in this project: 1920x1080, ~59.94 FPS.
# FPS is used only for an approximate video-relative timestamp.
WIDTH, HEIGHT = 1920, 1080
SOURCE_FPS = 59.94
MAX_GAP_FRAMES = 120  # Reset movement history after ~2 s without observation.
LINE_MARGIN = 0.008  # Normalised distance tolerance to reduce boundary jitter.

# Illustrative regions in normalised image coordinates (0..1), NOT physical
# locations or validated WILDTRACK scene boundaries. Adjust for each camera.
# zone = (left, top, right, bottom); line = (start_xy, end_xy).
RULES = {
    camera: {
        "zone": (0.30, 0.35, 0.70, 0.85),
        "line": ((0.50, 0.20), (0.50, 0.90)),
    }
    for camera in ("C1", "C2", "C3", "C4")
}


def footpoint(person, width=WIDTH, height=HEIGHT):
    """Bottom-centre of a detection box, normalised to [0, 1]."""
    x1, _, x2, y2 = person["xyxy"]
    return ((x1 + x2) / (2 * width), y2 / height)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def signed_side(point, line):
    """Signed distance to oriented line in normalised image coordinates."""
    a, b = line
    direction = (b[0] - a[0], b[1] - a[1])
    length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    if length == 0:
        raise ValueError("Line endpoints must differ")
    return cross(direction, (point[0] - a[0], point[1] - a[1])) / length


def segment_crosses(p, q, line):
    """True only when movement p->q intersects the finite line segment."""
    a, b = line
    r = (q[0] - p[0], q[1] - p[1])
    s = (b[0] - a[0], b[1] - a[1])
    denominator = cross(r, s)
    if abs(denominator) < 1e-12:
        return False
    offset = (a[0] - p[0], a[1] - p[1])
    t = cross(offset, s) / denominator
    u = cross(offset, r) / denominator
    return 0 <= t <= 1 and 0 <= u <= 1


class EventEngine:
    """Maintain simple per-camera, per-local-track movement state."""

    def __init__(self, rules=RULES, width=WIDTH, height=HEIGHT,
                 source_fps=SOURCE_FPS, max_gap_frames=MAX_GAP_FRAMES):
        self.rules = rules
        self.width, self.height = width, height
        self.source_fps = source_fps
        self.max_gap_frames = max_gap_frames
        self.states = {}  # (camera, local track ID) -> last movement state

    def update(self, frame_id, camera, people):
        """Process one camera record, returning zero or more event dictionaries."""
        if camera not in self.rules:
            return []

        # Retire stale histories so recycled track IDs do not generate fake events.
        for key, state in list(self.states.items()):
            if key[0] == camera and frame_id - state["frame_id"] > self.max_gap_frames:
                del self.states[key]

        zone = self.rules[camera]["zone"]
        line = self.rules[camera]["line"]
        events = []

        for person in people:
            key = (camera, int(person["track_id"]))
            point = footpoint(person, self.width, self.height)
            inside = zone[0] <= point[0] <= zone[2] and zone[1] <= point[1] <= zone[3]
            distance = signed_side(point, line)
            side = 1 if distance > LINE_MARGIN else -1 if distance < -LINE_MARGIN else 0
            previous = self.states.get(key)

            def emit(event_type, rule_id, direction=None):
                event = {
                    "event_id": f"{camera}:{key[1]}:{frame_id}:{event_type}",
                    "event_type": event_type,
                    "rule_id": rule_id,
                    "camera": camera,
                    "track_id": key[1],
                    "global_id": person.get("global_id"),  # Unverified ReID hypothesis.
                    "frame_id": frame_id,
                    "elapsed_s": round(frame_id / self.source_fps, 3),
                    "footpoint": [round(point[0], 4), round(point[1], 4)],
                }
                if direction:
                    event["direction"] = direction
                events.append(event)

            if previous is None:
                # Initial observation is a baseline, not an entry/crossing.
                self.states[key] = {
                    "frame_id": frame_id, "inside": inside,
                    "side": side, "stable_point": point if side else None,
                }
                continue
            if frame_id <= previous["frame_id"]:
                raise ValueError(f"Frames must increase for {camera} track {key[1]}")

            if inside != previous["inside"]:
                emit("zone_entry" if inside else "zone_exit", "zone_1")

            if side and previous["side"] and side != previous["side"]:
                if segment_crosses(previous["stable_point"], point, line):
                    direction = ("positive_to_negative" if previous["side"] == 1
                                 else "negative_to_positive")
                    emit("line_crossing", "line_1", direction)

            previous["frame_id"] = frame_id
            previous["inside"] = inside
            if side:
                previous["side"] = side
                previous["stable_point"] = point

        return events


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}\nRun 'python -m src.reid' first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "events.jsonl"
    engine = EventEngine()
    counts = Counter()
    with INPUT_PATH.open("r", encoding="utf-8") as source, \
            output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for event in engine.update(record["frame_id"], record["camera"],
                                       record["tracks"]):
                target.write(json.dumps(event) + "\n")
                counts[event["event_type"]] += 1

    print(f"Saved {output_path}")
    print(f"Events: {sum(counts.values())} | " +
          ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    if not counts:
        print("No crossings detected. Check rules or analyse a longer clip.")


if __name__ == "__main__":
    main()
