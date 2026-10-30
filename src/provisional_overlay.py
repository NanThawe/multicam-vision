"""Read-only ByteTrack provisional observations and four-camera preview drawing.

This module never promotes tracks, mutates tracker state or assigns public IDs.
Call collect_provisional AFTER MultiCameraTracker.update, but pass its output
ONLY to render_grid. ReID and event code must receive confirmed tracks only.
"""

import math

import cv2
import numpy as np

CAMERAS = ("C1", "C2", "C3", "C4")
TILE_W, TILE_H = 640, 360
CONFIRMED_BGR = (70, 220, 70)
PROVISIONAL_BGR = (0, 185, 255)


def intersection_over_union(a, b):
    """IoU of two xyxy boxes; used only for duplicate overlay suppression."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def collect_provisional(trackers, confirmed, *, duplicate_iou=0.7):
    """Extract only same-update unconfirmed boxes from four existing trackers.

    `trackers` maps camera to the underlying Ultralytics BYTETracker. `confirmed`
    is the UNMODIFIED MultiCameraTracker.update output. Provisional boxes have
    null public track IDs and no global IDs; they cannot be used by ReID/events.
    """
    if not (0.0 <= duplicate_iou <= 1.0):
        raise ValueError("duplicate_iou must be in [0, 1]")
    if set(confirmed) != set(trackers):
        raise ValueError("Camera keys must match existing tracker states")
    result = {}
    for camera, tracker in trackers.items():
        if not hasattr(tracker, "tracked_stracks") or not hasattr(tracker, "frame_id"):
            raise RuntimeError("Installed ByteTrack does not expose track lifecycle state")
        candidates = []
        for track in tracker.tracked_stracks:
            if bool(getattr(track, "is_activated", False)):
                continue
            if getattr(track, "frame_id", None) != tracker.frame_id:
                continue  # Do not show stale positions as new observations.
            box = [float(v) for v in track.xyxy]
            if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                continue
            score = float(track.score)
            if not math.isfinite(score):
                continue
            if any(intersection_over_union(box, p["xyxy"]) >= duplicate_iou
                   for p in confirmed[camera]):
                continue  # Presentation-only, does NOT affect tracking state.
            candidates.append({"xyxy": box, "confidence": score,
                               "track_id": None, "status": "provisional"})
        result[camera] = candidates
    return result


def _draw_box(tile, person, original_shape, *, provisional):
    h, w = original_shape[:2]
    x1, y1, x2, y2 = [float(x) for x in person["xyxy"]]
    a = (max(0, min(TILE_W - 1, round(x1 * TILE_W / w))),
         max(0, min(TILE_H - 1, round(y1 * TILE_H / h))))
    b = (max(0, min(TILE_W - 1, round(x2 * TILE_W / w))),
         max(0, min(TILE_H - 1, round(y2 * TILE_H / h))))
    colour = PROVISIONAL_BGR if provisional else CONFIRMED_BGR
    cv2.rectangle(tile, a, b, colour, 1 if provisional else 2)
    if provisional:
        label = "P"  # No provisional local/global ID on the public preview.
    else:
        label = f"L{int(person['track_id'])}"
        if person.get("global_id") is not None:
            label += f" G{int(person['global_id'])}"
    cv2.putText(tile, label, (a[0], max(17, a[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)


def render_grid(frames, confirmed, provisional, *, cameras=CAMERAS):
    """Render four camera tiles; neither frames nor people dictionaries mutate."""
    canvas = np.zeros((2 * TILE_H, 2 * TILE_W, 3), dtype=np.uint8)
    for index, camera in enumerate(cameras):
        frame = frames[camera]
        tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        for person in confirmed[camera]:
            if person.get("track_id") is None:
                raise ValueError("Only confirmed tracks may enter the confirmed preview layer")
            _draw_box(tile, person, frame.shape, provisional=False)
        for person in provisional[camera]:
            if person.get("track_id") is not None or person.get("status") != "provisional":
                raise ValueError("Provisional observations must have null IDs and provisional status")
            _draw_box(tile, person, frame.shape, provisional=True)
        cv2.rectangle(tile, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
        cv2.putText(tile, f"{camera}  |  tracked {len(confirmed[camera])}  |  P {len(provisional[camera])}",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 2, cv2.LINE_AA)
        row, col = divmod(index, 2)
        canvas[row*TILE_H:(row+1)*TILE_H, col*TILE_W:(col+1)*TILE_W] = tile
    return canvas
