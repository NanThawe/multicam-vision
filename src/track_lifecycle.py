"""Step 14.4: inspect ByteTrack lifecycle using cached detections.

Run from project root:
    python -m src.track_lifecycle --frames 100
    python -m src.track_lifecycle --frames 100 --video-batches 100

Requires Step 14.3's detections.jsonl. No changes to the live pipeline.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .track_ablation import CAMERAS, DATA, OUT as ABLATION_OUT, SETTINGS, load_images, tracker_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "evaluation" / "track_lifecycle"


def confidence_counts(detections, high=0.25, low=0.10):
    """Counts by ByteTrack confidence band, NOT exact association outcomes."""
    scores = [float(d["confidence"]) for d in detections]
    return {
        "detections": len(scores),
        "high_score_detections": sum(x >= high for x in scores),
        "low_score_detections": sum(low < x < high for x in scores),
        "at_or_below_low_threshold": sum(x <= low for x in scores),
        "below_new_track_threshold": sum(x < 0.35 for x in scores),
    }


def state_ids(tracker):
    """Snapshot public ByteTrack state, including unconfirmed tracks."""
    tracked = list(getattr(tracker, "tracked_stracks", []))
    return {
        "active": {int(t.track_id) for t in tracked if getattr(t, "is_activated", False)},
        "unconfirmed": {int(t.track_id) for t in tracked if not getattr(t, "is_activated", False)},
        "lost": {int(t.track_id) for t in getattr(tracker, "lost_stracks", [])},
        "removed": {int(t.track_id) for t in getattr(tracker, "removed_stracks", [])},
    }


def transition_counts(before, after, already_seen, already_removed):
    """Infer lifecycle transitions from snapshots. No claim about rejection causes."""
    current = after["active"] | after["unconfirmed"] | after["lost"]
    return {
        "new_track_ids": len(current - already_seen),
        "became_lost": len(after["lost"] - before["lost"]),
        "reactivated": len(before["lost"] & after["active"]),
        "newly_removed": len(after["removed"] - already_removed),
        "active_tracks": len(after["active"]),
        "unconfirmed_tracks": len(after["unconfirmed"]),
        "lost_tracks": len(after["lost"]),
    }


def output_indices(rows, detection_count):
    """Read ByteTrack's optional original detection index in column 8.

    Return None if the local Ultralytics version does not expose indices.
    """
    if len(rows) == 0:
        return set()  # no outputs
    if any(len(row) < 8 for row in rows):
        return None
    indices = [int(row[7]) for row in rows]
    if any(i < 0 or i >= detection_count for i in indices):
        return None
    return set(indices)


class LifecycleMeter:
    """One camera only: update an underlying BYTETracker and count its state."""

    def __init__(self, tracker, high=0.25, low=0.10):
        self.tracker = tracker
        self.high, self.low = high, low
        self.before = state_ids(tracker)
        self.ever_seen = set()
        self.ever_removed = set()

    def update(self, frame_id, image, detections):
        # Matches src/tracker.py exactly: Boxes with xyxy, confidence, class.
        import numpy as np
        from ultralytics.engine.results import Boxes

        arr = np.asarray([d["xyxy"] + [d["confidence"], d["class_id"]]
                          for d in detections], dtype=np.float32).reshape(-1, 6)
        boxes = Boxes(arr, orig_shape=image.shape[:2])
        rows = self.tracker.update(boxes, img=image)
        return self.record(frame_id, detections, rows)

    def record(self, frame_id, detections, rows):
        """Pure accounting after update: also usable with controlled mock states."""
        after = state_ids(self.tracker)
        stats = transition_counts(self.before, after, self.ever_seen, self.ever_removed)
        conf = confidence_counts(detections, self.high, self.low)
        out = list(rows)
        inds = output_indices(out, len(detections))
        stats.update(conf)
        stats.update({"frame_id": int(frame_id), "output_tracks": len(out),
                      "detection_indices_available": inds is not None})
        if inds is not None:
            matched_scores = [float(detections[i]["confidence"]) for i in inds]
            high_count = sum(x >= self.high for x in matched_scores)
            low_count = sum(self.low < x < self.high for x in matched_scores)
            stats.update({
                "returned_high_detections": high_count,
                "returned_low_detections": low_count,
                "high_not_in_output": conf["high_score_detections"] - high_count,
                "low_not_in_output": conf["low_score_detections"] - low_count,
            })
        # This is an observation, NOT a record of ByteTrack's internal reject reasons.
        self.ever_seen |= after["active"] | after["unconfirmed"] | after["lost"]
        self.ever_removed |= after["removed"]
        self.before = after
        tracks = [{"track_id": int(row[4]), "xyxy": [float(v) for v in row[:4]]}
                  for row in out]
        return stats, tracks


def new_meters(cameras=CAMERAS, factory=None):
    """Use independent trackers and IDs per camera. Factory injectable for tests."""
    if factory is None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        factory = lambda: BYTETracker(tracker_config(SETTINGS["baseline"]))
    return {cam: LifecycleMeter(factory()) for cam in cameras}


def evaluate_batches(batches, meters, ground_truth_fn=None):
    """batches yields (frame_id, images, camera detections)."""
    from .evaluate import match_frame
    rows = []
    for fid, images, detections in batches:
        gt_by_camera = ground_truth_fn(fid, images) if ground_truth_fn else None
        for camera, meter in meters.items():
            stats, tracks = meter.update(fid, images[camera], detections[camera])
            stats["camera"] = camera
            if gt_by_camera is not None:
                truth = gt_by_camera[camera]
                matched = len(match_frame(truth, tracks, threshold=0.5))
                stats.update({"gt_boxes": len(truth), "matched_boxes": matched})
            rows.append(stats)
    return rows


def summarise(rows, label, has_gt):
    cumulative_keys = (
        "detections", "high_score_detections", "low_score_detections",
        "at_or_below_low_threshold", "below_new_track_threshold",
        "output_tracks", "new_track_ids", "became_lost", "reactivated",
        "newly_removed", "returned_high_detections", "returned_low_detections",
        "high_not_in_output", "low_not_in_output", "matched_boxes", "gt_boxes",
    )
    summary = {}
    for cam in CAMERAS:
        seq = [r for r in rows if r["camera"] == cam]
        totals = {k: sum(r.get(k, 0) for r in seq) for k in cumulative_keys}
        available = bool(seq) and all(r["detection_indices_available"] for r in seq)
        result = {"batches": len(seq), "detection_indices_available": available}
        result.update({k: totals[k] for k in cumulative_keys[:9]})
        result["newly_removed"] = totals["newly_removed"]
        result["max_unconfirmed_tracks"] = max((r["unconfirmed_tracks"] for r in seq), default=0)
        result["max_lost_tracks"] = max((r["lost_tracks"] for r in seq), default=0)
        result["output_per_detection"] = round(totals["output_tracks"] / totals["detections"], 4) if totals["detections"] else None
        if available:
            result.update({k: totals[k] for k in ("returned_high_detections", "returned_low_detections",
                                                    "high_not_in_output", "low_not_in_output")})
        if has_gt:
            result["gt_boxes"] = totals["gt_boxes"]
            result["matched_boxes"] = totals["matched_boxes"]
            result["matched_gt_recall"] = (round(totals["matched_boxes"] / totals["gt_boxes"], 4)
                                          if totals["gt_boxes"] else None)
        summary[cam] = result
    all_det = sum(v["detections"] for v in summary.values())
    all_tracks = sum(v["output_tracks"] for v in summary.values())
    aggregate = {"detections": all_det, "output_tracks": all_tracks,
                 "output_per_detection": round(all_tracks / all_det, 4) if all_det else None,
                 "new_track_ids": sum(v["new_track_ids"] for v in summary.values()),
                 "became_lost": sum(v["became_lost"] for v in summary.values()),
                 "reactivated": sum(v["reactivated"] for v in summary.values()),
                 "newly_removed": sum(v["newly_removed"] for v in summary.values())}
    if has_gt:
        n_gt = sum(v["gt_boxes"] for v in summary.values())
        n_match = sum(v["matched_boxes"] for v in summary.values())
        aggregate.update({"gt_boxes": n_gt, "matched_boxes": n_match,
                          "matched_gt_recall": round(n_match / n_gt, 4) if n_gt else None})
    return {"label": label, "aggregate": aggregate, "per_camera": summary}


def write_results(output, report, rows):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = sorted({k for r in rows for k in r})
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=100, help="Number of annotated frame sets")
    ap.add_argument("--cache", type=Path, default=None, help="Existing Step 14.3 detections.jsonl")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT)
    ap.add_argument("--video-batches", type=int, default=0,
                    help="Optional, independent MP4 test; NOT ground-truth comparable")
    ap.add_argument("--video-fps", type=float, default=5)
    args = ap.parse_args(argv)
    if args.frames < 2 or args.video_batches < 0 or args.video_fps <= 0:
        ap.error("--frames must be >=2; video batches >=0; video FPS >0")

    path = args.cache or (ABLATION_OUT / "frames_100" / "detections.jsonl")
    if not path.is_file():
        ap.error(f"Missing YOLO cache {path}. First run: python -m src.track_ablation --frames 100")
    all_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(all_records) < args.frames:
        ap.error(f"Cache has only {len(all_records)} frame sets; requested {args.frames}")
    from .track_ablation import validate_records
    records = all_records[:args.frames]
    frame_ids = [int(r["frame_id"]) for r in records]
    if sorted(frame_ids) != frame_ids or len(set(frame_ids)) != len(frame_ids):
        ap.error("Cache frames must be unique and sorted chronologically")
    validate_records(records, frame_ids)

    from .evaluate import parse_gt
    def gt_fn(fid, frames):
        first = frames[CAMERAS[0]]
        return parse_gt(args.data_dir, fid, width=first.shape[1], height=first.shape[0])

    def replay(record_set):
        for record in record_set:
            fid = int(record["frame_id"])
            yield fid, load_images(args.data_dir, fid), record["detections"]

    output = args.output_dir
    report = {"settings": {"tracker": "ByteTrack baseline", "detector": "reused cached YOLO boxes"},
              "annotated": {}, "limitations": [
                  "Lifecycle state differences describe observed transitions, not internal rejection causes.",
                  "High/low not-in-output counts require an exposed ByteTrack detection-index column.",
                  "Only annotated ground-truth subsets are scored; valid evaluation ROI remains unverified.",
                  "Subsampling the annotated 2 FPS images to 1 FPS changes the temporal spacing and number of evaluated observations.",
                  "Optional 5 FPS MP4 data differ in preprocessing, geometry and timestamps; do not compare accuracy directly.",
              ]}
    all_rows = []
    for label, subset in (("annotated_2fps", records), ("annotated_1fps", records[::2])):
        print(f"Analysing {label}: {len(subset)} four-camera batches", flush=True)
        meters = new_meters()
        rows = evaluate_batches(replay(subset), meters, ground_truth_fn=gt_fn)
        report["annotated"][label] = summarise(rows, label, has_gt=True)
        all_rows.extend([{**row, "experiment": label} for row in rows])

    if args.video_batches:
        from .framesource import iter_frames
        from .detector import PersonDetector
        import torch
        detector = PersonDetector(device="cpu")
        meters = new_meters()
        video = iter_frames(sample_fps=args.video_fps)
        def video_batches():
            try:
                for i, (fid, frames) in enumerate(video):
                    if i >= args.video_batches:
                        break
                    if i == 0:
                        torch.set_num_threads(4)  # predictor warm-up may reset threads
                    yield fid, frames, detector.detect(frames)
            finally:
                video.close()
        print(f"Analysing MP4 at ~{args.video_fps} FPS (no aligned ground truth)...", flush=True)
        rows = evaluate_batches(video_batches(), meters)
        report["video_5fps_unlabelled"] = summarise(rows, "video_5fps_unlabelled", has_gt=False)
        all_rows.extend([{**row, "experiment": "video_5fps_unlabelled"} for row in rows])

    write_results(output, report, all_rows)
    print(f"Saved {output / 'summary.json'} and {output / 'per_frame.csv'}")
    for name, info in report["annotated"].items():
        agg = info["aggregate"]
        print(f"{name}: {agg['detections']} detections, {agg['output_tracks']} output tracks, "
              f"matched-GT recall {agg['matched_gt_recall']:.1%}")


if __name__ == "__main__":
    main()
