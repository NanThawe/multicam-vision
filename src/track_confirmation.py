"""Step 14.5: ByteTrack confirmation and frame-history diagnostics.

Run from project root:
    python -m src.track_confirmation --frames 100

Replays Step 14.3's cached YOLO detections without modifying production code.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .evaluate import match_frame, parse_gt, iou
from .track_ablation import CAMERAS, DATA, OUT, SETTINGS, load_images, tracker_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "evaluation" / "track_confirmation"


def same_frame_unconfirmed(tracker):
    """Return only NEW, unconfirmed tracks whose boxes belong to this update."""
    result = []
    for track in getattr(tracker, "tracked_stracks", []):
        if getattr(track, "is_activated", False):
            continue
        if getattr(track, "frame_id", None) != getattr(tracker, "frame_id", None):
            continue
        box = [float(v) for v in track.xyxy]
        if len(box) != 4:
            continue
        result.append({"track_id": int(track.track_id), "xyxy": box,
                       "confidence": float(track.score), "idx": getattr(track, "idx", None)})
    return result


def confirmed_from_rows(rows):
    """Convert ByteTrack rows [x1,y1,x2,y2,id,score,cls,idx]."""
    return [{"track_id": int(row[4]),
             "xyxy": [float(x) for x in row[:4]],
             "confidence": float(row[5]),
             "idx": row[7] if len(row) > 7 else None}
            for row in rows]


def validated_detection_indices(tracks, detections):
    """Return known input indices, or None for unsupported/ambiguous tracker output.

    Check both index and score/box correspondence to catch versions that expose
    an index relative to a filtered detection subset instead of the full input.
    """
    indices = set()
    for track in tracks:
        index = track.get("idx")
        try:
            j = int(index)
        except (TypeError, ValueError, OverflowError):
            return None
        if index is None or float(index) != j or not 0 <= j < len(detections):
            return None
        detection = detections[j]
        if abs(float(track["confidence"]) - float(detection["confidence"])) > 0.02:
            return None
        if iou(track["xyxy"], detection["xyxy"]) < 0.3:
            return None
        if j in indices:
            return None
        indices.add(j)
    return indices


def frame_statistics(frame_id, camera, detections, gt, confirmed, pending, score=True):
    """Compare standard output with diagnostic addition of unconfirmed tracks."""
    if {t["track_id"] for t in confirmed} & {t["track_id"] for t in pending}:
        raise ValueError("A track cannot be confirmed and unconfirmed simultaneously")
    report = {
        "frame_id": int(frame_id), "camera": camera,
        "detections": len(detections), "confirmed": len(confirmed),
        "unconfirmed": len(pending),
    }
    active_idx = validated_detection_indices(confirmed, detections)
    pending_idx = validated_detection_indices(pending, detections)
    valid = active_idx is not None and pending_idx is not None and not (active_idx & pending_idx)
    report["detection_indices_verified"] = valid
    if valid:
        report["high_unaccounted"] = sum(
            float(d["confidence"]) >= 0.25 and n not in (active_idx | pending_idx)
            for n, d in enumerate(detections)
        )
        report["low_unaccounted"] = sum(
            0.10 < float(d["confidence"]) < 0.25 and n not in (active_idx | pending_idx)
            for n, d in enumerate(detections)
        )
    if score:
        report["gt"] = len(gt)
        report["matched_confirmed"] = len(match_frame(gt, confirmed, threshold=0.5))
        report["matched_with_unconfirmed"] = len(match_frame(gt, confirmed + pending, threshold=0.5))
    return report


def build_update(tracker, frame, detections):
    """Use the same Boxes format as the existing production tracker."""
    import numpy as np
    from ultralytics.engine.results import Boxes

    values = np.asarray([d["xyxy"] + [d["confidence"], d["class_id"]]
                         for d in detections], dtype=np.float32).reshape(-1, 6)
    return tracker.update(Boxes(values, orig_shape=frame.shape[:2]), img=frame)


def validate_cache(records):
    ids = [int(r["frame_id"]) for r in records]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Cached frames must be nonempty, unique and chronological")
    if any(set(r["detections"]) != set(CAMERAS) for r in records):
        raise ValueError("Each cached frame must contain detections from all four cameras")
    return ids


def run_history(records, data_dir, include_frame, score_frame, tracker_factory=None,
                image_loader=load_images, gt_loader=None, updater=build_update):
    """Run independent ByteTrack instances and return scored observations.

    include_frame(index): whether a frame advances the tracker.
    score_frame(index): whether GT matching is performed on that frame.
    """
    if tracker_factory is None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        tracker_factory = lambda: BYTETracker(tracker_config(SETTINGS["baseline"]))
    trackers = {cam: tracker_factory() for cam in CAMERAS}
    if gt_loader is None:
        gt_loader = lambda fid, img: parse_gt(data_dir, fid, width=img.shape[1], height=img.shape[0])
    rows = []
    for index, record in enumerate(records):
        if not include_frame(index):
            continue
        fid = int(record["frame_id"])
        images = image_loader(data_dir, fid)
        scoring = score_frame(index)
        truth = gt_loader(fid, images[CAMERAS[0]]) if scoring else None
        for cam in CAMERAS:
            dets = record["detections"][cam]
            output = updater(trackers[cam], images[cam], dets)
            confirmed = confirmed_from_rows(output)
            pending = same_frame_unconfirmed(trackers[cam])
            rows.append(frame_statistics(fid, cam, dets,
                                         truth[cam] if scoring else [],
                                         confirmed, pending, score=scoring))
    return rows


def summarise(rows):
    scored = [r for r in rows if "gt" in r]
    if not scored:
        raise ValueError("No scored frames found")

    def totals(items):
        t = Counter()
        for r in items:
            for key in ("gt", "matched_confirmed", "matched_with_unconfirmed",
                        "detections", "confirmed", "unconfirmed"):
                t[key] += r.get(key, 0)
        gt = t["gt"]
        return {
            **{k: t[k] for k in ("gt", "matched_confirmed", "matched_with_unconfirmed",
                                 "detections", "confirmed", "unconfirmed")},
            "confirmed_recall": round(t["matched_confirmed"] / gt, 4) if gt else None,
            "diagnostic_with_unconfirmed_recall": round(t["matched_with_unconfirmed"] / gt, 4) if gt else None,
            "extra_gt_matches_from_unconfirmed": t["matched_with_unconfirmed"] - t["matched_confirmed"],
        }

    per_cam = {c: totals([r for r in scored if r["camera"] == c]) for c in CAMERAS}
    return {
        "scored_frame_sets": len({r["frame_id"] for r in scored}),
        "aggregate": totals(scored),
        "per_camera": per_cam,
        "index_integrity": {
            "checked_observations": len(rows),
            "verified_observations": sum(bool(r["detection_indices_verified"]) for r in rows),
        },
        "diagnostic_unaccounted": {
            "high": sum(r.get("high_unaccounted", 0) for r in rows),
            "low": sum(r.get("low_unaccounted", 0) for r in rows),
            "available": all(r["detection_indices_verified"] for r in rows),
        },
    }


def compare(records, data_dir, **kwargs):
    validate_cache(records)
    # One dense pass: all frames advance ByteTrack. Score all frames and the
    # even-index subset. Sparse run: only even frames advance ByteTrack.
    dense = run_history(records, data_dir, lambda i: True, lambda i: True, **kwargs)
    sparse = run_history(records, data_dir, lambda i: i % 2 == 0,
                         lambda i: True, **kwargs)
    same_frames = {int(r["frame_id"]) for r in records[::2]}
    dense_subset = [r for r in dense if r["frame_id"] in same_frames]
    return {
        "basis": "Cached identical YOLO detections on WILDTRACK annotated PNGs",
        "full_2fps": summarise(dense),
        "same_frame_history_test": {
            "2fps_history": summarise(dense_subset),
            "1fps_history": summarise(sparse),
        },
        "limitations": [
            "Unconfirmed boxes are shown for diagnosis ONLY; the tracker state is unchanged.",
            "Adding unconfirmed boxes does not preserve confirmed identities and is not a production fix.",
            "Input detection indices may be unavailable or unverifiable in some Ultralytics versions.",
            "Unaccounted detections have no unique known internal rejection cause.",
            "Matched-box recall uses IoU 0.5; no verified WILDTRACK evaluation-region mask.",
            "This is not official IDF1/HOTA or 5 FPS RTSP accuracy.",
        ],
    }, dense, sparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    cache_path = args.cache or (OUT / f"frames_{args.frames}" / "detections.jsonl")
    if not cache_path.is_file():
        parser.error(f"Cache not found: {cache_path}. Run python -m src.track_ablation --frames {args.frames}")
    records = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != args.frames:
        parser.error(f"Cache contains {len(records)} frames, expected {args.frames}")
    report, dense, sparse = compare(records, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        records_out = ([dict(r, experiment="2fps_history") for r in dense]
                       + [dict(r, experiment="1fps_history") for r in sparse])
        cols = sorted(set().union(*(r.keys() for r in records_out)))
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        writer.writerows(records_out)
    whole = report["full_2fps"]["aggregate"]
    pair = report["same_frame_history_test"]
    print(f"2 FPS full: confirmed GT recall {whole['confirmed_recall']:.1%}; "
          f"including unconfirmed {whole['diagnostic_with_unconfirmed_recall']:.1%}")
    print("Identical scored frames, confirmed GT recall: "
          f"2 FPS history={pair['2fps_history']['aggregate']['confirmed_recall']:.1%}, "
          f"1 FPS history={pair['1fps_history']['aggregate']['confirmed_recall']:.1%}")
    print(f"Saved {args.output_dir / 'summary.json'} and {args.output_dir / 'per_frame.csv'}")


if __name__ == "__main__":
    main()
