"""Step 14.3: controlled ByteTrack ablation on annotated WILDTRACK PNGs.

Run from the project root:
    python -m src.track_ablation --frames 10
    python -m src.track_ablation --frames 100

Reuses the SAME cached YOLO detections for every tracker configuration.
This evaluates post-tracker matched-box coverage and identity diagnostics;
formal MOT metrics and annotation-region precision remain separate tasks.
"""

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wildtrack_eval"
OUT = ROOT / "outputs" / "evaluation" / "bytetrack_ablation"
WEIGHTS = ROOT / "models" / "yolo26n.pt"
CAMERAS = ("C1", "C2", "C3", "C4")
CONF = 0.10
IMGSZ = 640
NMS_IOU = 0.70

# Match the existing src/tracker.py baseline exactly; vary only two thresholds.
SETTINGS = {
    "baseline": {"track_high_thresh": 0.25, "new_track_thresh": 0.35},
    "moderate": {"track_high_thresh": 0.20, "new_track_thresh": 0.25},
    "permissive": {"track_high_thresh": 0.15, "new_track_thresh": 0.15},
}


def tracker_config(setting):
    """Return the same settings as production except for selected thresholds."""
    high, new = setting["track_high_thresh"], setting["new_track_thresh"]
    if not 0.10 < high <= new <= 1.0:
        raise ValueError("Require 0.10 < track_high_thresh <= new_track_thresh <= 1")
    return SimpleNamespace(
        track_high_thresh=high,
        track_low_thresh=0.10,
        new_track_thresh=new,
        track_buffer=10,
        match_thresh=0.80,
        fuse_score=True,
    )


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_spec(frame_ids, data_dir, weights):
    """Key all cached detections to their dataset/weights and detector settings."""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    return {
        "frame_ids": list(frame_ids),
        "data_dir": str(Path(data_dir).resolve()),
        "weights_sha256": sha256_file(weights),
        "imgsz": IMGSZ,
        "confidence": CONF,
        "nms_iou": NMS_IOU,
        "cameras": list(CAMERAS),
    }


def validate_records(records, frame_ids):
    if [int(row["frame_id"]) for row in records] != list(frame_ids):
        raise ValueError("Cached frame IDs do not match the requested evaluation")
    for row in records:
        if set(row["detections"]) != set(CAMERAS):
            raise ValueError("Cached detections must contain exactly four cameras")
        for cam in CAMERAS:
            for det in row["detections"][cam]:
                if len(det["xyxy"]) != 4 or not 0 <= float(det["confidence"]) <= 1:
                    raise ValueError(f"Malformed detection for {cam}")
    return records


def read_cache(out_dir, spec):
    manifest = out_dir / "cache_manifest.json"
    cache = out_dir / "detections.jsonl"
    if not (manifest.is_file() and cache.is_file()):
        return None
    if json.loads(manifest.read_text(encoding="utf-8")) != spec:
        return None
    records = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    return validate_records(records, spec["frame_ids"])


def save_cache(out_dir, spec, records):
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_records(records, spec["frame_ids"])
    target = out_dir / "detections.jsonl"
    temp = target.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    temp.replace(target)
    (out_dir / "cache_manifest.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


def load_images(data_dir, frame_id):
    import cv2
    frames = {}
    for camera in CAMERAS:
        path = Path(data_dir) / "Image_subsets" / camera / f"{frame_id:08d}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot decode {path}")
        frames[camera] = image
    return frames


def build_cache(data_dir, frame_ids, weights, threads=4):
    import torch
    from ultralytics import YOLO

    # Explicitly hold all detector settings constant across tracker variants.
    model = YOLO(str(weights))
    records = []
    for idx, fid in enumerate(frame_ids, start=1):
        frames = load_images(data_dir, fid)
        results = model.predict(
            source=[frames[cam] for cam in CAMERAS],
            classes=[0], conf=CONF, iou=NMS_IOU, imgsz=IMGSZ,
            device="cpu", verbose=False,
        )
        detections = {
            cam: [{"xyxy": box, "confidence": score, "class_id": int(cls)}
                  for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                             result.boxes.conf.cpu().tolist(),
                                             result.boxes.cls.cpu().tolist())]
            for cam, result in zip(CAMERAS, results)
        }
        if idx == 1:
            torch.set_num_threads(threads)  # Ultralytics may reset threads on first predict.
        records.append({"frame_id": fid, "detections": detections})
        if idx % 10 == 0 or idx == len(frame_ids):
            print(f"Caching YOLO detections: {idx}/{len(frame_ids)} frame sets", flush=True)
    return validate_records(records, frame_ids)


def run_setting(records, data_dir, setting, tracker_factory=None):
    from .evaluate import parse_gt, match_frame, identity_f1
    if tracker_factory is None:
        from .tracker import MultiCameraTracker
        tracker_factory = MultiCameraTracker

    config = tracker_config(setting)
    tracker = tracker_factory(cameras=CAMERAS, config=config)
    gt_frames = {}
    predicted_frames = {}
    stats = {cam: Counter() for cam in CAMERAS}
    gt_track_pairs = {cam: Counter() for cam in CAMERAS}
    last_track = {}  # (camera, ground-truth person) -> last matched ID

    for record in records:
        fid = int(record["frame_id"])
        frames = load_images(data_dir, fid)
        truth = parse_gt(data_dir, fid, width=frames["C1"].shape[1],
                         height=frames["C1"].shape[0])
        tracks = tracker.update(frames, record["detections"])
        gt_frames[fid], predicted_frames[fid] = truth, tracks
        for cam in CAMERAS:
            matches = match_frame(truth[cam], tracks[cam], threshold=0.5)
            s = stats[cam]
            s["gt"] += len(truth[cam])
            s["tracks"] += len(tracks[cam])
            s["matched"] += len(matches)
            for gi, pi in matches:
                pid = int(truth[cam][gi]["gt_id"])
                track_id = int(tracks[cam][pi]["track_id"])
                gt_track_pairs[cam][(pid, track_id)] += 1
                key = (cam, pid)
                if key in last_track and last_track[key] != track_id:
                    s["observed_local_id_changes"] += 1
                last_track[key] = track_id
    result = {}
    for cam in CAMERAS:
        s = stats[cam]
        gt, ntrk, nmatched = s["gt"], s["tracks"], s["matched"]
        result[cam] = {
            "gt_boxes": gt,
            "output_tracks": ntrk,
            "matched_boxes": nmatched,
            "missed_gt_boxes": gt - nmatched,
            "unmatched_tracks": ntrk - nmatched,
            "matched_gt_recall": round(nmatched / gt, 4) if gt else None,
            "precision_proxy_unverified_roi": round(nmatched / ntrk, 4) if ntrk else None,
            "observed_local_id_changes": s["observed_local_id_changes"],
            "local_identity_f1_diagnostic": identity_f1(gt_track_pairs[cam], gt, ntrk),
        }
    aggregate = Counter()
    for s in stats.values():
        aggregate.update(s)
    total_gt = aggregate["gt"]
    total_tracks = aggregate["tracks"]
    total_matched = aggregate["matched"]
    overall = {
        "gt_boxes": total_gt,
        "output_tracks": total_tracks,
        "matched_boxes": total_matched,
        "missed_gt_boxes": total_gt - total_matched,
        "unmatched_tracks": total_tracks - total_matched,
        "matched_gt_recall": round(total_matched / total_gt, 4) if total_gt else None,
        "precision_proxy_unverified_roi": round(total_matched / total_tracks, 4) if total_tracks else None,
        "observed_local_id_changes": aggregate["observed_local_id_changes"],
    }
    return {"overall": overall, "per_camera": result}, gt_frames, predicted_frames


def run_ablation(data_dir, out_dir, frame_ids, records, export=True, tracker_factory=None):
    from .evaluate import export_trackeval
    comparison = {
        "basis": "Same cached YOLO boxes on matched WILDTRACK PNGs, 2 FPS annotated frames",
        "frame_sets": len(frame_ids), "cameras": list(CAMERAS),
        "detector": {"weights": "yolo26n.pt", "imgsz": IMGSZ, "conf": CONF},
        "settings": {},
        "limitations": [
            "Matched GT recall is at IoU 0.5 on a non-ROI-filtered annotation subset.",
            "Unmatched tracks are not necessarily false positives; official ROI was not verified.",
            "Identity F1 is a diagnostic, not official TrackEval IDF1 or HOTA.",
            "Observed local ID changes are not official MOTChallenge IDSW.",
            "Tracking on 2 FPS annotated PNGs does not measure 5 FPS RTSP performance.",
            "Ground truth is used for scoring only; tracker receives cached YOLO boxes.",
        ],
    }
    for name, setting in SETTINGS.items():
        metrics, gt_frames, pred_frames = run_setting(records, data_dir, setting,
                                                      tracker_factory=tracker_factory)
        output = Path(out_dir) / name
        output.mkdir(parents=True, exist_ok=True)
        if export:
            export_trackeval(gt_frames, pred_frames, output / "trackeval")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        comparison["settings"][name] = {
            "parameters": {"track_low_thresh": 0.10, "track_buffer": 10,
                           "match_thresh": 0.80, "fuse_score": True, **setting},
            **metrics,
        }
        print(f"{name}: recall={metrics['overall']['matched_gt_recall']:.3f}, "
              f"tracks={metrics['overall']['output_tracks']}, "
              f"ID changes={metrics['overall']['observed_local_id_changes']}", flush=True)
    baseline = comparison["settings"]["baseline"]["overall"]
    for name, report in comparison["settings"].items():
        other = report["overall"]
        report["versus_baseline"] = {
            "matched_boxes_difference": other["matched_boxes"] - baseline["matched_boxes"],
            "unmatched_tracks_difference": other["unmatched_tracks"] - baseline["unmatched_tracks"],
            "observed_id_changes_difference": (other["observed_local_id_changes"] -
                                               baseline["observed_local_id_changes"]),
        }
    path = Path(out_dir) / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 400 or args.threads < 1:
        parser.error("--frames must be 1..400, --threads must be positive")
    from .evaluate import available_frames
    frame_ids = available_frames(args.data_dir, args.frames)
    out = args.output_dir / f"frames_{len(frame_ids)}"
    spec = cache_spec(frame_ids, args.data_dir, args.weights)
    records = None if args.rebuild_cache else read_cache(out, spec)
    if records is None:
        print("Creating shared raw YOLO detection cache...")
        records = build_cache(args.data_dir, frame_ids, args.weights, args.threads)
        save_cache(out, spec, records)
    else:
        print(f"Reusing cached detections: {out / 'detections.jsonl'}")
    run_ablation(args.data_dir, out, frame_ids, records)


if __name__ == "__main__":
    main()
