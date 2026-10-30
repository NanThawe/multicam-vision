"""Step 14: evaluate the four-camera pipeline on labelled WILDTRACK PNG frames.

Run from the project root:
    python -m src.evaluate download --frames 20
    python -m src.evaluate run --frames 20

This intentionally does NOT evaluate raw MP4 frames against PNG annotations:
the labelled PNGs are synchronised and distortion-corrected by the dataset.
"""

import argparse
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

CAMERAS = ("C1", "C2", "C3", "C4")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wildtrack_eval"
OUT = ROOT / "outputs" / "evaluation"
HF_BASE = ("https://huggingface.co/datasets/disl/my_dataset/resolve/main/"
           "Wildtrack_dataset_full/Wildtrack_dataset")


def download_subset(folder, frames=20):
    """Download only the matching annotation and PNG files (about 11 MB/frame set)."""
    if not 1 <= frames <= 400:
        raise ValueError("frames must be between 1 and 400")
    folder = Path(folder)
    for i in range(frames):
        name = f"{i * 5:08d}"
        paths = [f"annotations_positions/{name}.json"] + [
            f"Image_subsets/{camera}/{name}.png" for camera in CAMERAS]
        for rel in paths:
            dest = folder / rel
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            print(f"Downloading {rel}", flush=True)
            try:
                req = urllib.request.Request(f"{HF_BASE}/{rel}",
                                             headers={"User-Agent": "multicam-evaluation/1.0"})
                with urllib.request.urlopen(req, timeout=90) as response, tmp.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                tmp.replace(dest)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
    print(f"Saved {frames} matched frame sets to {folder}")


def available_frames(folder, limit):
    """Use only IDs found in annotations AND each image folder."""
    folder = Path(folder)
    gt = {p.stem for p in (folder / "annotations_positions").glob("*.json")}
    if not gt:
        raise FileNotFoundError("Missing annotations_positions/*.json; run download first")
    for camera in CAMERAS:
        gt &= {p.stem for p in (folder / "Image_subsets" / camera).glob("*.png")}
    ids = sorted(gt, key=int)
    if not ids:
        raise FileNotFoundError("No annotated frame has PNG images from all four cameras")
    if limit:
        ids = ids[:limit]
    return [int(x) for x in ids]


def parse_gt(folder, frame_id, cameras=CAMERAS, width=1920, height=1080):
    """Decode true personID and per-camera boxes. viewNum is zero-indexed."""
    path = Path(folder) / "annotations_positions" / f"{frame_id:08d}.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected annotation list: {path}")
    result = {camera: [] for camera in cameras}
    for person in entries:
        pid = int(person["personID"])
        for view in person["views"]:
            camera = f"C{int(view['viewNum']) + 1}"
            if camera not in result:
                continue
            raw = [float(view[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            if all(v == -1 for v in raw):
                continue
            x1, y1, x2, y2 = raw
            # WILDTRACK includes boxes partially outside the image.
            x1, x2 = max(x1, 0.0), min(x2, float(width))
            y1, y2 = max(y1, 0.0), min(y2, float(height))
            if x2 > x1 and y2 > y1:
                result[camera].append({"gt_id": pid, "xyxy": [x1, y1, x2, y2]})
    return result


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_frame(gt, predictions, threshold=0.5):
    """One-to-one IoU matching; return (ground-truth index, prediction index)."""
    from scipy.optimize import linear_sum_assignment
    if not gt or not predictions:
        return []
    scores = [[iou(g["xyxy"], p["xyxy"]) for p in predictions] for g in gt]
    # Forbid below-threshold assignments. Hungarian maximises total IoU.
    costs = [[1 - s if s >= threshold else 1e6 for s in row] for row in scores]
    rows, cols = linear_sum_assignment(costs)
    return [(int(r), int(c)) for r, c in zip(rows, cols)
            if scores[r][c] >= threshold]


def identity_f1(pair_counts, gt_total, pred_total):
    """Diagnostic globally assigned identity F1 from IoU-matched observations.

    NOT official TrackEval IDF1. Official tracking metrics require TrackEval.
    """
    from scipy.optimize import linear_sum_assignment
    if gt_total + pred_total == 0:
        return None
    gids = sorted({gt for gt, _ in pair_counts}, key=str)
    pids = sorted({pred for _, pred in pair_counts}, key=str)
    correct = 0
    if gids and pids:
        weights = [[pair_counts.get((g, p), 0) for p in pids] for g in gids]
        rows, cols = linear_sum_assignment(weights, maximize=True)
        correct = sum(weights[r][c] for r, c in zip(rows, cols))
    return round(2 * correct / (gt_total + pred_total), 4)


def compute_metrics(gt_frames, pred_frames, threshold=0.5):
    """Compute box matching and cross-camera identity diagnostics.

    gt_frames, pred_frames: {frame_id: {camera: [{gt_id|track_id, xyxy, ...}]}}
    Global IDs are compared across cameras at each annotated instant.
    """
    if set(gt_frames) != set(pred_frames):
        raise ValueError("Prediction frame IDs do not match the labelled frame IDs")
    cameras = CAMERAS
    per_camera = {cam: {"gt_boxes": 0, "pred_boxes": 0, "matched_boxes": 0,
                        "observed_local_id_changes": 0} for cam in cameras}
    pair_local = {cam: Counter() for cam in cameras}
    pair_global = Counter()
    previous = {}
    global_gt_total = global_pred_total = 0
    gt_cross_pairs = pred_cross_pairs = true_cross_pairs = 0
    for frame_id in sorted(gt_frames):
        gt = gt_frames[frame_id]
        pr = pred_frames[frame_id]
        matched_by_cam = {}
        for cam in cameras:
            truth = gt.get(cam, [])
            people = pr.get(cam, [])
            stats = per_camera[cam]
            stats["gt_boxes"] += len(truth)
            stats["pred_boxes"] += len(people)
            global_gt_total += len(truth)
            global_pred_total += len(people)
            matches = match_frame(truth, people, threshold)
            stats["matched_boxes"] += len(matches)
            mapping = {}
            for gi, pi in matches:
                g_id = truth[gi]["gt_id"]
                p = people[pi]
                lid = p.get("track_id")
                if lid is not None:
                    pair_local[cam][(g_id, lid)] += 1
                    key = (cam, g_id)
                    if key in previous and previous[key] != lid:
                        stats["observed_local_id_changes"] += 1
                    previous[key] = lid
                gid = p.get("global_id")
                if gid is not None:
                    pair_global[(g_id, gid)] += 1
                mapping[pi] = g_id
            matched_by_cam[cam] = mapping
        # Only compare identity across DIFFERENT cameras at the same annotated instant.
        for index, a in enumerate(cameras):
            for b in cameras[index + 1:]:
                true_a = {p["gt_id"] for p in gt.get(a, [])}
                true_b = {p["gt_id"] for p in gt.get(b, [])}
                gt_cross_pairs += len(true_a & true_b)
                for ia, pa in enumerate(pr.get(a, [])):
                    gid = pa.get("global_id")
                    if gid is None:
                        continue
                    for ib, pb in enumerate(pr.get(b, [])):
                        if pb.get("global_id") != gid:
                            continue
                        pred_cross_pairs += 1
                        if (matched_by_cam[a].get(ia) is not None and
                            matched_by_cam[a].get(ia) == matched_by_cam[b].get(ib)):
                            true_cross_pairs += 1
    for cam, stats in per_camera.items():
        tp, ng, np = stats["matched_boxes"], stats["gt_boxes"], stats["pred_boxes"]
        stats["detection_precision"] = round(tp / np, 4) if np else None
        stats["detection_recall"] = round(tp / ng, 4) if ng else None
        stats["local_identity_f1_diagnostic"] = identity_f1(pair_local[cam], ng, np)
    precision = true_cross_pairs / pred_cross_pairs if pred_cross_pairs else None
    recall = true_cross_pairs / gt_cross_pairs if gt_cross_pairs else None
    return {
        "evaluation_basis": "annotated, matched WILDTRACK PNG images",
        "iou_threshold": threshold,
        "annotated_frames": len(gt_frames),
        "per_camera": per_camera,
        "cross_camera": {
            "true_gt_pairs": gt_cross_pairs,
            "predicted_same_identity_pairs": pred_cross_pairs,
            "correct_same_identity_pairs": true_cross_pairs,
            "pairwise_precision": round(precision, 4) if precision is not None else None,
            "pairwise_recall": round(recall, 4) if recall is not None else None,
            "pairwise_f1": round(2 * precision * recall / (precision + recall), 4)
            if precision is not None and recall is not None and precision + recall else None,
            "global_identity_f1_diagnostic": identity_f1(pair_global, global_gt_total, global_pred_total),
        },
        "warnings": [
            "Identity F1 fields are matching diagnostics, not official TrackEval IDF1 or HOTA.",
            "Observed local ID changes are not the official MOTChallenge IDSW metric.",
            "Metrics describe the sampled annotated image subset, not raw MP4 or live RTSP performance.",
            "Pairwise cross-camera metrics count person pairs per annotated frame and are detection-dependent.",
        ],
    }



def export_trackeval(gt_frames, pred_frames, dest):
    """Write per-camera MOTChallenge-format files for official TrackEval.

    This exports local ByteTrack IDs, not cross-camera ReID IDs. HOTA/IDF1
    calculated by TrackEval are therefore *single-camera* tracking metrics.
    """
    dest = Path(dest)
    benchmark = "WILDTRACK-EVAL-train"
    base_gt = dest / "gt" / "mot_challenge"
    base_pr = dest / "trackers" / "mot_challenge" / benchmark / "multicam" / "data"
    base_pr.mkdir(parents=True, exist_ok=True)
    seqmap = base_gt / "seqmaps" / f"{benchmark}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(CAMERAS) + "\n", encoding="utf-8")

    for cam in CAMERAS:
        seq_folder = base_gt / benchmark / cam
        gt_path = seq_folder / "gt" / "gt.txt"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path = base_pr / f"{cam}.txt"
        truth_rows, pred_rows = [], []
        for seq_frame, frame_id in enumerate(sorted(gt_frames), start=1):
            for obj in gt_frames[frame_id].get(cam, []):
                x1, y1, x2, y2 = obj["xyxy"]
                # MOTChallenge expects 1-based IDs and frame numbers.
                # Ground truth: mark=1, class=1 (pedestrian), visibility=1.
                truth_rows.append(f"{seq_frame},{int(obj['gt_id']) + 1},{x1:.2f},{y1:.2f},"
                                  f"{x2-x1:.2f},{y2-y1:.2f},1,1,1,-1")
            for obj in pred_frames[frame_id].get(cam, []):
                if obj.get("track_id") is None:
                    continue
                x1, y1, x2, y2 = obj["xyxy"]
                pred_rows.append(f"{seq_frame},{int(obj['track_id']) + 1},{x1:.2f},{y1:.2f},"
                                 f"{x2-x1:.2f},{y2-y1:.2f},"
                                 f"{float(obj.get('confidence', 1.0)):.3f},-1,-1,-1")
        gt_path.write_text("\n".join(truth_rows) + "\n", encoding="utf-8")
        pred_path.write_text("\n".join(pred_rows) + "\n", encoding="utf-8")
        (seq_folder / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={cam}\n"
            "imDir=img1\n"
            "frameRate=2\n"
            f"seqLength={len(gt_frames)}\n"
            "imWidth=1920\n"
            "imHeight=1080\n"
            "imExt=.png\n", encoding="utf-8")
    return dest

def run_pipeline(data_dir, output_dir, frames=20, threshold=0.5):
    import cv2
    from .detector import PersonDetector
    from .tracker import MultiCameraTracker
    from .reid import AppearanceEncoder, CrossCameraMatcher

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = available_frames(data_dir, frames)
    detector = PersonDetector()
    tracker = MultiCameraTracker()
    matcher = CrossCameraMatcher(AppearanceEncoder(backend="pytorch"),
                                 refresh_every=5, batch_across_cameras=True)
    gt_frames, pred_frames = {}, {}
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for idx, frame_id in enumerate(frame_ids, start=1):
            images = {}
            for cam in CAMERAS:
                path = Path(data_dir) / "Image_subsets" / cam / f"{frame_id:08d}.png"
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode {path}")
                images[cam] = image
            truth = parse_gt(data_dir, frame_id,
                             width=images[CAMERAS[0]].shape[1],
                             height=images[CAMERAS[0]].shape[0])
            local = tracker.update(images, detector.detect(images))
            predicted, _ = matcher.update(frame_id, images, local)
            gt_frames[frame_id], pred_frames[frame_id] = truth, predicted
            for camera in CAMERAS:
                out.write(json.dumps({"frame_id": frame_id, "camera": camera,
                                      "tracks": predicted[camera]}) + "\n")
            print(f"[{idx}/{len(frame_ids)}] frame {frame_id}: "
                  f"{sum(len(p) for p in predicted.values())} tracks", flush=True)
    metrics = compute_metrics(gt_frames, pred_frames, threshold)
    export_trackeval(gt_frames, pred_frames, output_dir / "trackeval")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved", output_dir / "metrics.json")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Download labelled image/annotation pairs (not raw MP4s)")
    d.add_argument("--frames", type=int, default=20)
    d.add_argument("--data-dir", type=Path, default=DATA)
    r = sub.add_parser("run", help="Run inference and evaluate predictions against ground truth")
    r.add_argument("--frames", type=int, default=20)
    r.add_argument("--data-dir", type=Path, default=DATA)
    r.add_argument("--output-dir", type=Path, default=OUT)
    r.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.command == "download":
        download_subset(args.data_dir, args.frames)
    elif args.command == "run":
        if args.frames < 1 or not 0 < args.iou <= 1:
            p.error("--frames must be positive and --iou in (0, 1]")
        run_pipeline(args.data_dir, args.output_dir, args.frames, args.iou)


if __name__ == "__main__":
    main()
