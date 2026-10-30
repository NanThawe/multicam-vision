"""Visualise WILDTRACK pipeline box errors without rerunning inference.

Run from project root:
    python -m src.error_analysis

Reads labelled PNGs/JSON and Step 14's saved predictions.jsonl.
Labels describe post-ByteTrack boxes, not raw YOLO detections.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .evaluate import CAMERAS, DATA, OUT, iou, match_frame, parse_gt

SIZE_GROUPS = ("small", "medium", "large")


def size_group(box):
    """Group by original-image box height: <64, 64-159, >=160 px."""
    height = box[3] - box[1]
    return "small" if height < 64 else "medium" if height < 160 else "large"


def read_predictions(path, limit=0):
    """Read Step 14 JSONL. Require four camera records per timestamp."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; first run: python -m src.evaluate run --frames 100")
    records = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            fid, camera = int(item["frame_id"]), str(item["camera"])
            if camera not in CAMERAS:
                raise ValueError(f"Unexpected camera {camera} on line {lineno}")
            if camera in records[fid]:
                raise ValueError(f"Duplicate prediction for frame {fid}, {camera}")
            tracks = item["tracks"]
            if not isinstance(tracks, list):
                raise ValueError(f"Expected a list of tracks at line {lineno}")
            records[fid][camera] = tracks
    ids = sorted(records)
    if limit < 0:
        raise ValueError("--frames must be nonnegative")
    if limit:
        ids = ids[:limit]
    if not ids:
        raise ValueError("Prediction file contains no frames")
    for fid in ids:
        if set(records[fid]) != set(CAMERAS):
            raise ValueError(f"Incomplete four-camera predictions at frame {fid}")
    return {fid: records[fid] for fid in ids}


def classify_frame(truth, predicted, threshold=0.5):
    """Return one-to-one matched boxes and indices not meeting IoU threshold."""
    matches = match_frame(truth, predicted, threshold)
    gt_hit = {i for i, _ in matches}
    pred_hit = {j for _, j in matches}
    unmatched_gt = [i for i in range(len(truth)) if i not in gt_hit]
    unmatched_pred = [j for j in range(len(predicted)) if j not in pred_hit]
    # An unmatched prediction overlapping an already matched GT may be a duplicate.
    # Others with modest overlap may be localisation errors. Neither is conclusive.
    overlap = Counter()
    for j in unmatched_pred:
        best = max((iou(predicted[j]["xyxy"], g["xyxy"]) for g in truth), default=0.0)
        if best >= threshold:
            overlap["overlap_gt_at_least_threshold"] += 1
        elif best >= 0.1:
            overlap["overlap_gt_0.1_to_threshold"] += 1
        else:
            overlap["little_or_no_gt_overlap"] += 1
    return {"matches": matches, "unmatched_gt": unmatched_gt,
            "unmatched_pred": unmatched_pred, "unmatched_overlap": dict(overlap)}


def analyse_records(predictions, data_dir, threshold=0.5):
    """Summarise post-tracking box coverage and prepare per-frame example rankings."""
    data_dir = Path(data_dir)
    summary = {cam: {"gt": 0, "predictions": 0, "matched": 0,
                     "gt_by_height": {key: {"gt": 0, "matched": 0} for key in SIZE_GROUPS},
                     "unmatched_overlap": Counter(), "matched_conf": [], "unmatched_conf": []}
               for cam in CAMERAS}
    frames = []
    classified = {}
    for fid, preds in predictions.items():
        annotation = data_dir / "annotations_positions" / f"{fid:08d}.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing matching ground truth: {annotation}")
        gt = parse_gt(data_dir, fid)
        classified[fid] = {}
        for cam in CAMERAS:
            people, truth = preds[cam], gt[cam]
            info = classify_frame(truth, people, threshold)
            classified[fid][cam] = info
            stats = summary[cam]
            stats["gt"] += len(truth)
            stats["predictions"] += len(people)
            stats["matched"] += len(info["matches"])
            stats["unmatched_overlap"].update(info["unmatched_overlap"])
            matched_gt = {i for i, _ in info["matches"]}
            for i, item in enumerate(truth):
                cell = stats["gt_by_height"][size_group(item["xyxy"])]
                cell["gt"] += 1
                cell["matched"] += int(i in matched_gt)
            for _, j in info["matches"]:
                if people[j].get("confidence") is not None:
                    stats["matched_conf"].append(float(people[j]["confidence"]))
            for j in info["unmatched_pred"]:
                if people[j].get("confidence") is not None:
                    stats["unmatched_conf"].append(float(people[j]["confidence"]))
            frames.append({"frame_id": fid, "camera": cam, "gt": len(truth),
                           "predictions": len(people), "matched": len(info["matches"]),
                           "missed": len(info["unmatched_gt"]),
                           "unmatched": len(info["unmatched_pred"])})
    result = {}
    for cam, stats in summary.items():
        n_gt, n_pred, matched = stats["gt"], stats["predictions"], stats["matched"]
        result[cam] = {
            "gt": n_gt, "predictions": n_pred, "matched": matched,
            "missed": n_gt - matched, "unmatched": n_pred - matched,
            "precision": round(matched / n_pred, 4) if n_pred else None,
            "recall": round(matched / n_gt, 4) if n_gt else None,
            "gt_by_height": {key: {**cell, "recall": round(cell["matched"] / cell["gt"], 4)
                                    if cell["gt"] else None}
                             for key, cell in stats["gt_by_height"].items()},
            "unmatched_overlap": dict(stats["unmatched_overlap"]),
            "median_confidence_matched": round(median(stats["matched_conf"]), 3)
            if stats["matched_conf"] else None,
            "median_confidence_unmatched": round(median(stats["unmatched_conf"]), 3)
            if stats["unmatched_conf"] else None,
        }
    ng = sum(x["gt"] for x in result.values())
    npred = sum(x["predictions"] for x in result.values())
    tp = sum(x["matched"] for x in result.values())
    report = {
        "basis": "WILDTRACK annotated PNGs versus post-ByteTrack/ReID saved tracks",
        "annotated_frame_sets": len(predictions), "iou_threshold": threshold,
        "height_groups_px": {"small": "<64", "medium": "64 to <160", "large": ">=160"},
        "overall": {"gt": ng, "predictions": npred, "matched": tp,
                    "missed": ng-tp, "unmatched": npred-tp,
                    "precision": round(tp/npred, 4) if npred else None,
                    "recall": round(tp/ng, 4) if ng else None},
        "per_camera": result,
        "limitations": [
            "All box metrics are post-tracking pipeline outputs, not standalone YOLO accuracy.",
            "Unmatched predictions need visual review; they may include annotation or box-alignment issues.",
            "Height bins are descriptive and are not WILDTRACK difficulty categories.",
            "No inference was rerun. Results describe only the saved predictions.",
            "No formal HOTA or TrackEval IDF1 is computed here.",
        ]
    }
    return report, frames, classified


def choose_examples(frame_rows, camera, count):
    """Prioritise high missed-box and high unmatched-box counts, without repeats."""
    rows = [r for r in frame_rows if r["camera"] == camera]
    picked = []
    for key in ("missed", "unmatched"):
        ranked = sorted(rows, key=lambda r: (-r[key], -r["gt"], r["frame_id"]))
        for row in ranked:
            if row["frame_id"] not in picked:
                picked.append(row["frame_id"])
                break
        if len(picked) >= count:
            return picked[:count]
    for row in sorted(rows, key=lambda r: (-(r["missed"] + r["unmatched"]), r["frame_id"])):
        if row["frame_id"] not in picked:
            picked.append(row["frame_id"])
        if len(picked) >= count:
            break
    return picked


def draw_example(image, truth, predicted, info, camera, frame_id, threshold=0.5, display_width=960):
    """Overlay TP (green), missed GT (red), unmatched tracks (amber)."""
    import cv2
    height, width = image.shape[:2]
    factor = display_width / width
    view = cv2.resize(image, (display_width, round(height*factor)))
    header = 56
    canvas = cv2.copyMakeBorder(view, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))

    def rect(box, color, label):
        x1, y1, x2, y2 = [round(v*factor) for v in box]
        cv2.rectangle(canvas, (x1, y1+header), (x2, y2+header), color, 2)
        cv2.putText(canvas, label, (max(0, x1), max(header+14, y1+header-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for i in info["unmatched_gt"]:
        gt = truth[i]
        rect(gt["xyxy"], (0, 0, 255), f"MISS GT:{gt['gt_id']}")
    for i, j in info["matches"]:
        gt, person = truth[i], predicted[j]
        rect(person["xyxy"], (0, 210, 0), f"MATCH GT:{gt['gt_id']} L:{person.get('track_id', '?')}")
    for j in info["unmatched_pred"]:
        person = predicted[j]
        rect(person["xyxy"], (0, 190, 255), f"UNMATCH L:{person.get('track_id', '?')}")
    cv2.putText(canvas, f"{camera}  frame {frame_id:08d}   IoU >= {threshold:.2f}", (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "GREEN matched    RED missed ground truth    AMBER unmatched track",
                (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
    return canvas


def run(pred_path, data_dir, out_dir, frames=0, iou_threshold=0.5, examples=2):
    import cv2
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if examples < 0:
        raise ValueError("examples must be >= 0")
    predictions = read_predictions(pred_path, limit=frames)
    report, rows, matches = analyse_records(predictions, data_dir, iou_threshold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_id", "camera", "gt", "predictions",
                                                       "matched", "missed", "unmatched"])
        writer.writeheader()
        writer.writerows(rows)
    if examples:
        gallery_dir = out_dir / "examples"
        gallery_dir.mkdir(exist_ok=True)
        for cam in CAMERAS:
            for fid in choose_examples(rows, cam, examples):
                image_path = Path(data_dir) / "Image_subsets" / cam / f"{fid:08d}.png"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(f"Cannot read annotated camera image: {image_path}")
                gt = parse_gt(data_dir, fid, width=image.shape[1], height=image.shape[0])[cam]
                picture = draw_example(image, gt, predictions[fid][cam], matches[fid][cam], cam, fid, iou_threshold)
                dest = gallery_dir / f"{cam}_frame_{fid:08d}.jpg"
                if not cv2.imwrite(str(dest), picture):
                    raise OSError(f"Failed to write {dest}")
    print(f"Saved {out_dir / 'summary.json'} and {out_dir / 'per_frame.csv'}")
    if examples:
        print(f"Saved annotated images to {out_dir / 'examples'}")
    for cam, s in report["per_camera"].items():
        print(f"{cam}: {s['matched']}/{s['gt']} GT matched; "
              f"missed={s['missed']}, unmatched tracks={s['unmatched']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUT / "predictions.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT / "error_analysis")
    parser.add_argument("--frames", type=int, default=0,
                        help="Read all saved frames by default; limit to first N if supplied")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=2,
                        help="Number of error overlay images per camera")
    args = parser.parse_args(argv)
    run(args.predictions, args.data_dir, args.output_dir, args.frames,
        args.iou, args.examples)


if __name__ == "__main__":
    main()
