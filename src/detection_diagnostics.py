"""Step 14.2: audit WILDTRACK geometry and compare raw YOLO vs saved tracks.

Run from project root:
    python -m src.detection_diagnostics audit --frames 100
    python -m src.detection_diagnostics compare --frames 20 --sizes 640,960,1280

No training and no modifications to the live inference pipeline.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from .evaluate import CAMERAS, DATA, OUT, available_frames, match_frame, parse_gt
from .error_analysis import read_predictions

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = OUT / "detection_diagnostics"
WEIGHTS = ROOT / "models" / "yolo26n.pt"


def validate_box(box, width, height):
    """Check the geometry of an unmodified or predicted XYXY box."""
    return (len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
            and all(float('-inf') < float(v) < float('inf') for v in box)
            and box[0] < box[2] and box[1] < box[3]
            and box[0] >= 0 and box[1] >= 0
            and box[2] <= width and box[3] <= height)


def image_and_gt(data_dir, frame_id):
    import cv2
    frames = {}
    for cam in CAMERAS:
        filename = Path(data_dir) / "Image_subsets" / cam / f"{frame_id:08d}.png"
        img = cv2.imread(str(filename))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {filename}")
        frames[cam] = img
    if len({img.shape[:2] for img in frames.values()}) != 1:
        raise ValueError(f"Camera image sizes differ at {frame_id}")
    h, w = frames[CAMERAS[0]].shape[:2]
    return frames, parse_gt(data_dir, frame_id, width=w, height=h)


def annotation_audit(data_dir, frame_ids):
    """Verify file, camera and box geometry; never infer ROI from GT density."""
    per_camera = {cam: Counter() for cam in CAMERAS}
    sizes = {}
    problems = []
    for fid in frame_ids:
        images, truth = image_and_gt(data_dir, fid)
        raw = json.loads((Path(data_dir) / "annotations_positions" /
                          f"{fid:08d}.json").read_text(encoding="utf-8"))
        pids = [int(x["personID"]) for x in raw]
        if len(set(pids)) != len(pids):
            problems.append(f"Frame {fid}: repeated personID in one annotation file")
        for cam in CAMERAS:
            h, w = images[cam].shape[:2]
            sizes.setdefault(cam, [w, h])
            if sizes[cam] != [w, h]:
                problems.append(f"{cam}: image resolution varies at {fid}")
            per_camera[cam]["gt_boxes"] += len(truth[cam])
            for g in truth[cam]:
                if not validate_box(g["xyxy"], w, h):
                    problems.append(f"{cam} frame {fid}: invalid clipped GT box")
        # Raw boxes can extend past the edge of the original image; report them
        # separately rather than treating intentional clipping as invalid GT.
        for person in raw:
            for view in person.get("views", []):
                cam = f"C{int(view['viewNum']) + 1}"
                if cam not in CAMERAS:
                    continue
                box = [float(view[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                if all(v == -1 for v in box):
                    continue
                per_camera[cam]["raw_visible_views"] += 1
                h, w = images[cam].shape[:2]
                if not validate_box(box, w, h):
                    per_camera[cam]["raw_outside_or_invalid"] += 1
                if box[2] <= box[0] or box[3] <= box[1]:
                    problems.append(f"{cam} frame {fid}: degenerate raw GT box")
    return {
        "frame_sets": len(frame_ids),
        "image_dimensions": sizes,
        "per_camera": {cam: dict(per_camera[cam]) for cam in CAMERAS},
        "geometry_problems": problems,
        "annotation_region": "WILDTRACK's 3D area of interest; a verified projected ROI mask is not loaded",
        "warning": "An absent GT box is not proof that a visible person is a false positive."
    }


def load_roi(path):
    """Optional hand-verified image polygons in 0..1 coordinates, one per camera.

    This is a custom region analysis, NOT an official WILDTRACK ROI evaluation.
    """
    if path is None:
        return None
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(content) != set(CAMERAS):
        raise ValueError("ROI JSON must define C1, C2, C3 and C4")
    for cam, polygon in content.items():
        if len(polygon) < 3 or any(len(pt) != 2 or any(not 0 <= float(v) <= 1 for v in pt)
                                     for pt in polygon):
            raise ValueError(f"Invalid normalised polygon for {cam}")
    return content


def inside_polygon(point, polygon):
    """Ray casting; includes points on straight polygon edges."""
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-10 and min(ax, bx) - 1e-10 <= x <= max(ax, bx) + 1e-10 and min(ay, by) - 1e-10 <= y <= max(ay, by) + 1e-10:
            return True
        if (ay > y) != (by > y):
            if x < ax + (bx - ax) * (y - ay) / (by - ay):
                inside = not inside
    return inside


def apply_roi(items, polygon, width, height):
    if polygon is None:
        return list(items), 0
    kept = []
    for obj in items:
        x1, _, x2, y2 = obj["xyxy"]
        if inside_polygon(((x1 + x2) / (2 * width), y2 / height), polygon):
            kept.append(obj)
    return kept, len(items) - len(kept)


def box_stats(truth, predictions, threshold=0.5):
    """Matched/unmatched box counts, NOT verified full-image detection precision."""
    pairs = match_frame(truth, predictions, threshold)
    return {"gt": len(truth), "predictions": len(predictions),
            "matched": len(pairs), "missed": len(truth) - len(pairs),
            "unmatched": len(predictions) - len(pairs),
            "precision_proxy": round(len(pairs)/len(predictions), 4) if predictions else None,
            "recall": round(len(pairs)/len(truth), 4) if truth else None}, pairs


def new_totals():
    return Counter({"gt": 0, "predictions": 0, "matched": 0, "missed": 0, "unmatched": 0,
                    "ignored_gt": 0, "ignored_predictions": 0})


def accumulate(target, stat):
    for key in target:
        target[key] += stat.get(key, 0)


def finish_totals(target):
    result = dict(target)
    result["precision_proxy"] = round(target["matched"] / target["predictions"], 4) if target["predictions"] else None
    result["recall"] = round(target["matched"] / target["gt"], 4) if target["gt"] else None
    return result


def compare_predictions(truth, raw, tracked, polygon=None, image_size=(1920, 1080), iou_threshold=0.5):
    """One-frame diagnostics. Track-vs-detector differences are associations, not causality."""
    w, h = image_size
    gt, ignored_gt = apply_roi(truth, polygon, w, h)
    yolo, ignored_raw = apply_roi(raw, polygon, w, h)
    tracks, ignored_tracks = apply_roi(tracked, polygon, w, h)
    det, dm = box_stats(gt, yolo, iou_threshold)
    trk, tm = box_stats(gt, tracks, iou_threshold)
    det.update(ignored_gt=ignored_gt, ignored_predictions=ignored_raw)
    trk.update(ignored_gt=ignored_gt, ignored_predictions=ignored_tracks)
    dm_ids, tm_ids = {i for i, _ in dm}, {i for i, _ in tm}
    deltas = {
        "gt_matched_by_both": len(dm_ids & tm_ids),
        "gt_matched_only_by_yolo": len(dm_ids - tm_ids),
        "gt_matched_only_by_tracks": len(tm_ids - dm_ids),
        "gt_missed_by_both": len(gt) - len(dm_ids | tm_ids),
    }
    return det, trk, deltas, (gt, yolo, tracks, dm, tm)


def draw_overlay(image, gt, detected, matches, heading, width=960):
    """Red=missed GT; green=matched raw detection; amber=unmatched detection."""
    import cv2
    scale = min(1.0, width / image.shape[1])
    img = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    view = cv2.copyMakeBorder(img, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    gt_seen = {i for i, _ in matches}
    pred_seen = {j for _, j in matches}
    for objects, selected, colour in ((gt, gt_seen, (0, 0, 255)), (detected, pred_seen, (0, 185, 255))):
        for idx, obj in enumerate(objects):
            if idx in selected:
                continue
            x1, y1, x2, y2 = [round(v*scale) for v in obj["xyxy"]]
            cv2.rectangle(view, (x1, y1+35), (x2, y2+35), colour, 2)
    for _, j in matches:
        x1, y1, x2, y2 = [round(v*scale) for v in detected[j]["xyxy"]]
        cv2.rectangle(view, (x1, y1+35), (x2, y2+35), (0, 190, 0), 2)
    cv2.putText(view, heading, (7, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def evaluate_setting(model, data_dir, frame_ids, imgsz, confidence, nms_iou, match_iou,
                     roi, tracks, output_dir, examples=1, threads=4):
    import cv2
    from statistics import median
    import torch

    per_camera = {cam: {"raw_yolo": new_totals(), "post_tracker": new_totals(),
                        "difference": Counter()} for cam in CAMERAS}
    times = []
    example_counts = Counter()
    for index, fid in enumerate(frame_ids, 1):
        images, truth = image_and_gt(data_dir, fid)
        # This is the same class-0 YOLO detector, with resolution varied explicitly.
        t0 = perf_counter()
        results = model.predict(source=[images[c] for c in CAMERAS], device="cpu", imgsz=imgsz,
                                conf=confidence, iou=nms_iou, classes=[0], verbose=False)
        times.append((perf_counter()-t0)*1000)
        if torch.get_num_threads() != threads:
            # Ultralytics may reset threads while initialising its predictor.
            torch.set_num_threads(threads)
        for cam, result in zip(CAMERAS, results):
            detected = [{"xyxy": box, "confidence": conf} for box, conf in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]
            track_boxes = tracks[fid][cam]
            h, w = images[cam].shape[:2]
            det, trk, delta, visual = compare_predictions(
                truth[cam], detected, track_boxes,
                None if roi is None else roi[cam], (w, h), match_iou)
            accumulate(per_camera[cam]["raw_yolo"], det)
            accumulate(per_camera[cam]["post_tracker"], trk)
            per_camera[cam]["difference"].update(delta)
            if example_counts[cam] < examples:
                valid_gt, valid_pred, _, dm, _ = visual
                picture = draw_overlay(images[cam], valid_gt, valid_pred, dm,
                                       f"{cam} frame {fid:08d} | YOLO {imgsz} | IoU {match_iou}")
                folder = output_dir / "examples"
                folder.mkdir(exist_ok=True, parents=True)
                if not cv2.imwrite(str(folder / f"{cam}_{fid:08d}_size{imgsz}.jpg"), picture):
                    raise OSError("Could not save diagnostic preview")
                example_counts[cam] += 1
        if index == 1 or index % 10 == 0 or index == len(frame_ids):
            print(f"Resolution {imgsz}: {index}/{len(frame_ids)} annotated frame sets", flush=True)
    combined = {}
    for key in ("raw_yolo", "post_tracker"):
        total = new_totals()
        for cam in CAMERAS:
            accumulate(total, per_camera[cam][key])
        combined[key] = finish_totals(total)
    return {
        "imgsz": imgsz, "confidence_threshold": confidence, "nms_iou": nms_iou,
        "match_iou": match_iou, "four_camera_batch_inference_ms_p50": round(median(times), 2),
        "per_camera": {cam: {
            "raw_yolo": finish_totals(per_camera[cam]["raw_yolo"]),
            "post_tracker": finish_totals(per_camera[cam]["post_tracker"]),
            "difference": dict(per_camera[cam]["difference"])
        } for cam in CAMERAS},
        "overall": combined,
        "note": "Post-tracker boxes came from the SAVED original pipeline. They were not rerun at this imgsz."
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for cmd in ("audit", "compare"):
        p = sub.add_parser(cmd)
        p.add_argument("--frames", type=int, default=20)
        p.add_argument("--data-dir", type=Path, default=DATA)
        p.add_argument("--output-dir", type=Path, default=OUTPUT)
    cp = sub.choices["compare"]
    cp.add_argument("--sizes", default="640,960,1280")
    cp.add_argument("--conf", type=float, default=0.10)
    cp.add_argument("--nms-iou", type=float, default=0.7)
    cp.add_argument("--match-iou", type=float, default=0.5)
    cp.add_argument("--tracks", type=Path, default=OUT / "predictions.jsonl")
    cp.add_argument("--roi-json", type=Path, default=None)
    cp.add_argument("--examples", type=int, default=1)
    cp.add_argument("--threads", type=int, default=4)
    cp.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 400:
        parser.error("--frames must be between 1 and 400")
    frame_ids = available_frames(args.data_dir, args.frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = annotation_audit(args.data_dir, frame_ids)
    (args.output_dir / "coverage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.mode == "audit":
        print("Wrote", args.output_dir / "coverage_audit.json")
        print("Ground-truth box counts:", {cam: audit["per_camera"][cam]["gt_boxes"] for cam in CAMERAS})
        print("Geometry problems:", len(audit["geometry_problems"]))
        return
    if not 0 < args.conf <= 1 or not 0 < args.match_iou <= 1 or not 0 < args.nms_iou <= 1 or args.threads < 1 or args.examples < 0:
        parser.error("Invalid confidence, IoU, thread count or example count")
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    if not sizes or len(set(sizes)) != len(sizes) or any(s < 32 or s % 32 for s in sizes):
        parser.error("--sizes must be unique positive multiples of 32")
    all_tracks = read_predictions(args.tracks)
    missing = [fid for fid in frame_ids if fid not in all_tracks]
    if missing:
        raise ValueError(f"Saved tracks are not aligned to annotated frames; missing IDs: {missing[:5]}")
    roi = load_roi(args.roi_json)
    import torch
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    # Initialise YOLO, then explicitly restore CPU threading (Ultralytics can override it).
    import cv2
    first = cv2.imread(str(Path(args.data_dir) / "Image_subsets" / "C1" / f"{frame_ids[0]:08d}.png"))
    if first is None:
        raise ValueError("First evaluation image is unreadable")
    model.predict(source=[first], classes=[0], device="cpu", imgsz=sizes[0],
                  conf=args.conf, iou=args.nms_iou, verbose=False)
    torch.set_num_threads(args.threads)
    reports = {}
    for size in sizes:
        reports[str(size)] = evaluate_setting(model, args.data_dir, frame_ids, size, args.conf,
                                             args.nms_iou, args.match_iou, roi, all_tracks,
                                             args.output_dir, args.examples, args.threads)
    output = {
        "dataset": "WILDTRACK annotated PNG frames",
        "frame_sets": len(frame_ids),
        "camera_count": len(CAMERAS),
        "roi_status": "manual polygons, non-official" if roi is not None else "not filtered; annotated area of interest unverified",
        "box_precision_warning": "An unmatched box outside WILDTRACK's annotation area is not necessarily a false positive.",
        "comparison_warning": "The post-tracker baseline is from the previously saved Step 14 run at imgsz=640. Do not interpret it as tracking output at other resolutions.",
        "inference_timing_warning": "Timing is per four-camera batch, excludes decoding, tracking and ReID. Use a dedicated warmed benchmark for production latency.",
        "settings": reports
    }
    dest = args.output_dir / "comparison.json"
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved", dest)
    for size, rep in reports.items():
        print(f"YOLO {size}: raw GT recall={rep['overall']['raw_yolo']['recall']}, "
              f"batch P50={rep['four_camera_batch_inference_ms_p50']} ms")


if __name__ == "__main__":
    main()
