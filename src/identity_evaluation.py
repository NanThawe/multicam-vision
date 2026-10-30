"""Step 14.19: Cross-camera ID diagnostics using WILDTRACK synchronized PNGs.

Modes:
  audit:        check annotated sample + count GT pairs and matcher eligibility.
  oracle:       run deployed OSNet/matcher on GT boxes and camera-local oracle tracks.
  compare:      score oracle and existing post-ByteTrack predictions on identical frames.

An oracle local track uses ground truth only to supply boxes and consistent
*within-camera* temporal track IDs. Shared GT identity is NEVER passed to
CrossCameraMatcher as a cross-camera association cue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import CAMERAS, DATA, OUT, available_frames, compute_metrics, match_frame, parse_gt

DEFAULT_OUT = OUT / "identity_evaluation"
MIN_CROP_WIDTH = 16
MIN_CROP_HEIGHT = 32
MIN_OBSERVATIONS = 3


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hash(gt_frames):
    """Hash exact ordered GT objects and sampled frame IDs, not just frame count."""
    canonical = json.dumps(gt_frames, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ground_truth(data_dir, frame_ids, width=1920, height=1080):
    return {int(fid): parse_gt(data_dir, int(fid), width=width, height=height)
            for fid in frame_ids}


def eligible(box):
    x1, y1, x2, y2 = box
    return x2 - x1 >= MIN_CROP_WIDTH and y2 - y1 >= MIN_CROP_HEIGHT


def pair_coverage(gt_frames):
    """Report a ceiling conditioned on observable crop size and warmup.

    Assumes each GT personID gives a perfect within-camera track and the matcher
    receives an eligible crop for every annotated appearance. This is a *GT
    opportunity count*, not a feasible estimate of deployed ReID recall.
    """
    observations = Counter()
    all_pairs = mature_pairs = crop_eligible_pairs = 0
    valid_boxes = total_boxes = 0
    per_pair = defaultdict(lambda: Counter())
    by_frame = []
    for fid, cameras in sorted(gt_frames.items()):
        current = {}
        for cam in CAMERAS:
            objects = cameras.get(cam, [])
            cmap = {}
            for obj in objects:
                pid = int(obj["gt_id"])
                total_boxes += 1
                if eligible(obj["xyxy"]):
                    valid_boxes += 1
                    observations[(cam, pid)] += 1
                    cmap[pid] = (True, observations[(cam, pid)] >= MIN_OBSERVATIONS)
                else:
                    cmap[pid] = (False, False)
            current[cam] = cmap
        for index, a in enumerate(CAMERAS):
            for b in CAMERAS[index + 1:]:
                key = f"{a}-{b}"
                common = set(current[a]) & set(current[b])
                all_pairs += len(common)
                per_pair[key]["all_gt_pairs"] += len(common)
                for pid in common:
                    va, ma = current[a][pid]
                    vb, mb = current[b][pid]
                    if va and vb:
                        crop_eligible_pairs += 1
                        per_pair[key]["crop_eligible_gt_pairs"] += 1
                    if ma and mb:
                        mature_pairs += 1
                        per_pair[key]["mature_eligible_gt_pairs"] += 1
        by_frame.append({"frame_id": fid,
                         "ground_truth_boxes": sum(len(cameras.get(cam, [])) for cam in CAMERAS)})
    return {
        "annotated_frame_sets": len(gt_frames),
        "gt_boxes": total_boxes,
        "gt_boxes_crop_eligible": valid_boxes,
        "true_gt_cross_camera_pairs": all_pairs,
        "crop_eligible_gt_pairs": crop_eligible_pairs,
        "mature_eligible_gt_pairs": mature_pairs,
        "mature_eligible_fraction": round(mature_pairs / all_pairs, 4) if all_pairs else None,
        "camera_pairs": {k: dict(v) for k, v in sorted(per_pair.items())},
        "notes": [
            "Mature means >=3 annotated eligible observations of a GT person in each camera.",
            "The fraction is an opportunity/coverage diagnostic, NOT an upper bound on matching precision or a guarantee of achievable recall.",
            "Any predicted cross-camera pair still requires appearance-based matching; GT IDs are only used in the evaluator.",
        ],
    }


def oracle_tracks(gt_by_camera, camera_id_map):
    """Convert GT boxes to camera-specific local IDs without exposing global GT IDs.

    A label may be numerically identical in distinct cameras; opaque local
    track IDs are independently allocated and only (camera, local ID) persists.
    """
    out = {}
    for cam in CAMERAS:
        tracks = []
        for obj in gt_by_camera.get(cam, []):
            if not eligible(obj["xyxy"]):
                continue
            pid = int(obj["gt_id"])
            if pid not in camera_id_map[cam]:
                camera_id_map[cam][pid] = len(camera_id_map[cam]) + 1
            tracks.append({"track_id": camera_id_map[cam][pid],
                           "xyxy": list(obj["xyxy"]), "confidence": 1.0})
        out[cam] = tracks
    return out


def load_predictions(path, frame_ids):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    ids = set(map(int, frame_ids))
    predictions = {int(fid): {} for fid in frame_ids}
    extras = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                fid, cam = int(item["frame_id"]), item["camera"]
                if cam not in CAMERAS:
                    raise ValueError(f"unknown camera {cam!r}")
                tracks = item["tracks"]
                if not isinstance(tracks, list):
                    raise ValueError("tracks must be a list")
                for t in tracks:
                    coords = t["xyxy"]
                    if (len(coords) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                                    for x in coords)):
                        raise ValueError("invalid track coordinates")
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid row in {path} at line {line_no}: {exc}") from exc
            if fid not in ids:
                extras += 1
                continue
            if cam in predictions[fid]:
                raise ValueError(f"Duplicate camera/frame row: {cam}/{fid} in {path}")
            predictions[fid][cam] = tracks
    missing = [(fid, cam) for fid in frame_ids for cam in CAMERAS
               if cam not in predictions[int(fid)]]
    if missing:
        raise ValueError(f"Missing {len(missing)} frame/camera prediction rows in {path}: {missing[:5]}")
    return predictions, extras


def pair_diagnostics(gt_frames, pred_frames, threshold=0.5):
    """Pair counts per camera pair using the same one-to-one IoU matching."""
    counters = defaultdict(Counter)
    for fid in sorted(gt_frames):
        truth, predictions = gt_frames[fid], pred_frames[fid]
        matched = {}
        for cam in CAMERAS:
            gt, pred = truth.get(cam, []), predictions.get(cam, [])
            matched[cam] = {p: gt[g]["gt_id"] for g, p in match_frame(gt, pred, threshold)}
        for idx, a in enumerate(CAMERAS):
            for b in CAMERAS[idx + 1:]:
                stats = counters[f"{a}-{b}"]
                stats["true_gt_pairs"] += len({x["gt_id"] for x in truth.get(a, [])} &
                                              {x["gt_id"] for x in truth.get(b, [])})
                for ai, ap in enumerate(predictions.get(a, [])):
                    if ap.get("global_id") is None:
                        continue
                    for bi, bp in enumerate(predictions.get(b, [])):
                        if ap["global_id"] != bp.get("global_id"):
                            continue
                        stats["predicted_pairs"] += 1
                        if matched[a].get(ai) is not None and matched[a].get(ai) == matched[b].get(bi):
                            stats["correct_pairs"] += 1
    out = {}
    for idx, a in enumerate(CAMERAS):
        for b in CAMERAS[idx + 1:]:
            key = f"{a}-{b}"
            c = counters[key]
            ng, npred, correct = c["true_gt_pairs"], c["predicted_pairs"], c["correct_pairs"]
            p = correct / npred if npred else None
            r = correct / ng if ng else None
            out[key] = {
                "true_gt_pairs": ng,
                "predicted_same_identity_pairs": npred,
                "correct_same_identity_pairs": correct,
                "false_same_identity_pairs": npred - correct,
                "pair_precision": round(p, 4) if p is not None else None,
                "pair_recall": round(r, 4) if r is not None else None,
                "pair_f1": round(2 * p * r / (p + r), 4)
                if p is not None and r is not None and p + r > 0 else None,
            }
    return out


def verify_prediction_alignment(gt, pred, label):
    if set(gt) != set(pred):
        raise ValueError(f"{label} frame IDs do not match selected GT IDs")
    for fid in gt:
        if set(pred[fid]) != set(CAMERAS):
            raise ValueError(f"{label} is missing cameras for frame {fid}")


def measure(gt_frames, predictions, threshold):
    verify_prediction_alignment(gt_frames, predictions, "predictions")
    return {"metrics": compute_metrics(gt_frames, predictions, threshold),
            "camera_pair_breakdown": pair_diagnostics(gt_frames, predictions, threshold)}


def _configure_execution(threads, affinity):
    if threads < 1:
        raise ValueError("--threads must be positive")
    import psutil
    process = psutil.Process()
    if affinity.lower() != "all":
        cpus = [int(cpu.strip()) for cpu in affinity.split(",")]
        if not cpus or len(set(cpus)) != len(cpus):
            raise ValueError("Affinity needs a nonempty unique CPU list")
        allowed = set(process.cpu_affinity())
        if not set(cpus).issubset(allowed):
            raise ValueError(f"Requested CPUs {cpus} not all allowed: {sorted(allowed)}")
        process.cpu_affinity(cpus)
    actual_affinity = process.cpu_affinity()
    import torch
    torch.set_num_threads(threads)
    actual_threads = torch.get_num_threads()
    if actual_threads != threads:
        raise RuntimeError(f"Expected {threads} PyTorch threads, got {actual_threads}")
    return actual_affinity, actual_threads


def run_oracle(data_dir, out_dir, frame_ids, threads=4, affinity="all"):
    import cv2
    actual_affinity, actual_threads = _configure_execution(threads, affinity)
    from .reid import AppearanceEncoder, CrossCameraMatcher, MATCH_THRESHOLD, MATCH_MARGIN

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_frames = ground_truth(data_dir, frame_ids)
    matcher = CrossCameraMatcher(
        AppearanceEncoder(backend="pytorch"), min_observations=MIN_OBSERVATIONS,
        threshold=MATCH_THRESHOLD, margin=MATCH_MARGIN,
        refresh_every=5, batch_across_cameras=True,
    )
    local_ids = {cam: {} for cam in CAMERAS}
    oracle = {}
    start = time.perf_counter()
    prediction_path = out_dir / "oracle_predictions.jsonl"
    decisions_path = out_dir / "oracle_decisions.jsonl"
    # All four synchronized PNG images are used at each annotated instant.
    with prediction_path.open("w", encoding="utf-8") as output, \
            decisions_path.open("w", encoding="utf-8") as decfile:
        for index, fid in enumerate(frame_ids, start=1):
            images = {}
            for cam in CAMERAS:
                image_path = Path(data_dir) / "Image_subsets" / cam / f"{fid:08d}.png"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"Cannot read image {image_path}")
                if image.shape[:2] != (1080, 1920):
                    raise ValueError(f"Expected 1920x1080 rectified WILDTRACK PNG: {image_path}")
                images[cam] = image
            local = oracle_tracks(gt_frames[fid], local_ids)
            annotated, decisions = matcher.update(fid, images, local)
            oracle[fid] = annotated
            for cam in CAMERAS:
                output.write(json.dumps({"frame_id": fid, "camera": cam,
                                         "tracks": annotated[cam]}) + "\n")
            for decision in decisions:
                decfile.write(json.dumps(decision) + "\n")
            print(f"[{index}/{len(frame_ids)}] annotated frame {fid} | "
                  f"oracle-local crops {sum(len(x) for x in local.values())}", flush=True)
    elapsed_s = time.perf_counter() - start
    manifest = {
        "basis": "synchronized rectified WILDTRACK PNG / annotations_positions",
        "frame_ids": list(map(int, frame_ids)),
        "gt_sha256": source_hash(gt_frames),
        "model": "deployed BoxMOT PyTorch OSNet checkpoint and CrossCameraMatcher",
        "encoder_refresh_every": 5,
        "batch_across_cameras": True,
        "min_observations": MIN_OBSERVATIONS,
        "match_threshold": MATCH_THRESHOLD,
        "match_margin": MATCH_MARGIN,
        "local_tracks": "camera-local opaque IDs from GT temporal identities; GT boxes; NOT cross-camera GT IDs",
        "execution": {"torch_threads": actual_threads, "affinity": actual_affinity,
                      "elapsed_wall_s": round(elapsed_s, 3)},
    }
    dump_json(out_dir / "oracle_manifest.json", manifest)
    result = measure(gt_frames, oracle, threshold=0.5)
    result["gt_opportunity"] = pair_coverage(gt_frames)
    result["assumptions"] = [
        "Perfect annotated bounding boxes, within-camera track IDs, and temporal continuity are supplied to the matcher.",
        "The matcher uses pretrained appearance features; cross-camera GT IDs are NOT used to associate cameras.",
        "This is NOT pure encoder retrieval accuracy: association policy, threshold, margin, warmup, and cache affect outcomes.",
        "Annotated PNGs are at 2 FPS; live RTSP is unsynchronized 5 FPS and cannot use these annotations directly.",
        "This is not official IDF1/HOTA; crop-box and ROI limitations still apply.",
    ]
    dump_json(out_dir / "oracle_metrics.json", result)
    return result


def compare(data_dir, oracle_dir, pipeline_path, out_dir, threshold=0.5):
    oracle_dir, out_dir = Path(oracle_dir), Path(out_dir)
    manifest = json.loads((oracle_dir / "oracle_manifest.json").read_text(encoding="utf-8"))
    frame_ids = manifest["frame_ids"]
    if len(frame_ids) != len(set(frame_ids)) or not frame_ids:
        raise ValueError("Oracle manifest frame IDs are invalid")
    available = set(available_frames(data_dir, None))
    if not set(frame_ids).issubset(available):
        raise ValueError("Annotated frames used by oracle are missing from data directory")
    gt_frames = ground_truth(data_dir, frame_ids)
    if source_hash(gt_frames) != manifest["gt_sha256"]:
        raise ValueError("Dataset/annotation fingerprint changed since oracle run")
    oracle_predictions, oracle_extras = load_predictions(
        oracle_dir / "oracle_predictions.jsonl", frame_ids)
    if oracle_extras:
        raise ValueError("Oracle predictions have extra frame IDs")
    pipeline_predictions, pipeline_extras = load_predictions(pipeline_path, frame_ids)
    o = measure(gt_frames, oracle_predictions, threshold)
    p = measure(gt_frames, pipeline_predictions, threshold)
    om, pm = o["metrics"]["cross_camera"], p["metrics"]["cross_camera"]
    if om["true_gt_pairs"] != pm["true_gt_pairs"]:
        raise AssertionError("Oracle and pipeline were scored against different GT cross-camera pairs")
    coverage = pair_coverage(gt_frames)
    result = {
        "evaluation_basis": "same labeled WILDTRACK frame IDs, same IoU and same pairwise scorer",
        "frame_ids": frame_ids,
        "gt_sha256": manifest["gt_sha256"],
        "iou_threshold": threshold,
        "ground_truth_opportunities": coverage,
        "oracle_local_tracks": o,
        "full_pipeline": p,
        "headline": {
            "gt_cross_camera_pairs": om["true_gt_pairs"],
            "oracle_correct_pairs": om["correct_same_identity_pairs"],
            "pipeline_correct_pairs": pm["correct_same_identity_pairs"],
            "oracle_pair_precision": om["pairwise_precision"],
            "oracle_pair_recall": om["pairwise_recall"],
            "pipeline_pair_precision": pm["pairwise_precision"],
            "pipeline_pair_recall": pm["pairwise_recall"],
        },
        "warnings": [
            "Oracle supplies GT boxes and perfect camera-local tracks; this is an upper-input-quality comparison, NOT a bound on deployed accuracy.",
            "Do not subtract recalls and call the difference a pure detector/tracker effect: matcher state and associations also change.",
            "Original post-ByteTrack predictions must come from the same annotated PNGs, NOT RTSP/raw MP4.",
            "Precision is conditional on the dataset annotation coverage/ROI; ROI mask has not been verified.",
            "Do not tune association thresholds on the exact frames used for final reporting.",
            "These pairwise scores are diagnostics; not official global IDF1 or HOTA.",
        ],
        "pipeline_extraneous_file_rows_ignored": pipeline_extras,
    }
    dump_json(out_dir / "comparison.json", result)
    with (out_dir / "camera_pair_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "camera_pair", "true_gt_pairs", "predicted_pairs", "correct_pairs",
                    "false_pairs", "pair_precision", "pair_recall", "pair_f1"])
        for label, row in (("oracle_local_tracks", o), ("full_pipeline", p)):
            for pair, metrics in sorted(row["camera_pair_breakdown"].items()):
                w.writerow([label, pair, metrics["true_gt_pairs"],
                            metrics["predicted_same_identity_pairs"],
                            metrics["correct_same_identity_pairs"],
                            metrics["false_same_identity_pairs"], metrics["pair_precision"],
                            metrics["pair_recall"], metrics["pair_f1"]])
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    for name in ("audit", "oracle"):
        sub = subs.add_parser(name)
        sub.add_argument("--data-dir", type=Path, default=DATA)
        sub.add_argument("--frames", type=int, default=20)
        sub.add_argument("--output-dir", type=Path, default=DEFAULT_OUT / name)
        if name == "oracle":
            sub.add_argument("--threads", type=int, default=4)
            sub.add_argument("--affinity", type=str, default="all")
    cp = subs.add_parser("compare")
    cp.add_argument("--data-dir", type=Path, default=DATA)
    cp.add_argument("--oracle-dir", type=Path, default=DEFAULT_OUT / "oracle")
    cp.add_argument("--pipeline-predictions", type=Path, default=OUT / "predictions.jsonl")
    cp.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    cp.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.command in ("audit", "oracle"):
        if not 1 <= args.frames <= 400:
            ap.error("--frames must be 1..400")
        ids = available_frames(args.data_dir, args.frames)
        if len(ids) != args.frames:
            ap.error(f"Only {len(ids)} complete annotated frame sets found; requested {args.frames}")
        if args.command == "audit":
            gt = ground_truth(args.data_dir, ids)
            result = {"frame_ids": ids, "gt_sha256": source_hash(gt),
                      "opportunity": pair_coverage(gt)}
            dump_json(args.output_dir / "audit.json", result)
            print(json.dumps(result["opportunity"], indent=2))
        else:
            result = run_oracle(args.data_dir, args.output_dir, ids,
                                threads=args.threads, affinity=args.affinity)
            print("Oracle cross-camera:", json.dumps(result["metrics"]["cross_camera"], indent=2))
    else:
        if not 0 < args.iou <= 1:
            ap.error("--iou must be in (0, 1]")
        result = compare(args.data_dir, args.oracle_dir, args.pipeline_predictions,
                         args.output_dir, args.iou)
        print("Comparison:", json.dumps(result["headline"], indent=2))
        print("Saved", args.output_dir / "comparison.json")


if __name__ == "__main__":
    main()
