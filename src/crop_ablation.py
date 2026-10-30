"""Step 14.23: WILDTRACK crop audit and isolated OSNet context-padding ablation.

Diagnostic only: do not edit the production crop path or matcher.
Uses cached original features from Step 14.20 and *the same* OSNet model for
10%/20% padded boxes. GT labels are used only for audit/scoring, never encoding.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json
from .reid_diagnostics import DEFAULT_DIAG, read_features, score_pairs, sha256, normalize

DEFAULT_ORACLE = DEFAULT_OUT / "oracle_100"
DEFAULT_OUT_DIR = DEFAULT_OUT / "crop_ablation_100"
IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080
FOCUS_GT = {115, 116}
FOCUS_FRAMES = (405, 425, 445, 465, 485)
VERSION = 1


def box_padding(box, fraction, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Return a padded XYXY box clipped to image bounds (no class/ID-dependent edits)."""
    if not math.isfinite(float(fraction)) or not 0 <= fraction <= 0.5:
        raise ValueError("Padding fraction must be finite and within [0, 0.5]")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid original XYXY box or bounds")
    w, h = x2-x1, y2-y1
    return (max(0.0, x1-fraction*w), max(0.0, y1-fraction*h),
            min(float(width), x2+fraction*w), min(float(height), y2+fraction*h))


def overlap_fraction(a, b):
    """Intersection divided by area of a; measures other-person contamination risk."""
    ix = max(0., min(a[2], b[2])-max(a[0], b[0]))
    iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    return ix*iy / max(1., (a[2]-a[0])*(a[3]-a[1]))


def summarize_boxes(gt, fractions=(0.0, 0.1, 0.2)):
    """All GT crops in gt are already eligible (filtered by caller)."""
    result = {}
    for frac in fractions:
        n = touch = contam = 0
        by_cam = defaultdict(lambda: {"crops": 0, "touch_border": 0, "overlap_other_gt": 0})
        widths, heights = [], []
        for fid in sorted(gt):
            for cam in CAMERAS:
                objects = gt[fid].get(cam, [])
                padded = [box_padding(obj["xyxy"], frac) for obj in objects]
                for i, bounds in enumerate(padded):
                    n += 1
                    c = by_cam[cam]; c["crops"] += 1
                    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
                    widths.append(w); heights.append(h)
                    at_edge = bounds[0] <= 0 or bounds[1] <= 0 or bounds[2] >= IMAGE_WIDTH or bounds[3] >= IMAGE_HEIGHT
                    touch += at_edge; c["touch_border"] += int(at_edge)
                    overlaps = any(overlap_fraction(bounds, other["xyxy"]) > 0.0 for j, other in enumerate(objects) if j != i)
                    contam += overlaps; c["overlap_other_gt"] += int(overlaps)
        result[f"pad_{round(frac*100)}"] = {
            "padding_fraction": frac, "crops": n, "touch_border": touch,
            "overlap_other_gt": contam, "median_width_px": round(float(np.median(widths)), 1) if widths else None,
            "median_height_px": round(float(np.median(heights)), 1) if heights else None,
            "by_camera": dict(by_cam),
        }
    return result


def eligible_gt(by_frame):
    # Convert validated cached metadata into the source objects used for the audit.
    return {fid: {cam: [{"gt_id": pid, "xyxy": list(box)} for pid, box, _ in camera.get(cam, [])]
                  for cam in CAMERAS} for fid, camera in by_frame.items()}


def pair_cosines(by_frame, left=("C1", 116), right=("C2", 115), frames=FOCUS_FRAMES):
    result = []
    for fid in frames:
        frame = by_frame.get(fid, {})
        a = next((e for e in frame.get(left[0], []) if e[0] == left[1]), None)
        b = next((e for e in frame.get(right[0], []) if e[0] == right[1]), None)
        if a is not None and b is not None:
            result.append({"frame_id": fid, "camera_a": left[0], "gt_id_a": left[1],
                           "camera_b": right[0], "gt_id_b": right[1],
                           "cosine": round(float(np.dot(a[2], b[2])), 6)})
    return result


def create_contact_sheet(data_dir, by_frame, output, frames=FOCUS_FRAMES, cv2_module=None):
    """Save local GT labeled crops for the 115/116 failure case. Never shows identities automatically."""
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    cell_w, cell_h, header_h = 260, 300, 44
    cols = [("C1", 116), ("C2", 115), ("C1", 115), ("C2", 116)]
    canvas = np.full((len(frames)*(cell_h+header_h), len(cols)*cell_w, 3), 245, np.uint8)
    image_cache = {}
    written = 0
    for row, fid in enumerate(frames):
        for col, (cam, pid) in enumerate(cols):
            x, y = col*cell_w, row*(cell_h+header_h)
            label = f"GT person {pid} | {cam} | frame {fid}"
            cv2.putText(canvas, label, (x+4, y+26), cv2.FONT_HERSHEY_SIMPLEX, .42, (20,20,20), 1, cv2.LINE_AA)
            candidate = next((e for e in by_frame.get(fid, {}).get(cam, []) if e[0] == pid), None)
            if candidate is None:
                cv2.putText(canvas, "Not annotated / ineligible", (x+4, y+header_h+35),
                            cv2.FONT_HERSHEY_SIMPLEX, .42, (50,50,50), 1, cv2.LINE_AA)
                continue
            key = (fid, cam)
            if key not in image_cache:
                path = Path(data_dir) / "Image_subsets" / cam / f"{fid:08d}.png"
                image_cache[key] = cv2.imread(str(path))
                if image_cache[key] is None or image_cache[key].shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise FileNotFoundError(f"Missing WILDTRACK image or wrong dimensions: {path}")
            b = candidate[1]
            x1, y1 = int(b[0]), int(b[1]); x2, y2 = int(math.ceil(b[2])), int(math.ceil(b[3]))
            crop = image_cache[key][y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError("Empty audited crop")
            scale = min((cell_w-12)/crop.shape[1], (cell_h-12)/crop.shape[0])
            new_w = max(1, int(crop.shape[1]*scale)); new_h = max(1, int(crop.shape[0]*scale))
            thumb = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            ox, oy = x+(cell_w-new_w)//2, y+header_h+(cell_h-new_h)//2
            canvas[oy:oy+new_h, ox:ox+new_w] = thumb
            written += 1
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Failed to write contact sheet {output}")
    return written


def audit(data_dir, oracle_dir, features_dir, output_dir, make_sheet=True):
    fm, _, by_frame = read_features(features_dir, data_dir, oracle_dir)
    if len(fm["frame_ids"]) != 100:
        raise ValueError("This audit expects the full 100-frame Step 14.20 embedding set")
    gt = eligible_gt(by_frame)
    stats = summarize_boxes(gt)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source_sha256": fm["ground_truth_sha256"],
              "features_sha256": fm["features_sha256"],
              "checkpoint_sha256": fm["checkpoint_sha256"],
              "crop_geometry": stats,
              "hard_case": {"wrong_gt_pair": "115:116", "first_wrong_assignment_frame": 405,
                            "source": "Step 14.22 false-merge report; ground-truth labels for audit only",
                            "raw_same_frame_cosines": pair_cosines(by_frame)},
              "warnings": ["Padding can include background or other pedestrians: see overlap_other_gt counts.",
                           "Contact sheet uses GT labels for human review, not encoder input.",
                           "The original model's internal resize and normalization are unchanged."]}
    if make_sheet:
        report["contact_sheet_observations"] = create_contact_sheet(
            data_dir, by_frame, output_dir / "false_merge_115_116.png")
    dump_json(output_dir / "audit.json", report)
    return report


def rows_for_frame(by_frame, fid):
    return [(cam, [(pid, box, feature) for pid, box, feature in by_frame[fid].get(cam, [])])
            for cam in CAMERAS if by_frame[fid].get(cam)]


def encode_padding(by_frame, data_dir, encoder, fraction, cv2_module=None):
    """Preserve original crop metadata; only expand XYXY boxes *sent* to OSNet."""
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module
    feats, rows = [], []
    for fid in sorted(by_frame):
        requests, requests_rows = [], []
        for cam, entities in rows_for_frame(by_frame, fid):
            image_path = Path(data_dir) / "Image_subsets" / cam / f"{fid:08d}.png"
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise FileNotFoundError(f"Missing/invalid image: {image_path}")
            persons = [{"xyxy": box_padding(box, fraction), "confidence": 1.0} for _, box, _ in entities]
            requests.append((cam, fid, image, persons))
            requests_rows.append([(fid, CAMERAS.index(cam), pid, *box) for pid, box, _ in entities])
        if not requests:
            continue
        result = encoder.encode_batch(requests)
        if len(result) != len(requests):
            raise ValueError("Encoder batch count differs from request count")
        for batch, metadata in zip(result, requests_rows):
            if len(batch) != len(metadata):
                raise ValueError("Encoder returned missing crop embeddings")
            for vector, row in zip(batch, metadata):
                feats.append(normalize(vector))
                rows.append(row)
    if not feats:
        raise ValueError("No eligible crops were encoded")
    return np.stack(feats).astype(np.float32), np.asarray(rows, dtype=np.float64)


def _expected_rows(by_frame):
    return np.asarray([(fid, CAMERAS.index(cam), pid, *box)
                       for fid in sorted(by_frame) for cam in CAMERAS
                       for pid, box, _ in by_frame[fid].get(cam, [])], dtype=np.float64)


def validate_meta(feature_array, meta, expected):
    if feature_array.ndim != 2 or len(feature_array) != len(expected) or not feature_array.shape[1]:
        raise ValueError("Feature shape mismatch")
    if meta.shape != expected.shape or not np.array_equal(meta, expected):
        raise ValueError("Crop metadata no longer matches baseline exactly")
    if not np.isfinite(feature_array).all() or np.max(np.abs(np.linalg.norm(feature_array, axis=1)-1.0)) > 1e-3:
        raise ValueError("Invalid or non-normalized features")


def save_variant(output_dir, fraction, features, metadata, baseline_manifest, expected):
    """Write atomic variant and hash ledger; never overwrite baseline features."""
    validate_meta(features, metadata, expected)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = f"pad{int(round(fraction*100)):02d}"
    path = out / f"features_{suffix}.npz"
    with path.with_suffix(".npz.partial").open("wb") as f:
        np.savez_compressed(f, features=features, metadata=metadata)
    path.with_suffix(".npz.partial").replace(path)
    manifest = {"version": VERSION, "fraction": fraction, "features_file": path.name,
                "features_sha256": sha256(path), "baseline_features_sha256": baseline_manifest["features_sha256"],
                "ground_truth_sha256": baseline_manifest["ground_truth_sha256"],
                "checkpoint_sha256": baseline_manifest["checkpoint_sha256"],
                "crops": len(features), "feature_dim": features.shape[1],
                "method": "Same WILDTRACK rectified images; original boxes expanded geometrically; BoxMOT OSNet unchanged",
                "no_production_change": True}
    dump_json(out / f"manifest_{suffix}.json", manifest)
    return manifest


def load_variant(output_dir, fraction, baseline_manifest, expected):
    suffix = f"pad{int(round(fraction*100)):02d}"
    manpath = Path(output_dir) / f"manifest_{suffix}.json"
    m = json.loads(manpath.read_text(encoding="utf-8"))
    if m["version"] != VERSION or m["fraction"] != fraction or m["baseline_features_sha256"] != baseline_manifest["features_sha256"] or m["ground_truth_sha256"] != baseline_manifest["ground_truth_sha256"] or m["checkpoint_sha256"] != baseline_manifest["checkpoint_sha256"]:
        raise ValueError("Variant input/checkpoint mismatch")
    fp = Path(output_dir) / m["features_file"]
    if sha256(fp) != m["features_sha256"]:
        raise ValueError("Variant file hash mismatch")
    with np.load(fp, allow_pickle=False) as archive:
        features = np.array(archive["features"], dtype=np.float32)
        meta = np.array(archive["metadata"], dtype=np.float64)
    validate_meta(features, meta, expected)
    if len(features) != m["crops"] or features.shape[1] != m["feature_dim"]:
        raise ValueError("Variant dimensions do not match manifest")
    return features


def rebuild_by_frame(baseline_by_frame, features):
    expected_count = sum(len(baseline_by_frame[fid].get(cam, [])) for fid in baseline_by_frame for cam in CAMERAS)
    if len(features) != expected_count:
        raise ValueError("Feature count mismatch during reconstruction")
    i = 0
    result = {}
    for fid in sorted(baseline_by_frame):
        result[fid] = {}
        for cam in CAMERAS:
            entries = []
            for pid, box, _ in baseline_by_frame[fid].get(cam, []):
                entries.append((pid, box, features[i])); i += 1
            result[fid][cam] = entries
    if i != len(features):
        raise ValueError("Feature count mismatch during reconstruction")
    return result


def _experiment_runner(data_dir, oracle_dir, features_dir, output_dir, fractions, threads, affinity):
    from .identity_evaluation import _configure_execution
    from .reid import AppearanceEncoder, MODEL_WEIGHTS
    if not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(f"OSNet model weights unavailable: {MODEL_WEIGHTS}")
    baseline_manifest, _, by_frame = read_features(features_dir, data_dir, oracle_dir)
    if len(baseline_manifest["frame_ids"]) != 100:
        raise ValueError("Full 100-frame cached embeddings are required")
    if sha256(MODEL_WEIGHTS) != baseline_manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from cached baseline: stop")
    actual_affinity, actual_threads = _configure_execution(threads, affinity)
    encoder = AppearanceEncoder(backend="pytorch")
    expected = _expected_rows(by_frame)
    result = []
    for frac in fractions:
        start = time.perf_counter()
        feats, meta = encode_padding(by_frame, data_dir, encoder, frac)
        m = save_variant(output_dir, frac, feats, meta, baseline_manifest, expected)
        m["encode_seconds"] = round(time.perf_counter() - start, 3)
        m["threads"] = actual_threads; m["affinity"] = actual_affinity
        dump_json(Path(output_dir) / f"manifest_pad{int(round(frac*100)):02d}.json", m)
        print(f"Padding {frac:.0%}: encoded {len(feats)} crops in {m['encode_seconds']} s", flush=True)
        result.append(m)
    return result


def compare(data_dir, oracle_dir, features_dir, output_dir, fractions):
    baseline_manifest, _, base = read_features(features_dir, data_dir, oracle_dir)
    if len(baseline_manifest["frame_ids"]) != 100:
        raise ValueError("Full 100-frame cached embeddings are required")
    expected = _expected_rows(base)
    variants = [(0., base)]
    for frac in fractions:
        feat = load_variant(output_dir, frac, baseline_manifest, expected)
        variants.append((frac, rebuild_by_frame(base, feat)))
    summaries = {}
    rows = []
    for frac, frames in variants:
        key = f"pad{int(round(frac*100)):02d}"
        summary, _, _ = score_pairs(frames)
        summaries[key] = {"padding_fraction": frac, "roc_auc": summary["roc_auc"],
                          "genuine_p50": summary["genuine"]["p50"],
                          "impostor_p50": summary["impostor"]["p50"],
                          "camera_pairs": {pair: {"auc": stats["roc_auc"],
                                                   "top1": stats["retrieval"]["top1"],
                                                   "top5": stats["retrieval"]["top5"],
                                                   "eligible_queries": stats["retrieval"]["eligible_queries"]}
                                           for pair, stats in summary["camera_pairs"].items()},
                          "hard_case_wrong_115_116": pair_cosines(frames)}
        for pair, metrics in summaries[key]["camera_pairs"].items():
            rows.append({"variant": key, "camera_pair": pair, **metrics})
        print(f"{key}: cosine ROC-AUC {summary['roc_auc']} | C1-C2 Top1 {summaries[key]['camera_pairs']['C1-C2']['top1']}", flush=True)
    # Recalculate exact baseline to catch changed scoring code or mismatched cached features.
    prior = Path(features_dir) / "diagnostics.json"
    if prior.is_file():
        raw = json.loads(prior.read_text(encoding="utf-8"))["embedding_quality"]
        if summaries["pad00"]["roc_auc"] != raw["roc_auc"]:
            raise ValueError("Baseline cosine AUC differs from Step 14.20; comparison invalid")
        for pair in summaries["pad00"]["camera_pairs"]:
            if summaries["pad00"]["camera_pairs"][pair]["top1"] != raw["camera_pairs"][pair]["retrieval"]["top1"]:
                raise ValueError("Baseline retrieval differs from Step 14.20; comparison invalid")
    report = {"experiment": "Same-encoder box-padding ablation; no association changes",
              "base_features_sha256": baseline_manifest["features_sha256"],
              "gt_sha256": baseline_manifest["ground_truth_sha256"],
              "checkpoint_sha256": baseline_manifest["checkpoint_sha256"],
              "variants": summaries,
              "no_selected_winner": True,
              "limitations": ["All 100 annotated frame sets have already been studied. No blind generalization claim.",
                              "Retrieval uses same-instant candidate galleries, not the live matcher.",
                              "Repeated frames/identities are correlated: pair AUC and top-k are descriptive.",
                              "Padding may include other people or background; audit overlap before interpretation.",
                              "No second pretrained encoder, new dataset, live FPS or identity assignment was tested.",
                              "Keep the production crop path and matcher unchanged."]}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "comparison.json", report)
    with (out / "camera_pair_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "camera_pair", "auc", "top1", "top5", "eligible_queries"])
        writer.writeheader(); writer.writerows(rows)
    return report


def parse_fractions(raw):
    fractions = [float(x.strip()) for x in raw.split(",")]
    if not fractions or len(set(fractions)) != len(fractions) or any(not (0 < x <= .5) or not math.isfinite(x) or abs(x*100 - round(x*100)) > 1e-9 for x in fractions):
        raise ValueError("Supply unique, positive padding fractions with whole-percent values, e.g. 0.10,0.20")
    return fractions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run", "compare"):
        p = subs.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=DATA)
        p.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE)
        p.add_argument("--features-dir", type=Path, default=DEFAULT_DIAG)
        p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
        if name in ("run", "compare"):
            p.add_argument("--paddings", default="0.10,0.20")
        if name == "run":
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--affinity", default="all")
        if name == "audit":
            p.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.data_dir, args.oracle_dir, args.features_dir,
                       args.output_dir, make_sheet=not args.no_sheet)
        print("Audit saved", args.output_dir / "audit.json",
              "| wrong-pair same-frame scores", len(result["hard_case"]["raw_same_frame_cosines"]))
    elif args.command == "run":
        _experiment_runner(args.data_dir, args.oracle_dir, args.features_dir,
                           args.output_dir, parse_fractions(args.paddings), args.threads, args.affinity)
    else:
        result = compare(args.data_dir, args.oracle_dir, args.features_dir,
                         args.output_dir, parse_fractions(args.paddings))
        print("Comparison saved", args.output_dir / "comparison.json")
        for name, metrics in result["variants"].items():
            print(name, "AUC:", metrics["roc_auc"], "C1-C2 Top-1:", metrics["camera_pairs"]["C1-C2"]["top1"])


if __name__ == "__main__":
    main()
