"""Step 14.6: Compare ByteTrack confirmed, provisional and early-promotion policies.

Run from project root:
    python -m src.confirmation_policy --frames 100

Experiment ONLY. Does not modify src/tracker.py, ReID, events or RTSP.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import identity_f1, match_frame, parse_gt
from .track_ablation import CAMERAS, DATA, OUT, SETTINGS, load_images, tracker_config
from .track_confirmation import (build_update, confirmed_from_rows,
                                 same_frame_unconfirmed, validate_cache)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "evaluation" / "confirmation_policy"
POLICIES = ("confirmed_only", "provisional_visibility", "early_promotion")


def normalise_confirmed(rows):
    """Public confirmed observations with stable local ID and explicit status."""
    return [{"xyxy": p["xyxy"], "confidence": p["confidence"],
             "track_id": p["track_id"], "status": "confirmed"}
            for p in confirmed_from_rows(rows)]


def provisional_candidates(tracker):
    """Same-frame unconfirmed candidates; do NOT expose an identity to consumers."""
    return [{"xyxy": p["xyxy"], "confidence": p["confidence"],
             "track_id": None, "status": "provisional"}
            for p in same_frame_unconfirmed(tracker)]


def promote_current_candidates(tracker):
    """Experimental internal state mutation: early-activate new pending tracks.

    This alters ByteTrack's FUTURE matching behaviour. Never use in production
    without an independent safety and identity evaluation. No library files change.
    """
    candidates = []
    for track in getattr(tracker, "tracked_stracks", []):
        if getattr(track, "is_activated", False):
            continue
        if getattr(track, "frame_id", None) != getattr(tracker, "frame_id", None):
            continue
        track.is_activated = True
        candidates.append({
            "xyxy": [float(v) for v in track.xyxy],
            "confidence": float(track.score),
            "track_id": int(track.track_id),
            "status": "experimental_early_confirmed",
        })
    return candidates


def score_observations(gt, confirmed, emitted):
    """Compare one-to-one box matching; preserve ID-only scores separately."""
    normal = match_frame(gt, confirmed, threshold=0.5)
    all_matches = match_frame(gt, emitted, threshold=0.5)
    # Provisional boxes can displace confirmed boxes during joint IoU matching.
    # Preserve baseline identity matches, not the rematched provisional assignment.
    has_provisional = any(x["status"] == "provisional" for x in emitted)
    eligible = normal if has_provisional else all_matches
    return {
        "gt_boxes": len(gt),
        "confirmed_boxes": len(confirmed),
        "emitted_boxes": len(emitted),
        "provisional_boxes": sum(x["status"] == "provisional" for x in emitted),
        "early_promoted_boxes": sum(x["status"] == "experimental_early_confirmed" for x in emitted),
        "confirmed_matches": len(normal),
        "matched_boxes": len(all_matches),
        "extra_matches_beyond_confirmed": len(all_matches) - len(normal),
        "unmatched_boxes_roi_unverified": len(emitted) - len(all_matches),
        "provisional_matches": sum(emitted[pi]["status"] == "provisional" for _, pi in all_matches),
    }, eligible


def run_pass(records, data_dir, early_promotion=False, tracker_factory=None,
             image_loader=load_images, gt_loader=None, updater=build_update):
    """One independent 4-camera tracker pass. Normal pass scores A and B together."""
    if tracker_factory is None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        tracker_factory = lambda: BYTETracker(tracker_config(SETTINGS["baseline"]))
    trackers = {cam: tracker_factory() for cam in CAMERAS}
    if gt_loader is None:
        gt_loader = lambda fid, img: parse_gt(data_dir, fid, width=img.shape[1], height=img.shape[0])
    stats_rows, preview_rows = [], []
    for record in records:
        fid = int(record["frame_id"])
        images = image_loader(data_dir, fid)
        truth = gt_loader(fid, images[CAMERAS[0]])
        for cam in CAMERAS:
            tracker = trackers[cam]
            detections = record["detections"][cam]
            rows = updater(tracker, images[cam], detections)
            confirmed = normalise_confirmed(rows)
            if early_promotion:
                promoted = promote_current_candidates(tracker)
                choices = [("early_promotion", confirmed + promoted)]
            else:
                pending = provisional_candidates(tracker)
                choices = [("confirmed_only", confirmed),
                           ("provisional_visibility", confirmed + pending)]
            for policy, emitted in choices:
                per_frame, identity_matches = score_observations(truth[cam], confirmed, emitted)
                stats_rows.append({**per_frame, "policy": policy, "frame_id": fid,
                                   "camera": cam, "identity_matches": [
                                       (int(truth[cam][gi]["gt_id"]), int(emitted[pi]["track_id"]))
                                       for gi, pi in identity_matches
                                   ]})
                preview_rows.append({"policy": policy, "frame_id": fid, "camera": cam,
                                     "observations": emitted})
    return stats_rows, preview_rows


def summarise_policy(rows):
    """Scores and descriptive ID churn, computed separately within each camera."""
    if not rows:
        raise ValueError("Cannot summarise an empty policy")
    camera_scores = {}
    for cam in CAMERAS:
        subset = [r for r in rows if r["camera"] == cam]
        totals = Counter()
        pair_counts, id_observation_counts = Counter(), Counter()
        last_id_by_person = {}
        changes = 0
        for r in subset:
            for name in ("gt_boxes", "confirmed_boxes", "emitted_boxes", "provisional_boxes",
                         "early_promoted_boxes", "confirmed_matches", "matched_boxes",
                         "extra_matches_beyond_confirmed", "unmatched_boxes_roi_unverified",
                         "provisional_matches"):
                totals[name] += r[name]
            for gt_id, local_id in r["identity_matches"]:
                pair_counts[(gt_id, local_id)] += 1
                id_observation_counts[local_id] += 1
                if gt_id in last_id_by_person and last_id_by_person[gt_id] != local_id:
                    changes += 1
                last_id_by_person[gt_id] = local_id
        gt_total = totals["gt_boxes"]
        output_total = totals["emitted_boxes"]
        identity_samples = sum(id_observation_counts.values())
        camera_scores[cam] = {
            **dict(totals),
            "recall": round(totals["matched_boxes"] / gt_total, 4) if gt_total else None,
            "precision_proxy_unverified_roi": round(totals["matched_boxes"] / output_total, 4)
            if output_total else None,
            "observed_local_id_changes_diagnostic": changes,
            "identity_matched_observations": identity_samples,
            "identity_f1_diagnostic": identity_f1(pair_counts, gt_total,
                                                   sum(r["emitted_boxes"] - r["provisional_boxes"]
                                                       for r in subset)),
            "distinct_matched_local_ids": len(id_observation_counts),
            "single_matched_observation_ids": sum(n == 1 for n in id_observation_counts.values()),
        }
    totals = Counter()
    for cam in CAMERAS:
        s = camera_scores[cam]
        for k in ("gt_boxes", "confirmed_boxes", "emitted_boxes", "provisional_boxes",
                  "early_promoted_boxes", "confirmed_matches", "matched_boxes",
                  "extra_matches_beyond_confirmed", "unmatched_boxes_roi_unverified",
                  "provisional_matches", "observed_local_id_changes_diagnostic",
                  "identity_matched_observations", "distinct_matched_local_ids",
                  "single_matched_observation_ids"):
            totals[k] += s[k]
    overall = {
        **dict(totals),
        "recall": round(totals["matched_boxes"] / totals["gt_boxes"], 4)
        if totals["gt_boxes"] else None,
        "precision_proxy_unverified_roi": round(totals["matched_boxes"] / totals["emitted_boxes"], 4)
        if totals["emitted_boxes"] else None,
    }
    return {"overall": overall, "per_camera": camera_scores}


def compare_policies(records, data_dir, **kwargs):
    validate_cache(records)
    regular, regular_preview = run_pass(records, data_dir, False, **kwargs)
    early, early_preview = run_pass(records, data_dir, True, **kwargs)
    rows = regular + early
    summaries = {p: summarise_policy([r for r in rows if r["policy"] == p]) for p in POLICIES}
    base = summaries["confirmed_only"]["overall"]
    # A and B MUST share unchanged tracker state and identical confirmed matches.
    visibility = summaries["provisional_visibility"]["overall"]
    if base["confirmed_matches"] != visibility["confirmed_matches"]:
        raise AssertionError("Visibility policy unexpectedly changed baseline confirmed matches")
    report = {
        "basis": "Same cached 640px YOLO boxes, 2 FPS annotated WILDTRACK images; IoU 0.5",
        "frame_sets": len(records), "cameras": list(CAMERAS),
        "policies": summaries,
        "comparisons": {
            "visibility_extra_matched_gt": visibility["matched_boxes"] - base["matched_boxes"],
            "visibility_recall_gain_pp": round(100 * (visibility["recall"] - base["recall"]), 2),
            "early_recall_gain_pp": round(100 * (summaries["early_promotion"]["overall"]["recall"]
                                               - base["recall"]), 2),
        },
        "limitations": [
            "Provisional observations have track_id=null in public preview; not eligible for ReID or events.",
            "Early promotion mutates a separate experimental ByteTrack state; never enabled in production.",
            "Provisional visibility increases box coverage, not validated persistent identity quality.",
            "Unmatched boxes are not verified false positives; WILDTRACK evaluation ROI mask is absent.",
            "Observed local ID changes and identity F1 are diagnostics, not official IDSW/IDF1/HOTA.",
            "Evaluated on annotated PNGs at 2 FPS, not deployment RTSP at 5 FPS.",
        ],
    }
    return report, rows, regular_preview + early_preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if not 2 <= args.frames <= 400:
        parser.error("--frames must be between 2 and 400")
    cache = args.cache or (OUT / f"frames_{args.frames}" / "detections.jsonl")
    if not cache.is_file():
        parser.error(f"YOLO cache missing: {cache}. Run python -m src.track_ablation --frames {args.frames}")
    records = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(records) != args.frames:
        parser.error(f"Found {len(records)} frame sets in {cache}; expected {args.frames}")
    manifest_path = cache.parent / "cache_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("imgsz") != 640 or manifest.get("confidence") != 0.10
                or manifest.get("frame_ids") != [int(r["frame_id"]) for r in records]
                or manifest.get("cameras") != list(CAMERAS)):
            parser.error("Cache manifest does not match the expected 640px, conf=0.10, four-camera experiment")
    report, rows, previews = compare_policies(records, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        names = ["policy", "frame_id", "camera", "gt_boxes", "confirmed_boxes", "emitted_boxes",
                 "provisional_boxes", "early_promoted_boxes", "confirmed_matches", "matched_boxes",
                 "extra_matches_beyond_confirmed", "provisional_matches",
                 "unmatched_boxes_roi_unverified"]
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in names})
    with (args.output_dir / "policy_preview.jsonl").open("w", encoding="utf-8") as f:
        for preview in previews:
            f.write(json.dumps(preview) + "\n")
    for policy in POLICIES:
        s = report["policies"][policy]["overall"]
        print(f"{policy}: matched={s['matched_boxes']}, recall={s['recall']:.2%}, "
              f"unmatched_boxes={s['unmatched_boxes_roi_unverified']}, "
              f"observed_id_changes={s['observed_local_id_changes_diagnostic']}")
    print(f"Saved results in {args.output_dir}")


if __name__ == "__main__":
    main()
