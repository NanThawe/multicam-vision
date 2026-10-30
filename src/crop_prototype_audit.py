"""Step 14.24: geometry validation and experimental ReID prototype forensics.

Diagnostic only. Replays the *undeployed* Step 14.21 camera-balanced matcher,
whose false merge was established in Step 14.22. It never modifies production.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .association_ablation import Policy, build_tracks, partition_ids, score_subset
from .crop_ablation import IMAGE_HEIGHT, IMAGE_WIDTH, DEFAULT_OUT_DIR as PREVIOUS_CROP_OUT, box_padding, eligible_gt, pair_cosines
from .false_merge_forensics import (TracedMatcher, false_pair_timeline,
                                   local_label_map, reconstruct_cases)
from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json
from .reid_diagnostics import DEFAULT_DIAG, FrozenEncoder, read_features

DEFAULT_ORACLE = DEFAULT_OUT / "oracle_100"
DEFAULT_FORENSICS = DEFAULT_OUT / "false_merge_forensics_100"
DEFAULT_OUTPUT = DEFAULT_OUT / "crop_prototype_audit_100"


def _rounded(x):
    return round(float(x), 6) if x is not None else None


def _cosine(a, b):
    return _rounded(np.dot(a, b))


def _edges(box, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, tol=1e-6):
    x1, y1, x2, y2 = map(float, box)
    return {"left": x1 <= tol, "top": y1 <= tol,
            "right": x2 >= width - tol, "bottom": y2 >= height - tol}


def _percentile(values, q):
    return round(float(np.percentile(values, q)), 3) if values else None


def border_geometry(gt, paddings=(0.1, 0.2), width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Separate original clipping from *new* padding-induced border contact.

    gt is keyed by annotated frame and camera; each box must already be eligible.
    Doesn't assume touching image border implies incorrect coordinates.
    """
    results = {}
    samples = []
    for cam in CAMERAS:
        records = []
        for fid in sorted(gt):
            for ob in gt[fid].get(cam, []):
                box = tuple(map(float, ob["xyxy"]))
                # Fail on impossible boxes rather than mischaracterise clipping.
                box_padding(box, 0.0, width, height)
                original = _edges(box, width, height)
                margins = {"left": box[0], "top": box[1],
                           "right": width - box[2], "bottom": height - box[3]}
                records.append((int(fid), int(ob["gt_id"]), box, original, margins))
        original_counts = Counter(edge for _, _, _, e, _ in records for edge, yes in e.items() if yes)
        original_any = sum(any(e.values()) for _, _, _, e, _ in records)
        report = {"crops": len(records), "original_touch_any": original_any,
                  "original_touch_by_edge": dict(original_counts),
                  "original_margin_px": {edge: {"p10": _percentile([r[4][edge] for r in records], 10),
                                                "p50": _percentile([r[4][edge] for r in records], 50)}
                                         for edge in ("left", "top", "right", "bottom")},
                  "variants": {}}
        for fraction in paddings:
            touch = Counter(); newly = Counter(); any_touch = newly_any = 0
            for fid, pid, box, old, margins in records:
                new = _edges(box_padding(box, fraction, width, height), width, height)
                any_touch += int(any(new.values()))
                is_new = any(new[e] and not old[e] for e in new)
                newly_any += int(is_new)
                for edge in new:
                    touch[edge] += int(new[edge]); newly[edge] += int(new[edge] and not old[edge])
                if is_new and sum(x["camera"] == cam for x in samples) < 20:
                    samples.append({"frame_id": fid, "camera": cam, "gt_id": pid,
                                    "padding_fraction": fraction,
                                    "new_touch_edges": ",".join(e for e in new if new[e] and not old[e]),
                                    "old_xyxy": ",".join(str(round(v, 2)) for v in box),
                                    "nearest_margin_px": round(min(margins.values()), 2)})
            report["variants"][f"pad_{int(round(fraction * 100))}"] = {
                "touch_any": any_touch, "new_touch_any": newly_any,
                "touch_by_edge": dict(touch), "new_touch_by_edge": dict(newly)}
        results[cam] = report
    return results, samples


def inspect_raw_annotations(data_dir, gt, image_reader=None):
    """Verify that cached box coordinates equal clamped raw boxes, with real image size.

    Checks original raw annotation coverage, the viewNum convention, and image
    dimensions. Raw source boxes can legitimately extend outside the image.
    """
    if image_reader is None:
        import cv2
        image_reader = cv2.imread
    data_dir = Path(data_dir)
    if not gt:
        raise ValueError("No annotated frames")
    first = next(iter(sorted(gt)))
    dimensions = {}
    for cam in CAMERAS:
        path = data_dir / "Image_subsets" / cam / f"{first:08d}.png"
        image = image_reader(str(path))
        if image is None:
            raise FileNotFoundError(f"Missing rectified image: {path}")
        height, width = image.shape[:2]
        dimensions[cam] = [int(width), int(height)]
        if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(f"Wrong image dimensions for {cam}: {width}x{height}")
    raw_clip = defaultdict(Counter)
    source_counts = Counter()
    for fid in sorted(gt):
        path = data_dir / "annotations_positions" / f"{fid:08d}.json"
        source = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(source, list):
            raise ValueError(f"Invalid source annotation format: {path}")
        per_camera = {cam: {} for cam in CAMERAS}
        for person in source:
            pid = int(person["personID"])
            for view in person["views"]:
                cam = f"C{int(view['viewNum']) + 1}"
                if cam not in per_camera:
                    continue
                raw = [float(view[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                if raw == [-1.0] * 4:
                    continue
                x1, y1, x2, y2 = raw
                clamped = (max(0., x1), max(0., y1),
                           min(float(IMAGE_WIDTH), x2), min(float(IMAGE_HEIGHT), y2))
                if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                    continue
                if pid in per_camera[cam]:
                    raise ValueError(f"Duplicate person {pid}/{cam}/{fid}")
                per_camera[cam][pid] = clamped
                raw_clip[cam]["all_valid_raw_boxes"] += 1
                for key, yes in {"left": x1 < 0., "top": y1 < 0.,
                                 "right": x2 > IMAGE_WIDTH, "bottom": y2 > IMAGE_HEIGHT}.items():
                    raw_clip[cam][f"source_outside_{key}"] += int(yes)
        for cam in CAMERAS:
            for obj in gt[fid].get(cam, []):
                pid = obj["gt_id"]
                if pid not in per_camera[cam]:
                    raise ValueError(f"Cached crop not in source GT: {cam} frame {fid} id {pid}")
                if not np.allclose(per_camera[cam][pid], obj["xyxy"], atol=1e-6, rtol=0):
                    raise ValueError(f"Cached vs raw-clamped box mismatch: {cam} frame {fid} id {pid}")
                source_counts[cam] += 1
    return {"sample_image_dimensions_wh": dimensions,
            "eligible_box_verified_count": dict(source_counts),
            "raw_annotation_box_audit": {cam: dict(raw_clip[cam]) for cam in CAMERAS},
            "note": "Checks source-to-cached coordinate agreement; it cannot validate the original human annotation's visual accuracy."}


class PrototypeHistoryMatcher(TracedMatcher):
    """Observe *unchanged* experimental decisions before and after feature updates."""
    def __init__(self, policy, track_gt, focus_gid):
        super().__init__(policy, track_gt)
        self.focus_gid = int(focus_gid)
        self.history = []

    def _snapshot(self, gid, cam, action, track_id, feature, before, after, extra=None):
        if gid != self.focus_gid:
            return
        ident = self.globals[gid]
        origins = {c: self.track_gt[(c, tid)] for c, tid in ident["cameras"].items()}
        row = {"frame_id": self.fid, "batch_index": self.batch_index,
               "action": action, "global_id": gid, "camera": cam,
               "track_id": track_id,
               "gt_person_audit_only": self.track_gt.get((cam, track_id)),
               "prior_to_update_cosine": _cosine(before, feature) if before is not None else None,
               "prototype_change_one_minus_cosine": _rounded(1 - float(np.dot(before, after)))
               if before is not None else None,
               "origin_cameras": ",".join(sorted(origins)),
               "origin_gt_persons_audit_only": ",".join(str(origins[c]) for c in sorted(origins)),
               "distinct_origin_gt_people": len(set(origins.values())),
               "prototype_cameras": len(ident["per_camera"]),
               "prototype_norm": _rounded(np.linalg.norm(after))}
        if extra:
            row.update(extra)
        self.history.append(row)

    def _record_identity(self, gid, cam, person, state):
        orig = self.globals[gid]
        prior = orig["feature"].copy()
        prior_cams = dict(orig["cameras"])
        track_id = int(person["track_id"])
        # This audit does not affect the association result.
        super()._record_identity(gid, cam, person, state)
        self._snapshot(gid, cam, "attach", track_id, state["feature"], prior,
                       self.globals[gid]["feature"],
                       {"prior_cameras": ",".join(sorted(prior_cams)),
                        "new_cross_camera_attachment": bool(prior_cams and cam not in prior_cams)})

    def _update_global(self, gid, cam, state):
        before = self.globals[gid]["feature"].copy()
        tid = self.globals[gid]["cameras"].get(cam)
        super()._update_global(gid, cam, state)
        self._snapshot(gid, cam, "update", int(tid), state["feature"], before,
                       self.globals[gid]["feature"],
                       {"encoded_observation_count": state["count"],
                        "feature_last_encoded_batch": state["last_encoded"],
                        "reuses_cached_feature": state["last_encoded"] != self.batch_index})


def replay_history(by_frame, policy, focus_gid):
    tracks, labels = build_tracks(by_frame)
    matcher = PrototypeHistoryMatcher(policy, local_label_map(tracks, labels), focus_gid)
    features = FrozenEncoder(by_frame).lookup
    predictions = {}
    for fid, cams in tracks.items():
        predictions[fid] = matcher.update(fid, cams, features)
    return matcher, predictions, labels


def trace_summary(history, first_wrong_frame, focus_gid):
    before = [h for h in history if h["frame_id"] < first_wrong_frame]
    at = [h for h in history if h["frame_id"] == first_wrong_frame]
    first_contaminated = next((x for x in history if x["distinct_origin_gt_people"] > 1), None)
    return {"focus_global_id": focus_gid, "first_wrong_assignment_frame": first_wrong_frame,
            "events_before_wrong_merge": len(before), "events_at_wrong_frame": len(at),
            "first_heterogeneous_prototype_event": first_contaminated,
            "prior_source_identities": sorted({r["gt_person_audit_only"] for r in before
                                              if r["gt_person_audit_only"] is not None}),
            "prototype_history_events": len(history)}


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir, oracle_dir, features_dir, forensics_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fm, gt_frames, by_frame = read_features(features_dir, data_dir, oracle_dir)
    if len(fm["frame_ids"]) != 100:
        raise ValueError("The full 100 annotated WILDTRACK frame sets are required")
    previous = json.loads((Path(forensics_dir) / "comparison.json").read_text(encoding="utf-8"))
    if previous["source"]["embeddings_sha256"] != fm["features_sha256"] or previous["source"]["frames_sha256"] != fm["ground_truth_sha256"]:
        raise ValueError("Step 14.22 and cached embeddings/annotations differ")
    if not previous.get("selected_policy_replay_parity"):
        raise ValueError("Step 14.22 did not verify policy replay parity")
    selection = previous["selected_policy"]
    policy = Policy(selection["name"], selection["threshold"], selection["margin"],
                    selection["prototype"], selection["assignment"])
    if policy.prototype != "camera_balanced" or policy.assignment != "batch_hungarian":
        raise ValueError("This prototype audit targets the selected camera-balanced batch matcher")
    cases = previous["false_merge_cases"]
    if len(cases) != 1 or cases[0]["wrong_gt_pair"] != "115:116":
        raise ValueError("Step 14.22 false-merge case differs; stop and investigate")
    case = cases[0]
    wrong_assignment = case["first_wrong_assignment"]
    if not wrong_assignment or wrong_assignment["decision"] != "match_existing":
        raise ValueError("No recorded first wrong association")
    gid = int(case["global_id"])
    first_wrong = int(wrong_assignment["frame_id"])
    if first_wrong != 405:
        raise ValueError("Unexpected first wrong frame; do not reuse previous conclusions")

    gt_eligible = eligible_gt(by_frame)
    geometry, boundary_samples = border_geometry(gt_eligible)
    source_audit = inspect_raw_annotations(data_dir, gt_eligible)
    # Cross-check the prior audit, which discovered the suspicious C2 border rate.
    previous_crop_path = PREVIOUS_CROP_OUT / "audit.json"
    if previous_crop_path.is_file():
        previous_crop = json.loads(previous_crop_path.read_text(encoding="utf-8"))
        if previous_crop["source_sha256"] != fm["ground_truth_sha256"] or previous_crop["features_sha256"] != fm["features_sha256"]:
            raise ValueError("Previous crop audit uses different source annotations or features")
        for fraction in (0, 10, 20):
            original = previous_crop["crop_geometry"][f"pad_{fraction}"]
            for cam in CAMERAS:
                if original["by_camera"][cam]["crops"] != geometry[cam]["crops"]:
                    raise ValueError(f"Step 14.23 crop count changed: {cam} pad{fraction}")
                actual = (geometry[cam]["original_touch_any"] if fraction == 0
                          else geometry[cam]["variants"][f"pad_{fraction}"]["touch_any"])
                if actual != original["by_camera"][cam]["touch_border"]:
                    raise ValueError(f"Step 14.23 border count changed: {cam} pad{fraction}")
        source_audit["previous_crop_audit_parity"] = True
    else:
        source_audit["previous_crop_audit_parity"] = None
    matcher, predictions, labels = replay_history(by_frame, policy, gid)
    false_rows = false_pair_timeline(predictions, labels)
    traced_cases = reconstruct_cases(false_rows, matcher.audit)
    found = next((r for r in traced_cases if r["wrong_gt_pair"] == case["wrong_gt_pair"] and r["global_id"] == gid), None)
    if (found is None or found["false_pair_observations"] != case["false_pair_observations"] or
            not found["first_wrong_assignment"] or found["first_wrong_assignment"]["frame_id"] != first_wrong):
        raise ValueError("Replay changed the historical false merge; cannot interpret trace")
    if abs(found["first_wrong_assignment"]["match_cosine"] - wrong_assignment["match_cosine"]) > 1e-5:
        raise ValueError("Replay similarity differs from Step 14.22")
    dev, held = partition_ids(by_frame, seed=1421)
    for key, cohort in (("development", dev), ("previously_inspected_heldout", held)):
        stats = score_subset(predictions, labels, gt_frames, cohort)
        past = previous["selected_policy_scores"][key]
        for field in ("tp", "fp", "fn", "unique_wrongly_merged_identity_pairs"):
            if stats[field] != past[field]:
                raise ValueError(f"Replay parity failure for {key}/{field}")

    report = {
        "experiment": "Undeployed Step 14.21 camera-balanced prototype forensic replay and source crop geometry validation",
        "input": {"annotation_sha256": fm["ground_truth_sha256"],
                  "embeddings_sha256": fm["features_sha256"],
                  "checkpoint_sha256": fm["checkpoint_sha256"],
                  "annotated_frame_sets": len(fm["frame_ids"])},
        "geometry": geometry,
        "raw_annotation_validation": source_audit,
        "prototype": {"policy": selection, "replay_parity_verified": True,
                      "first_wrong_assignment": found["first_wrong_assignment"],
                      "same_frame_wrong_person_cosines": pair_cosines(by_frame),
                      "lineage": trace_summary(matcher.history, first_wrong, gid)},
        "limits": ["The offending prototype belongs to the Step 14.21 UNDEPLOYED matcher, not deployed EMA policy.",
                   "Raw current-frame cosine and stateful accumulated prototype cosine are different quantities.",
                   "Image boundary contact can be legitimate annotation truncation, not proof of a coordinate bug.",
                   "Ground-truth labels are used only to annotate audit records, never matcher scoring.",
                   "100 frames already examined; this is not a fresh held-out accuracy experiment.",
                   "No RTSP, MQTT, identity-event accuracy, model improvement, or official IDF1/HOTA measured."]}
    dump_json(output_dir / "audit.json", report)
    write_csv(output_dir / "boundary_examples.csv", boundary_samples,
              ["frame_id", "camera", "gt_id", "padding_fraction", "new_touch_edges", "old_xyxy", "nearest_margin_px"])
    write_csv(output_dir / "prototype_history.csv", matcher.history,
              ["frame_id", "batch_index", "action", "global_id", "camera", "track_id",
               "gt_person_audit_only", "prior_to_update_cosine", "prototype_change_one_minus_cosine",
               "origin_cameras", "origin_gt_persons_audit_only", "distinct_origin_gt_people",
               "prototype_cameras", "prototype_norm", "prior_cameras",
               "new_cross_camera_attachment", "encoded_observation_count", "feature_last_encoded_batch",
               "reuses_cached_feature"])
    dump_json(output_dir / "false_merge_verified.json", found)
    return report


def report(output_dir):
    data = json.loads((Path(output_dir) / "audit.json").read_text(encoding="utf-8"))
    print("Source box/annotation alignment: verified")
    for cam, vals in data["geometry"].items():
        p = vals["variants"]["pad_10"]
        print(f"{cam}: original border {vals['original_touch_any']}/{vals['crops']}; "
              f"pad10 border {p['touch_any']}/{vals['crops']}; newly affected {p['new_touch_any']}")
    prototype = data["prototype"]
    print("Replay parity:", prototype["replay_parity_verified"])
    print("First erroneous assignment:", prototype["first_wrong_assignment"])
    print("Prototype lineage:", prototype["lineage"])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        if name == "run":
            p.add_argument("--data-dir", type=Path, default=DATA)
            p.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE)
            p.add_argument("--features-dir", type=Path, default=DEFAULT_DIAG)
            p.add_argument("--forensics-dir", type=Path, default=DEFAULT_FORENSICS)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.data_dir, args.oracle_dir, args.features_dir,
                     args.forensics_dir, args.output_dir)
        print("Diagnostic saved to", args.output_dir / "audit.json")
        print("First wrong association cosine:", result["prototype"]["first_wrong_assignment"]["match_cosine"])
    else:
        report(args.output_dir)


if __name__ == "__main__":
    main()
