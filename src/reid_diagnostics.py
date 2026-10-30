"""Step 14.20: Frozen OSNet cross-camera embedding & association-gate audit.

This is a diagnostic only. It NEVER writes to production reid.py or modifies
matching thresholds. Dataset labels are retained for scoring, never passed to
CrossCameraMatcher or used in candidate ranking.

Typical use:
  python -m src.reid_diagnostics extract --oracle-dir outputs/evaluation/identity_evaluation/oracle_100 --threads 4 --affinity 0,2,4,6,8,10
  python -m src.reid_diagnostics analyze --oracle-dir outputs/evaluation/identity_evaluation/oracle_100
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

import numpy as np

from .identity_evaluation import (
    CAMERAS, DATA, DEFAULT_OUT, MIN_OBSERVATIONS, dump_json, eligible,
    ground_truth, load_predictions, oracle_tracks, pair_diagnostics, source_hash,
)

DEFAULT_DIAG = DEFAULT_OUT / "reid_diagnostics_100"
CAM_TO_INDEX = {c: i for i, c in enumerate(CAMERAS)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gt(data_dir, oracle_dir, frame_limit=None):
    manifest = json.loads((Path(oracle_dir) / "oracle_manifest.json").read_text(encoding="utf-8"))
    ids = [int(v) for v in manifest["frame_ids"]]
    if not ids or len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Oracle manifest has invalid or unsorted frame IDs")
    if frame_limit is not None:
        if frame_limit < 1 or frame_limit > len(ids):
            raise ValueError("--frames must be between 1 and the oracle frame count")
        ids = ids[:frame_limit]
    gt = ground_truth(data_dir, ids)
    # Never silently mix altered annotations with a prior oracle run.
    if frame_limit is None and source_hash(gt) != manifest["gt_sha256"]:
        raise ValueError("Oracle GT fingerprint mismatch: annotations have changed")
    if frame_limit is not None:
        full = ground_truth(data_dir, manifest["frame_ids"])
        if source_hash(full) != manifest["gt_sha256"]:
            raise ValueError("Oracle GT fingerprint mismatch: annotations have changed")
    return ids, gt, manifest


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not np.all(np.isfinite(v)) or n < 1e-12:
        raise ValueError("Invalid or near-zero embedding")
    return v / n


def extract(data_dir, oracle_dir, output_dir, threads, affinity, frame_limit=None):
    """Extract every eligible annotated crop exactly once with deployed OSNet.

    This is independent of the matcher's feature cache: all crops get features,
    so raw encoder separation is not mixed with cache/track warmup policy.
    """
    import cv2
    from .identity_evaluation import _configure_execution
    from .reid import AppearanceEncoder, MODEL_WEIGHTS

    ids, gt, om = get_gt(data_dir, oracle_dir, frame_limit)
    actual_affinity, actual_threads = _configure_execution(threads, affinity)
    encoder = AppearanceEncoder(backend="pytorch")
    if not Path(MODEL_WEIGHTS).is_file():
        raise FileNotFoundError(f"Missing OSNet checkpoint {MODEL_WEIGHTS}")
    vectors = []
    rows = []
    start = time.perf_counter()
    for i, fid in enumerate(ids, 1):
        requests, label_batches = [], []
        for cam in CAMERAS:
            people = []
            labels = []
            for obj in gt[fid][cam]:
                if not eligible(obj["xyxy"]):
                    continue
                people.append({"xyxy": list(obj["xyxy"]), "confidence": 1.0})
                labels.append(int(obj["gt_id"]))
            if not people:
                continue
            path = Path(data_dir) / "Image_subsets" / cam / f"{fid:08d}.png"
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError(f"Missing or invalid WILDTRACK rectified image: {path}")
            requests.append((cam, fid, image, people))
            label_batches.append(labels)
        if requests:
            feature_batches = encoder.encode_batch(requests)
            if len(feature_batches) != len(requests):
                raise ValueError("Unexpected encoder result count")
            for (cam, fid2, _, people), labels, features in zip(requests, label_batches, feature_batches):
                if len(features) != len(people):
                    raise ValueError("Missing returned crop embeddings")
                for obj, label, feature in zip(people, labels, features):
                    vectors.append(normalize(feature))
                    rows.append((int(fid2), CAM_TO_INDEX[cam], int(label),
                                 *[float(v) for v in obj["xyxy"]]))
        print(f"[{i}/{len(ids)}] annotated frame {fid}: {len(rows)} total crops", flush=True)
    if not vectors:
        raise ValueError("No eligible crops were encoded")
    if len({len(x) for x in vectors}) != 1:
        raise ValueError("Inconsistent encoder output dimensions")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "embeddings.npz"
    with features_path.with_suffix(".npz.part").open("wb") as f:
        np.savez_compressed(f, features=np.stack(vectors).astype(np.float32),
                            metadata=np.asarray(rows, dtype=np.float64))
    features_path.with_suffix(".npz.part").replace(features_path)
    result = {
        "source": "same synchronized rectified WILDTRACK PNGs as Step 14.19",
        "frame_ids": ids,
        "ground_truth_sha256": source_hash(gt),
        "parent_oracle_gt_sha256": om["gt_sha256"],
        "checkpoint_sha256": sha256(MODEL_WEIGHTS),
        "feature_file": features_path.name,
        "features_sha256": sha256(features_path),
        "crop_count": len(rows),
        "feature_dimension": len(vectors[0]),
        "encoder": "deployed BoxMOT PyTorch OSNet; every crop encoded; L2 normalised",
        "execution": {"threads": actual_threads, "affinity": actual_affinity,
                      "seconds": round(time.perf_counter() - start, 3)},
        "warning": "GT IDs are in score metadata only, not supplied to OSNet or to cross-camera matching",
    }
    dump_json(output_dir / "embeddings_manifest.json", result)
    return result


def read_features(output_dir, data_dir, oracle_dir):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "embeddings_manifest.json").read_text(encoding="utf-8"))
    ids, gt, _ = get_gt(data_dir, oracle_dir, None)
    if not manifest["frame_ids"] or not set(manifest["frame_ids"]).issubset(set(ids)) or manifest["frame_ids"] != ids[:len(manifest["frame_ids"])]:
        raise ValueError("Embedding frame IDs are not an oracle prefix")
    selected = {fid: gt[fid] for fid in manifest["frame_ids"]}
    if source_hash(selected) != manifest["ground_truth_sha256"]:
        raise ValueError("Annotation fingerprint changed since embedding extraction")
    feat_path = output_dir / manifest["feature_file"]
    if sha256(feat_path) != manifest["features_sha256"]:
        raise ValueError("Embedding file fingerprint changed")
    with np.load(feat_path, allow_pickle=False) as data:
        feats = np.array(data["features"], dtype=np.float32)
        meta = np.array(data["metadata"], dtype=np.float64)
    if feats.ndim != 2 or meta.shape != (len(feats), 7) or not len(feats):
        raise ValueError("Invalid feature array or metadata shape")
    if len(feats) != manifest["crop_count"] or feats.shape[1] != manifest["feature_dimension"]:
        raise ValueError("Cached embeddings do not match extraction manifest")
    if not np.all(np.isfinite(feats)) or not np.all(np.isfinite(meta)):
        raise ValueError("Non-finite stored data")
    if np.max(np.abs(np.linalg.norm(feats, axis=1) - 1)) > 1e-3:
        raise ValueError("Embeddings are not L2 normalized")
    by_frame = defaultdict(lambda: defaultdict(list))
    for fid in manifest["frame_ids"]:
        by_frame[int(fid)]  # Keep empty annotated frames in temporal replay.
    observed = set()
    for feat, vals in zip(feats, meta):
        fid, ci, pid = [int(v) for v in vals[:3]]
        if any(float(int(v)) != v for v in vals[:3]) or ci not in range(len(CAMERAS)):
            raise ValueError("Invalid feature metadata IDs")
        cam = CAMERAS[ci]
        if fid not in selected:
            raise ValueError("Unknown annotated frame in cached embeddings")
        xyxy = tuple(float(v) for v in vals[3:])
        key = (fid, cam, xyxy)
        if key in observed:
            raise ValueError("Duplicate coordinates within a camera/frame; cannot replay safely")
        observed.add(key)
        truth = [g for g in selected[fid][cam] if int(g["gt_id"]) == pid
                 and tuple(map(float, g["xyxy"])) == xyxy]
        if len(truth) != 1 or not eligible(xyxy):
            raise ValueError("Embedding crop does not match exactly one eligible GT crop")
        by_frame[fid][cam].append((pid, xyxy, feat))
    expected = sum(eligible(o["xyxy"]) for fid in selected
                   for cam in CAMERAS for o in selected[fid][cam])
    if len(feats) != expected:
        raise ValueError(f"Expected {expected} GT crops, found {len(feats)} cached features")
    return manifest, selected, by_frame


def quantiles(values):
    if not values:
        return {"n": 0, "mean": None, "p05": None, "p25": None,
                "p50": None, "p75": None, "p95": None}
    v = np.asarray(values, dtype=float)
    q = np.quantile(v, [.05, .25, .5, .75, .95])
    return dict(n=len(values), mean=round(float(np.mean(v)), 4),
                **{k: round(float(x), 4) for k, x in zip(("p05", "p25", "p50", "p75", "p95"), q)})


def auc_score(positive, negative):
    """Tie-aware AUC via ranks; returns None if either class absent."""
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=float)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=int)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_v[j] == sorted_v[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    npos, nneg = len(positive), len(negative)
    return round(float((np.sum(ranks[labels == 1]) - npos * (npos + 1) / 2) / (npos * nneg)), 4)


def score_pairs(by_frame, thresholds=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)):
    """Same-instant cross-camera pairs and directed retrieval queries.

    Each scored observation is a frame+camera pair, NOT an independent person.
    """
    per_pair = {f"{a}-{b}": {"genuine": [], "impostor": [], "ranks": [],
                              "query_count": 0, "queries_with_gt_gallery": 0,
                              "identity_pairs": set()} for i, a in enumerate(CAMERAS)
                for b in CAMERAS[i + 1:]}
    hard_impostors, low_genuine = [], []
    for fid, frame in sorted(by_frame.items()):
        for i, a in enumerate(CAMERAS):
            for b in CAMERAS[i + 1:]:
                A, B = frame.get(a, []), frame.get(b, [])
                if not A or not B:
                    continue
                af = np.stack([x[2] for x in A]); bf = np.stack([x[2] for x in B])
                similarities = af @ bf.T
                r = per_pair[f"{a}-{b}"]
                labels_a, labels_b = [x[0] for x in A], [x[0] for x in B]
                for ai, pid_a in enumerate(labels_a):
                    for bi, pid_b in enumerate(labels_b):
                        s = float(similarities[ai, bi])
                        record = {"frame_id": fid, "camera_a": a, "camera_b": b,
                                  "gt_id_a": pid_a, "gt_id_b": pid_b,
                                  "cosine": round(s, 5)}
                        if pid_a == pid_b:
                            r["genuine"].append(s)
                            r["identity_pairs"].add(pid_a)
                            low_genuine.append(record)
                        else:
                            r["impostor"].append(s)
                            hard_impostors.append(record)
                # Each direction has a separate target gallery and matching task.
                for query_labels, gallery_labels, scores in (
                    (labels_a, labels_b, similarities),
                    (labels_b, labels_a, similarities.T),
                ):
                    gallery = {pid: j for j, pid in enumerate(gallery_labels)}
                    for qi, pid in enumerate(query_labels):
                        r["query_count"] += 1
                        if pid not in gallery:
                            continue
                        r["queries_with_gt_gallery"] += 1
                        # Stable, deterministic rank; ties are resolved by gallery order.
                        order = np.argsort(-scores[qi], kind="stable")
                        rank = int(np.flatnonzero(order == gallery[pid])[0]) + 1
                        r["ranks"].append(rank)
    details, global_p, global_n = {}, [], []
    for pair, r in per_pair.items():
        global_p.extend(r["genuine"])
        global_n.extend(r["impostor"])
        ranks = r["ranks"]
        details[pair] = {
            "genuine": quantiles(r["genuine"]), "impostor": quantiles(r["impostor"]),
            "roc_auc": auc_score(r["genuine"], r["impostor"]),
            "distinct_genuine_gt_ids": len(r["identity_pairs"]),
            "retrieval": {"queries": r["query_count"], "eligible_queries": len(ranks),
                          "top1": round(sum(x <= 1 for x in ranks) / len(ranks), 4) if ranks else None,
                          "top5": round(sum(x <= 5 for x in ranks) / len(ranks), 4) if ranks else None,
                          "mean_reciprocal_rank": round(sum(1 / x for x in ranks) / len(ranks), 4) if ranks else None},
        }
    summary = {
        "genuine": quantiles(global_p), "impostor": quantiles(global_n),
        "roc_auc": auc_score(global_p, global_n),
        "threshold_sensitivity": [], "camera_pairs": details,
        "methodology": "Same-instant cross-camera crop cosine; directed Top1/5 retrieval within each opposing camera's current-frame gallery; no learned classifier or matcher gates",
        "caveats": [
            "Repeated annotated instants and identities create correlated observations; this is diagnostic, not an independent-person generalization estimate.",
            "Same-frame candidate galleries differ from the deployed matcher's running global identity gallery.",
            "Threshold sensitivity describes raw cosine scores ONLY, not the deployed margin/stateful matching policy.",
            "Precision and false-positive rate are conditioned on annotated ROI; projected ROI mask is not verified.",
        ],
    }
    for t in thresholds:
        summary["threshold_sensitivity"].append({
            "threshold": t,
            "genuine_above_threshold": round(sum(s >= t for s in global_p) / len(global_p), 4) if global_p else None,
            "impostor_above_threshold": round(sum(s >= t for s in global_n) / len(global_n), 4) if global_n else None,
        })
    return summary, sorted(hard_impostors, key=lambda r: -r["cosine"])[:50], sorted(low_genuine, key=lambda r: r["cosine"])[:50]


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FrozenEncoder:
    """Encoder stand-in for a reference matcher: only boxes locate saved vectors.

    Label metadata is never passed to the matcher through this adapter.
    """
    def __init__(self, by_frame):
        self.lookup = {}
        for fid, cameras in by_frame.items():
            for cam, entries in cameras.items():
                for _, xyxy, feature in entries:
                    key = (fid, cam, xyxy)
                    if key in self.lookup:
                        raise ValueError("Ambiguous frozen crop")
                    self.lookup[key] = feature

    def encode_batch(self, requests):
        result = []
        for cam, fid, _, people in requests:
            result.append(np.stack([
                self.lookup[(fid, cam, tuple(map(float, p["xyxy"])))]
                for p in people
            ]))
        return result


def shadow_audit(by_frame, gt_frames, original_oracle_predictions=None):
    """Reimplement matching gates *as an auditor*, assert parity with live matcher.

    Production CrossCameraMatcher is instantiated and replayed with frozen
    features, unchanged. The independent shadow state records rejected candidates.
    Any change in reference matching semantics makes the test fail closed.
    """
    from .reid import CrossCameraMatcher, MATCH_THRESHOLD, MATCH_MARGIN, unit
    encoder = FrozenEncoder(by_frame)
    matcher = CrossCameraMatcher(encoder, min_observations=MIN_OBSERVATIONS,
                                 threshold=MATCH_THRESHOLD, margin=MATCH_MARGIN,
                                 refresh_every=5, batch_across_cameras=True)
    batch_index = -1
    local, global_states, global_truth = {}, {}, defaultdict(set)
    next_id = 1
    local_ids = {cam: {} for cam in CAMERAS}
    audit = []
    reference_predictions = {}
    for fid in sorted(by_frame):
        batch_index += 1
        tracks = oracle_tracks(gt_frames[fid], local_ids)
        # Exact oracle input coordinates and ordering, verified by embedding extraction.
        for cam in CAMERAS:
            frame_entries = by_frame[fid].get(cam, [])
            if [tuple(map(float, x["xyxy"])) for x in tracks[cam]] != [x[1] for x in frame_entries]:
                raise ValueError(f"GT crop order mismatch at {fid}/{cam}; cannot replay")
        # Production matcher does not inspect image data when using FrozenEncoder.
        reference, production_decisions = matcher.update(fid, {cam: None for cam in CAMERAS}, tracks)
        reference_predictions[fid] = reference
        annotated = {cam: [dict(p, global_id=None) for p in tracks[cam]] for cam in CAMERAS}
        gt_of_track = {cam: {p["track_id"]: entry[0]
                             for p, entry in zip(tracks[cam], by_frame[fid].get(cam, []))}
                       for cam in CAMERAS}
        # Stage 1: identical encoding/cache update schedule.
        for cam in CAMERAS:
            for person in annotated[cam]:
                key = cam, person["track_id"]
                state = local.get(key)
                if (state is None or state["count"] < MIN_OBSERVATIONS or
                        batch_index - state["last_encoded"] >= 5):
                    feature = encoder.lookup[(fid, cam, tuple(map(float, person["xyxy"])))]
                    if state is None:
                        local[key] = {"feature": unit(feature), "count": 1,
                                      "global_id": None, "last_encoded": batch_index}
                    else:
                        state["feature"] = unit(0.8 * state["feature"] + 0.2 * unit(feature))
                        state["count"] += 1
                        state["last_encoded"] = batch_index
        shadow_decisions = []
        # Stage 2: deterministic sequential camera assignment (same as matcher).
        for cam in CAMERAS:
            for person in annotated[cam]:
                key = cam, person["track_id"]
                state = local.get(key)
                if state is None or state["count"] < MIN_OBSERVATIONS:
                    continue
                if state["global_id"] is None:
                    candidates = sorted(((float(np.dot(state["feature"], v["feature"])), gid)
                                         for gid, v in global_states.items() if cam not in v["cameras"]),
                                        reverse=True)
                    best = candidates[0][0] if candidates else -1.0
                    second = candidates[1][0] if len(candidates) > 1 else -1.0
                    matched = best >= MATCH_THRESHOLD and best - second >= MATCH_MARGIN
                    reason = ("accepted" if matched else "no_candidate" if not candidates
                              else "below_threshold" if best < MATCH_THRESHOLD else "below_margin")
                    gt_id = gt_of_track[cam][person["track_id"]]
                    truthful = [(rank, score, gid) for rank, (score, gid) in enumerate(candidates, 1)
                                if gt_id in global_truth[gid]]
                    decision_row = {
                        "frame_id": fid, "camera": cam, "local_track_id": person["track_id"],
                        "gt_id_for_scoring_only": gt_id,
                        "reason": reason, "best_similarity": round(best, 5) if candidates else None,
                        "second_similarity": round(second, 5) if len(candidates) > 1 else None,
                        "best_minus_second": round(best - second, 5) if candidates else None,
                        "candidate_count": len(candidates),
                        "correct_candidate_exists": bool(truthful),
                        "correct_candidate_rank": truthful[0][0] if truthful else None,
                        "correct_candidate_similarity": round(truthful[0][1], 5) if truthful else None,
                        "best_candidate_correct": bool(candidates and gt_id in global_truth[candidates[0][1]]),
                    }
                    if matched:
                        gid = candidates[0][1]
                        decision = "cross_camera_match"
                    else:
                        gid = next_id
                        next_id += 1
                        global_states[gid] = {"feature": state["feature"].copy(), "cameras": {}}
                        decision = "new_identity"
                    state["global_id"] = gid
                    global_states[gid]["cameras"][cam] = person["track_id"]
                    global_truth[gid].add(gt_id)  # scoring-only ledger, not candidate selection
                    decision_row.update({"assigned_gid": gid, "decision": decision,
                                         "assigned_gid_consistent_with_gt": len(global_truth[gid]) == 1})
                    audit.append(decision_row)
                    shadow_decisions.append({"frame_id": fid, "camera": cam,
                                             "track_id": person["track_id"], "global_id": gid,
                                             "decision": decision,
                                             "similarity": round(best, 3) if matched else None})
                gid = state["global_id"]
                identity = global_states[gid]
                identity["feature"] = unit(0.9 * identity["feature"] + 0.1 * state["feature"])
                person["global_id"] = gid
        if shadow_decisions != production_decisions:
            raise RuntimeError(f"Matcher decision parity failed at annotated frame {fid}; audit invalid")
        for cam in CAMERAS:
            if [p["global_id"] for p in annotated[cam]] != [p["global_id"] for p in reference[cam]]:
                raise RuntimeError(f"Matcher global-ID parity failed at frame {fid}/{cam}")
    original_comparison = None
    if original_oracle_predictions is not None:
        # An optional prior run may differ slightly from the independently
        # re-encoded frozen features; this comparison is informational only.
        comparisons = differences = 0
        for fid, cam_predictions in reference_predictions.items():
            for cam in CAMERAS:
                if len(cam_predictions[cam]) != len(original_oracle_predictions[fid][cam]):
                    raise ValueError("Original oracle observation count changed")
                for p, q in zip(cam_predictions[cam], original_oracle_predictions[fid][cam]):
                    comparisons += 1
                    differences += int(p["global_id"] != q["global_id"])
        original_comparison = {"compared_observations": comparisons,
                               "global_id_differences": differences}
    c = Counter(row["reason"] for row in audit)
    correct = [r for r in audit if r["correct_candidate_exists"]]
    counts = Counter(row["reason"] for row in correct)
    pair_totals = {"true_gt_pairs": 0, "predicted_pairs": 0, "correct_pairs": 0}
    breakdown = pair_diagnostics(gt_frames, reference_predictions)
    for row in breakdown.values():
        pair_totals["true_gt_pairs"] += row["true_gt_pairs"]
        pair_totals["predicted_pairs"] += row["predicted_same_identity_pairs"]
        pair_totals["correct_pairs"] += row["correct_same_identity_pairs"]
    pair_totals["precision"] = round(pair_totals["correct_pairs"] / pair_totals["predicted_pairs"], 4) if pair_totals["predicted_pairs"] else None
    pair_totals["recall"] = round(pair_totals["correct_pairs"] / pair_totals["true_gt_pairs"], 4) if pair_totals["true_gt_pairs"] else None
    result = {
        "total_identity_creation_decisions": len(audit),
        "gate_reasons": dict(c),
        "decisions_with_existing_correct_candidate": len(correct),
        "gates_when_correct_candidate_exists": dict(counts),
        "best_candidate_correct_when_correct_exists": sum(r["best_candidate_correct"] for r in correct),
        "reference_matcher_parity": "verified for every global ID and association decision",
        "match_threshold": MATCH_THRESHOLD, "match_margin": MATCH_MARGIN,
        "refresh_every": 5, "min_encoded_observations": MIN_OBSERVATIONS,
        "cross_camera_pair_totals": pair_totals,
        "pairwise_diagnostics": breakdown,
        "previous_oracle_comparison": original_comparison,
        "caveats": [
            "A creation decision is not equivalent to a ground-truth frame pair; counts have different denominators.",
            "Existing correct candidate means at least one currently known global identity contains the GT person, not necessarily that the correct identity is ranked first.",
            "The scoring-only identity ledger never changes similarity, filtering, or assignment.",
            "This is frozen-feature policy replay; thresholds/policy are unchanged, and no new live pipeline performance is claimed.",
            "The extraction encodes every eligible crop while the original oracle batched only crops due for refresh; tiny numeric differences can affect borderline decisions.",
        ],
    }
    return result, audit


def analyze(data_dir, oracle_dir, output_dir):
    output_dir = Path(output_dir)
    manifest, gt, by_frame = read_features(output_dir, data_dir, oracle_dir)
    score, hard_impostors, low_genuine = score_pairs(by_frame)
    previous = Path(oracle_dir) / "oracle_predictions.jsonl"
    previous_predictions = None
    if previous.is_file():
        previous_predictions, unused_rows = load_predictions(previous, manifest["frame_ids"])
        if unused_rows:
            raise ValueError("Previous oracle prediction file contains unexpected frame IDs")
        for fid in manifest["frame_ids"]:
            for cam in CAMERAS:
                original_boxes = [tuple(map(float, p["xyxy"])) for p in previous_predictions[fid][cam]]
                frozen_boxes = [entry[1] for entry in by_frame[fid].get(cam, [])]
                if original_boxes != frozen_boxes:
                    raise ValueError("Original oracle boxes differ from cached diagnostic inputs; cannot compare")
    audit_summary, decisions = shadow_audit(by_frame, gt, previous_predictions)
    report = {
        "input": {"frame_count": len(manifest["frame_ids"]),
                  "crop_count": manifest["crop_count"],
                  "ground_truth_sha256": manifest["ground_truth_sha256"],
                  "checkpoint_sha256": manifest["checkpoint_sha256"],
                  "feature_dimension": manifest["feature_dimension"]},
        "embedding_quality": score,
        "association_gates": audit_summary,
        "recommended_diagnostic_readout": [
            "High retrieval with low association recall suggests current global-ID gating/state policy contributes substantially.",
            "Low same-frame retrieval suggests pretrained features, crop quality or camera-domain shift are limiting even without gating.",
            "Do not adopt a lower live threshold using this diagnostic data; calibrate on disjoint identities and test held-out annotated identities.",
        ],
    }
    dump_json(output_dir / "diagnostics.json", report)
    rows = []
    for pair, v in sorted(score["camera_pairs"].items()):
        rows.append({"camera_pair": pair, "genuine_pairs": v["genuine"]["n"],
                     "impostor_pairs": v["impostor"]["n"], "auc": v["roc_auc"],
                     "genuine_p50": v["genuine"]["p50"], "impostor_p50": v["impostor"]["p50"],
                     "eligible_retrieval_queries": v["retrieval"]["eligible_queries"],
                     "top1": v["retrieval"]["top1"], "top5": v["retrieval"]["top5"]})
    write_csv(output_dir / "camera_pair_retrieval.csv", rows,
              ["camera_pair", "genuine_pairs", "impostor_pairs", "auc", "genuine_p50",
               "impostor_p50", "eligible_retrieval_queries", "top1", "top5"])
    write_csv(output_dir / "gate_decisions.csv", decisions,
              ["frame_id", "camera", "local_track_id", "gt_id_for_scoring_only", "reason",
               "best_similarity", "second_similarity", "best_minus_second", "candidate_count",
               "correct_candidate_exists", "correct_candidate_rank", "correct_candidate_similarity",
               "best_candidate_correct", "assigned_gid", "decision",
               "assigned_gid_consistent_with_gt"])
    hard_columns = ["frame_id", "camera_a", "camera_b", "gt_id_a", "gt_id_b", "cosine"]
    write_csv(output_dir / "hard_impostors.csv", hard_impostors, hard_columns)
    write_csv(output_dir / "low_genuine.csv", low_genuine, hard_columns)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "analyze"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--data-dir", type=Path, default=DATA)
        cmd.add_argument("--oracle-dir", type=Path, default=DEFAULT_OUT / "oracle_100")
        cmd.add_argument("--output-dir", type=Path, default=DEFAULT_DIAG)
        if command == "extract":
            cmd.add_argument("--frames", type=int, default=None, help="first N oracle annotated frames; omit for all")
            cmd.add_argument("--threads", type=int, default=4)
            cmd.add_argument("--affinity", default="all")
    args = parser.parse_args(argv)
    if args.command == "extract":
        result = extract(args.data_dir, args.oracle_dir, args.output_dir,
                         args.threads, args.affinity, args.frames)
        print("Extracted", result["crop_count"], "embeddings to", args.output_dir)
    else:
        result = analyze(args.data_dir, args.oracle_dir, args.output_dir)
        print("Raw cosine AUC:", result["embedding_quality"]["roc_auc"])
        print("Matcher gate counts:", result["association_gates"]["gate_reasons"])
        print("Saved", args.output_dir / "diagnostics.json")


if __name__ == "__main__":
    main()
