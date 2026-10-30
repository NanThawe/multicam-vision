"""Step 14.21: identity-disjoint, cached-feature association-policy ablations.

No production changes. Commands:
  python -m src.association_ablation run --oracle-dir ... --features-dir ...
  python -m src.association_ablation report --output-dir ...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json, oracle_tracks
from .reid_diagnostics import DEFAULT_DIAG, FrozenEncoder, read_features, sha256
from .reid import CrossCameraMatcher, MATCH_THRESHOLD, MATCH_MARGIN, unit

DEFAULT_OUTPUT = DEFAULT_OUT / "association_ablation_100"


@dataclass(frozen=True)
class Policy:
    name: str
    threshold: float = MATCH_THRESHOLD
    margin: float = MATCH_MARGIN
    prototype: str = "ema"      # ema (deployed) or camera_balanced
    assignment: str = "greedy"   # greedy (deployed) or batch_hungarian


def partition_ids(by_frame, seed=1421, dev_fraction=0.6):
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    ids = sorted({int(pid) for frame in by_frame.values() for entries in frame.values()
                  for pid, _, _ in entries})
    if len(ids) < 4:
        raise ValueError("Need at least four GT identities to split")
    # Hash based only on person identity, never frame number or model performance.
    ranked = sorted(ids, key=lambda pid: hashlib.sha256(f"{seed}:{pid}".encode()).digest())
    n = min(len(ids)-2, max(2, round(dev_fraction * len(ids))))
    dev = set(ranked[:n])
    heldout = set(ranked[n:])
    assert not dev & heldout and dev | heldout == set(ids)
    return dev, heldout


def build_tracks(by_frame):
    """Use ground-truth ID only to manufacture stable *camera-local* track keys.

    A cross-camera GT ID is never put into the matcher input.
    """
    local_ids = {cam: {} for cam in CAMERAS}
    output = {}
    gt_ids = {}
    for fid, frame in sorted(by_frame.items()):
        tracks = {}
        labels = {}
        for cam in CAMERAS:
            tracks[cam] = []
            labels[cam] = []
            seen_pids = set()
            for pid, xyxy, _ in frame.get(cam, []):
                if pid in seen_pids:
                    raise ValueError(f"Duplicate GT person {pid} in camera {cam}, frame {fid}")
                seen_pids.add(pid)
                if pid not in local_ids[cam]:
                    local_ids[cam][pid] = len(local_ids[cam]) + 1
                tracks[cam].append({"track_id": local_ids[cam][pid],
                                    "xyxy": list(xyxy), "confidence": 1.0})
                labels[cam].append(int(pid))
        output[int(fid)] = tracks
        gt_ids[int(fid)] = labels
    return output, gt_ids


def baseline_replay(by_frame):
    encoder = FrozenEncoder(by_frame)
    matcher = CrossCameraMatcher(encoder, min_observations=3,
                                 threshold=MATCH_THRESHOLD, margin=MATCH_MARGIN,
                                 refresh_every=5, batch_across_cameras=True)
    tracks, gt_labels = build_tracks(by_frame)
    result = {}
    for fid, track in tracks.items():
        annotated, _ = matcher.update(fid, {cam: None for cam in CAMERAS}, track)
        result[fid] = {cam: [p["global_id"] for p in annotated[cam]] for cam in CAMERAS}
    return result, gt_labels


class ExperimentalMatcher:
    """Offline alternative; does not modify production CrossCameraMatcher.

    Cache/warmup mirror the deployed matcher. Camera-balanced prototype is the
    L2-normalized mean of latest per-camera local prototypes, not a GT-ID mean.
    Batch Hungarian competes unassigned tracks for a *camera's* existing IDs;
    already assigned camera IDs are never reused in that camera.
    """
    def __init__(self, policy):
        if policy.prototype not in ("ema", "camera_balanced"):
            raise ValueError("Unknown prototype")
        if policy.assignment not in ("greedy", "batch_hungarian"):
            raise ValueError("Unknown assignment")
        if not 0 <= policy.threshold <= 1 or not 0 <= policy.margin <= 1:
            raise ValueError("Invalid similarity gate")
        self.policy = policy
        self.batch_index = -1
        self.local = {}
        self.globals = {}
        self.next_id = 1
        self.decisions = Counter()

    def _new_global(self, feature):
        gid = self.next_id
        self.next_id += 1
        self.globals[gid] = {"feature": feature.copy(), "cameras": {}, "per_camera": {}}
        self.decisions["new_id"] += 1
        return gid

    def _record_identity(self, gid, cam, person, state):
        identity = self.globals[gid]
        identity["cameras"][cam] = person["track_id"]
        state["global_id"] = gid
        if self.policy.prototype == "camera_balanced":
            identity["per_camera"][cam] = state["feature"].copy()
            identity["feature"] = unit(np.mean(list(identity["per_camera"].values()), axis=0))
        # For EMA, initialization and the per-observation update follow in update().

    def _update_global(self, gid, cam, state):
        identity = self.globals[gid]
        if self.policy.prototype == "ema":
            identity["feature"] = unit(0.9 * identity["feature"] + 0.1 * state["feature"])
        else:
            identity["per_camera"][cam] = state["feature"].copy()
            identity["feature"] = unit(np.mean(list(identity["per_camera"].values()), axis=0))

    def _score(self, feature, cam):
        return sorted(((float(np.dot(feature, identity["feature"])), gid)
                       for gid, identity in self.globals.items()
                       if cam not in identity["cameras"]), reverse=True)

    def _greedy_camera(self, cam, people):
        for person in people:
            state = self.local.get((cam, person["track_id"]))
            if state is None or state["count"] < 3:
                continue
            if state["global_id"] is None:
                candidates = self._score(state["feature"], cam)
                best = candidates[0][0] if candidates else -1.
                second = candidates[1][0] if len(candidates) > 1 else -1.
                if best >= self.policy.threshold and best-second >= self.policy.margin:
                    gid = candidates[0][1]
                    self.decisions["cross_camera_match"] += 1
                else:
                    gid = self._new_global(state["feature"])
                self._record_identity(gid, cam, person, state)
            self._update_global(state["global_id"], cam, state)

    def _batch_camera(self, cam, people):
        # Keep ongoing tracks' global feature updates, then match only newly
        # mature local tracks simultaneously. New assignments never share a gid.
        pending = []
        for person in people:
            state = self.local.get((cam, person["track_id"]))
            if state is None or state["count"] < 3:
                continue
            if state["global_id"] is None:
                pending.append((person, state))
            else:
                self._update_global(state["global_id"], cam, state)
        if not pending:
            return
        from scipy.optimize import linear_sum_assignment
        available = [gid for gid, identity in sorted(self.globals.items())
                     if cam not in identity["cameras"]]
        n, m = len(pending), len(available)
        chosen = {}
        if m:
            # Each row competes for available global identities and its own
            # dummy (new ID), with inaccessible choices set to very high cost.
            scores = np.stack([s["feature"] for _, s in pending]) @ np.stack(
                [self.globals[g]["feature"] for g in available]).T
            costs = np.full((n, m+n), 1e6, dtype=np.float64)
            for i in range(n):
                order = np.argsort(-scores[i], kind="stable")
                top = float(scores[i, order[0]])
                second = float(scores[i, order[1]]) if m > 1 else -1.
                # With a positive margin, only the top candidate can pass;
                # with zero margin, multiple candidates can compete.
                for j in range(m):
                    score = float(scores[i, j])
                    if score < self.policy.threshold:
                        continue
                    if self.policy.margin > 0 and (j != order[0] or top-second < self.policy.margin):
                        continue
                    costs[i, j] = -score
                costs[i, m+i] = -self.policy.threshold + 1e-6
            rows, cols = linear_sum_assignment(costs)
            chosen = {int(i): available[int(j)] for i, j in zip(rows, cols) if j < m and costs[i, j] < 1e5}
        for i, (person, state) in enumerate(pending):
            gid = chosen.get(i)
            if gid is None:
                gid = self._new_global(state["feature"])
            else:
                self.decisions["cross_camera_match"] += 1
            self._record_identity(gid, cam, person, state)
            self._update_global(gid, cam, state)

    def update(self, fid, tracks, feature_lookup):
        self.batch_index += 1
        for cam in CAMERAS:
            for person in tracks[cam]:
                key = cam, person["track_id"]
                state = self.local.get(key)
                if state is None or state["count"] < 3 or self.batch_index-state["last_encoded"] >= 5:
                    feature = feature_lookup[(fid, cam, tuple(map(float, person["xyxy"])))]
                    if state is None:
                        self.local[key] = {"feature": unit(feature), "count": 1,
                                           "global_id": None, "last_encoded": self.batch_index,
                                           "sum": feature.copy(), "n": 1}
                    else:
                        if self.policy.prototype == "ema":
                            state["feature"] = unit(0.8 * state["feature"] + 0.2 * feature)
                        else:
                            state["sum"] = state["sum"] + feature
                            state["n"] += 1
                            state["feature"] = unit(state["sum"] / state["n"])
                        state["count"] += 1
                        state["last_encoded"] = self.batch_index
        for cam in CAMERAS:
            if self.policy.assignment == "greedy":
                self._greedy_camera(cam, tracks[cam])
            else:
                self._batch_camera(cam, tracks[cam])
        return {cam: [self.local.get((cam, p["track_id"]), {}).get("global_id")
                      for p in tracks[cam]] for cam in CAMERAS}


def replay(by_frame, policy):
    lookup = FrozenEncoder(by_frame).lookup
    tracks, labels = build_tracks(by_frame)
    matcher = ExperimentalMatcher(policy)
    predictions = {}
    for fid, track in tracks.items():
        predictions[fid] = matcher.update(fid, track, lookup)
    return predictions, labels, dict(matcher.decisions)


def score_subset(predictions, gt_labels, gt_frames, selected_ids):
    """Score entire live gallery but only labeled target IDs' opportunities.

    False merges with a target ID and *any* other ID count against precision.
    Non-target/non-target pairs are ignored. GT denominator includes tiny boxes
    excluded from embedding extraction, so missing crops remain missed pairs.
    """
    selected_ids = set(selected_ids)
    count = Counter()
    camera_pairs = defaultdict(Counter)
    true_identity_ids = set()
    tp_identity_ids = set()
    wrongly_merged_identity_pairs = set()
    for fid, cam_preds in predictions.items():
        for i, a in enumerate(CAMERAS):
            for b in CAMERAS[i+1:]:
                key = f"{a}-{b}"
                # GT opportunities include small boxes omitted by embedding extraction.
                gt_a = {int(o["gt_id"]) for o in gt_frames[fid][a]}
                gt_b = {int(o["gt_id"]) for o in gt_frames[fid][b]}
                shared_target_ids = (gt_a & gt_b) & selected_ids
                opportunities = len(shared_target_ids)
                true_identity_ids.update(shared_target_ids)
                count["true_pairs"] += opportunities
                camera_pairs[key]["true_pairs"] += opportunities
                la, lb = gt_labels[fid][a], gt_labels[fid][b]
                pa, pb = cam_preds[a], cam_preds[b]
                if len(la) != len(pa) or len(lb) != len(pb):
                    raise ValueError("Prediction/GT embedding-row lengths differ")
                for pid_a, gid_a in zip(la, pa):
                    if gid_a is None:
                        continue
                    for pid_b, gid_b in zip(lb, pb):
                        if gid_b is None or gid_a != gid_b:
                            continue
                        if pid_a not in selected_ids and pid_b not in selected_ids:
                            continue
                        if pid_a == pid_b and pid_a in selected_ids:
                            tp_identity_ids.add(pid_a)
                            count["tp"] += 1
                            camera_pairs[key]["tp"] += 1
                        else:
                            wrongly_merged_identity_pairs.add(tuple(sorted((pid_a, pid_b))))
                            count["fp"] += 1
                            camera_pairs[key]["fp"] += 1
    tp, fp, n_gt = count["tp"], count["fp"], count["true_pairs"]
    if tp > n_gt:
        raise ValueError("Correctly matched pairs exceeded distinct GT opportunities; check duplicated labels")
    precision = tp / (tp+fp) if tp+fp else None
    recall = tp / n_gt if n_gt else None
    f1 = 2*precision*recall/(precision+recall) if (precision is not None and recall is not None and precision+recall > 0) else 0.0
    return {"tp": tp, "fp": fp, "fn": n_gt-tp, "true_pairs": n_gt,
            "predicted_pairs": tp+fp,
            "unique_true_gt_identities": len(true_identity_ids),
            "unique_correctly_linked_gt_identities": len(tp_identity_ids),
            "unique_wrongly_merged_identity_pairs": len(wrongly_merged_identity_pairs),
            "precision": round(precision, 5) if precision is not None else None,
            "recall": round(recall, 5) if recall is not None else None,
            "f1": round(f1, 5) if n_gt else None,
            "camera_pairs": {k: dict(camera_pairs[k]) for k in sorted(camera_pairs)}}


def candidates(grid="standard"):
    if grid == "smoke":
        thresholds, margins = (0.7, 0.85), (0., 0.05)
    elif grid == "standard":
        thresholds, margins = (0.60, 0.70, 0.80, 0.85), (0., 0.05)
    else:
        raise ValueError("grid must be smoke or standard")
    result = [Policy("deployed_baseline")]
    for proto, assign in (("ema", "greedy"), ("ema", "batch_hungarian"),
                          ("camera_balanced", "greedy"),
                          ("camera_balanced", "batch_hungarian")):
        for threshold in thresholds:
            for margin in margins:
                name = f"{proto}_{assign}_t{int(threshold*100):02}_m{int(margin*100):02}"
                if (proto, assign, threshold, margin) == ("ema", "greedy", MATCH_THRESHOLD, MATCH_MARGIN):
                    continue
                result.append(Policy(name, threshold, margin, proto, assign))
    return result


def _config_rank(row):
    metrics, cfg = row["dev"], row["config"]
    # Conservative tie-break: keep deployed matching if no measured benefit;
    # then favor simpler policies and stricter gates at equal effectiveness.
    complexity = (0 if cfg["name"] == "deployed_baseline" else
                  1 + int(cfg.get("prototype", "ema") != "ema") + 2 * int(cfg.get("assignment", "greedy") != "greedy"))
    return (metrics["f1"], metrics["recall"], metrics["precision"],
            -metrics["fp"], -complexity, cfg.get("threshold", MATCH_THRESHOLD),
            cfg.get("margin", MATCH_MARGIN), cfg["name"])


def choose_candidate(rows, precision_floor=0.9):
    feasible = [r for r in rows if r["dev"]["true_pairs"] > 0 and
                r["dev"]["predicted_pairs"] > 0 and
                r["dev"]["precision"] is not None and
                r["dev"]["precision"] >= precision_floor]
    if not feasible:
        return None
    return max(feasible, key=_config_rank)


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(features_dir, oracle_dir, data_dir, output_dir, seed=1421,
        precision_floor=0.90, grid="standard"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, gt, by_frame = read_features(features_dir, data_dir, oracle_dir)
    if len(manifest["frame_ids"]) < 20:
        raise ValueError("Need >=20 annotated frame sets for calibration; run the 100-frame extraction")
    dev_ids, heldout_ids = partition_ids(by_frame, seed=seed)
    baseline_pred, labels = baseline_replay(by_frame)
    baseline_dev = score_subset(baseline_pred, labels, gt, dev_ids)
    # Fail closed if earlier Step 14.20's verified production replay disagrees.
    existing_diag = Path(features_dir) / "diagnostics.json"
    if existing_diag.is_file():
        prior = json.loads(existing_diag.read_text(encoding="utf-8"))
        prior_input = prior.get("input", {})
        if prior_input.get("ground_truth_sha256") != manifest["ground_truth_sha256"] or \
                prior_input.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
            raise ValueError("Previous diagnostic input fingerprint differs from cached features")
        prior_counts = prior.get("association_gates", {}).get("cross_camera_pair_totals", {})
        all_ids = dev_ids | heldout_ids
        replay_counts = score_subset(baseline_pred, labels, gt, all_ids)
        for key, previous in (("true_pairs", prior_counts.get("true_gt_pairs")),
                              ("predicted_pairs", prior_counts.get("predicted_pairs")),
                              ("tp", prior_counts.get("correct_pairs"))):
            if previous is None or replay_counts[key] != previous:
                raise ValueError(f"Frozen-feature baseline parity failed for {key}: "
                                 f"{replay_counts[key]} != {previous}")
    baseline_heldout = None
    base_row = {"config": asdict(Policy("deployed_baseline")), "dev": baseline_dev}
    rows = [base_row]
    policies = candidates(grid)
    for ix, policy in enumerate(policies[1:], 1):
        preds, policy_labels, decisions = replay(by_frame, policy)
        if policy_labels != labels:
            raise ValueError("Identity labels/ordering changed between policies")
        row = {"config": asdict(policy), "dev": score_subset(preds, labels, gt, dev_ids),
               "identity_decisions": decisions}
        rows.append(row)
        print(f"[{ix}/{len(policies)-1}] {policy.name}: dev F1={row['dev']['f1']:.4f}, "
              f"precision={row['dev']['precision']}, recall={row['dev']['recall']}", flush=True)
    selected = choose_candidate(rows, precision_floor)
    selected_policy = Policy(**selected["config"]) if selected else None
    # Test identities are scored exactly here, AFTER the development-only selection.
    baseline_heldout = score_subset(baseline_pred, labels, gt, heldout_ids)
    selected_heldout = None
    if selected_policy is not None:
        if selected_policy.name == "deployed_baseline":
            selected_pred = baseline_pred
        else:
            selected_pred, _, _ = replay(by_frame, selected_policy)
        selected_heldout = score_subset(selected_pred, labels, gt, heldout_ids)
    source = {
        "frame_ids": manifest["frame_ids"],
        "frames_sha256": manifest["ground_truth_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "embeddings_sha256": manifest["features_sha256"],
        "seed": seed,
        "development_ids": sorted(dev_ids),
        "heldout_ids": sorted(heldout_ids),
        "dev_identity_count": len(dev_ids),
        "heldout_identity_count": len(heldout_ids),
        "dev_true_pairs": baseline_dev["true_pairs"],
        "heldout_true_pairs": baseline_heldout["true_pairs"],
    }
    if not source["dev_true_pairs"] or not source["heldout_true_pairs"]:
        raise ValueError("Identity split has no genuine cross-camera opportunities in one partition; vary the predeclared seed and rerun")
    dump_json(out / "split_manifest.json", source)
    _write_csv(out / "development_grid.csv", [{**r["config"], **{k: r["dev"][k]
                for k in ("tp", "fp", "fn", "true_pairs", "unique_true_gt_identities",
                          "unique_correctly_linked_gt_identities", "unique_wrongly_merged_identity_pairs",
                          "precision", "recall", "f1")}}
                for r in rows],
               ["name", "threshold", "margin", "prototype", "assignment",
                "tp", "fp", "fn", "true_pairs", "unique_true_gt_identities",
                "unique_correctly_linked_gt_identities", "unique_wrongly_merged_identity_pairs",
                "precision", "recall", "f1"])
    result = {
        "experiment": "offline cached OSNet features; unchanged detector, tracker, ReID encoder and production matcher",
        "source": {k: source[k] for k in ("frame_ids", "frames_sha256", "checkpoint_sha256", "embeddings_sha256")},
        "selection": {"precision_floor": precision_floor,
                      "selection_metric": "development F1 among configurations with development precision >= floor",
                      "selected": selected["config"] if selected else None,
                      "note": "No selected candidate means no proposed deployment; dev grid is diagnostic only."},
        "split": {k: source[k] for k in ("seed", "dev_identity_count", "heldout_identity_count",
                                     "dev_true_pairs", "heldout_true_pairs")},
        "baseline": {"dev": baseline_dev, "heldout": baseline_heldout},
        "selected": {"dev": selected["dev"], "heldout": selected_heldout} if selected else None,
        "comparability": {
            "gallery": "All annotated identities remain in the gallery for both splits, including heldout persons as unlabeled distractors during development scoring.",
            "identity_disjoint_labels": True,
            "shared_scene_limitation": "Development and heldout people share frames and camera conditions. This is NOT a separate-scene/dataset generalization test.",
            "source_tracking": "Perfect camera-local tracks and GT crop boxes; global labels used only in scoring, never association.",
            "baseline_reference": "Baseline uses unmodified src.reid.CrossCameraMatcher and unchanged frozen features.",
            "alternative_scope": "Threshold/margin calibration, camera-balanced prototypes and frame-local Hungarian assignment are offline variants; none affect live RTSP.",
            "metrics": "Per-frame cross-camera person pairs; not TrackEval IDF1/HOTA or identity-level independent samples.",
            "precision": "Precision is conditional on annotated ROI; projected ROI mask not verified.",
        },
    }
    dump_json(out / "comparison.json", result)
    print("Selected development configuration:", selected_policy.name if selected_policy else "NONE")
    print("Heldout baseline:", baseline_heldout)
    print("Heldout selected:", selected_heldout)
    return result


def report(output_dir):
    path = Path(output_dir) / "comparison.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    print("Selection:", result["selection"])
    for k in ("baseline", "selected"):
        item = result.get(k)
        if item:
            print(k, "heldout:", {x: item["heldout"][x] for x in
                  ("tp", "fp", "fn", "precision", "recall", "f1")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("run")
    a.add_argument("--data-dir", type=Path, default=DATA)
    a.add_argument("--oracle-dir", type=Path, default=DEFAULT_OUT / "oracle_100")
    a.add_argument("--features-dir", type=Path, default=DEFAULT_DIAG)
    a.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    a.add_argument("--seed", type=int, default=1421)
    a.add_argument("--grid", choices=("smoke", "standard"), default="standard")
    a.add_argument("--precision-floor", type=float, default=0.90)
    b = sub.add_parser("report")
    b.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "run":
        if not 0 < args.precision_floor <= 1:
            parser.error("--precision-floor must be in (0,1]")
        run(args.features_dir, args.oracle_dir, args.data_dir, args.output_dir,
            args.seed, args.precision_floor, args.grid)
    else:
        report(args.output_dir)


if __name__ == "__main__":
    main()
