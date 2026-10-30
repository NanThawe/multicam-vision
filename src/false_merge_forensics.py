"""Step 14.22: reconstruct false ReID merges and test temporal confirmation offline.

Uses the cached, checksum-verified Step 14.20 OSNet features and Step 14.21's
experimental matcher. NO production-model, tracker, or threshold changes.

  python -m src.false_merge_forensics run
  python -m src.false_merge_forensics report
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .association_ablation import (
    DEFAULT_OUTPUT as PREVIOUS_OUT, ExperimentalMatcher, Policy,
    build_tracks, score_subset, partition_ids,
)
from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json
from .reid_diagnostics import DEFAULT_DIAG, FrozenEncoder, read_features
from .reid import unit

DEFAULT_OUTDIR = DEFAULT_OUT / "false_merge_forensics_100"


def write_csv(path, entries, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def local_label_map(tracks, labels):
    """GT labels for *audit only*; never passed into matcher._score / update."""
    mapping = {}
    for fid, cams in tracks.items():
        for cam in CAMERAS:
            if len(cams[cam]) != len(labels[fid][cam]):
                raise ValueError("Different numbers of GT labels and oracle tracks")
            for obs, pid in zip(cams[cam], labels[fid][cam]):
                key = (cam, int(obs["track_id"]))
                if key in mapping and mapping[key] != int(pid):
                    raise ValueError("Oracle camera-local track changed GT person")
                mapping[key] = int(pid)
    return mapping


class TracedMatcher(ExperimentalMatcher):
    """Read-only audit of Step 14.21 selected policy's assignment decisions."""

    def __init__(self, policy, track_gt):
        super().__init__(policy)
        self.track_gt = track_gt
        self.fid = None
        self.audit = []

    def update(self, fid, tracks, feature_lookup):
        self.fid = int(fid)
        return super().update(fid, tracks, feature_lookup)

    def _record_identity(self, gid, cam, person, state):
        prior = self.globals[gid]
        existing = dict(prior["cameras"])
        candidate_scores = self._score(state["feature"], cam)
        score = float(np.dot(state["feature"], prior["feature"])) if existing else None
        gt_pid = self.track_gt[(cam, int(person["track_id"]))]
        other_gt_ids = sorted({self.track_gt[(c, tid)] for c, tid in existing.items()})
        wrong = bool(existing) and bool(other_gt_ids) and any(p != gt_pid for p in other_gt_ids)
        self.audit.append({
            "frame_id": self.fid, "camera": cam, "track_id": int(person["track_id"]),
            "ground_truth_person": gt_pid, "global_id": int(gid),
            "decision": "match_existing" if existing else "new_identity",
            "prior_cameras": ",".join(sorted(existing)),
            "prior_gt_persons": ",".join(map(str, other_gt_ids)),
            "match_cosine": round(score, 6) if score is not None else None,
            "best_available_cosine_at_assignment": round(candidate_scores[0][0], 6) if candidate_scores else None,
            "runner_up_cosine_at_assignment": round(candidate_scores[1][0], 6) if len(candidate_scores) > 1 else None,
            "wrong_gt_merge_at_assignment": int(wrong),
        })
        return super()._record_identity(gid, cam, person, state)


def traced_replay(by_frame, policy):
    tracks, labels = build_tracks(by_frame)
    matcher = TracedMatcher(policy, local_label_map(tracks, labels))
    lookup = FrozenEncoder(by_frame).lookup
    predictions = {}
    for fid, cameras in tracks.items():
        predictions[fid] = matcher.update(fid, cameras, lookup)
    return predictions, labels, matcher.audit


def false_pair_timeline(predictions, labels):
    """Every false *frame-pair observation*, not one unique mistaken decision."""
    records = []
    for fid, cameras in predictions.items():
        for i, cam_a in enumerate(CAMERAS):
            for cam_b in CAMERAS[i+1:]:
                for pid_a, gid_a in zip(labels[fid][cam_a], cameras[cam_a]):
                    if gid_a is None:
                        continue
                    for pid_b, gid_b in zip(labels[fid][cam_b], cameras[cam_b]):
                        if gid_a == gid_b and gid_b is not None and pid_a != pid_b:
                            records.append({"frame_id": int(fid), "camera_pair": f"{cam_a}-{cam_b}",
                                            "global_id": int(gid_a), "gt_person_a": int(pid_a),
                                            "gt_person_b": int(pid_b),
                                            "person_pair": f"{min(pid_a, pid_b)}:{max(pid_a, pid_b)}"})
    return records


def reconstruct_cases(rows, decisions):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["person_pair"], row["global_id"])].append(row)
    cases = []
    for (pair, gid), observations in sorted(groups.items()):
        observations.sort(key=lambda r: r["frame_id"])
        relevant = [d for d in decisions if d["global_id"] == gid and d["wrong_gt_merge_at_assignment"]]
        first_wrong_assignment = min(relevant, key=lambda d: d["frame_id"]) if relevant else None
        cases.append({
            "wrong_gt_pair": pair,
            "global_id": gid,
            "false_pair_observations": len(observations),
            "first_observed_frame": observations[0]["frame_id"],
            "last_observed_frame": observations[-1]["frame_id"],
            "camera_pairs": dict(sorted(Counter(x["camera_pair"] for x in observations).items())),
            "first_wrong_assignment": first_wrong_assignment,
            "note": "An association error may create multiple false frame-pair observations."
        })
    return sorted(cases, key=lambda x: (-x["false_pair_observations"], x["wrong_gt_pair"]))


@dataclass(frozen=True)
class Guard:
    name: str
    independent_votes: int
    require_reciprocal: bool = False
    max_pending_batches: int = 15


class EvidenceMatcher(ExperimentalMatcher):
    """Conservative *offline* confirmation added to the selected 14.21 matcher.

    A vote can only occur on a newly encoded crop (not a reused cached vector).
    A vote is valid if the same candidate ID wins threshold+margin tests on
    successive fresh encodings. Until confirmed, no cross-camera ID is emitted.
    A rejected/expired pending track gets its own ID; retroactive ID edits never
    occur. Optional reciprocal filter is mutual rank-1 between pending local
    tracks and the current per-camera available global-ID gallery.
    """

    def __init__(self, policy, guard):
        if policy.assignment != "batch_hungarian":
            raise ValueError("Evidence confirmation expects batch_hungarian base policy")
        if guard.independent_votes < 1 or guard.max_pending_batches < 1:
            raise ValueError("Invalid evidence guard")
        super().__init__(policy)
        self.guard = guard
        self.pending = {}

    def _finalize(self, gid, cam, person, state, matched):
        if gid is None:
            gid = self._new_global(state["feature"])
            self.decisions["separate_identity"] += 1
        else:
            self.decisions["cross_camera_match"] += 1
            self.decisions["confirmed_matches"] += 1
        self.pending.pop((cam, person["track_id"]), None)
        self._record_identity(gid, cam, person, state)
        self._update_global(gid, cam, state)

    def _batch_camera(self, cam, people):
        waiting = []
        for person in people:
            state = self.local.get((cam, person["track_id"]))
            if state is None or state["count"] < 3:
                continue
            if state["global_id"] is not None:
                self._update_global(state["global_id"], cam, state)
            else:
                waiting.append((person, state))
        if not waiting:
            return
        available = [gid for gid, identity in sorted(self.globals.items())
                     if cam not in identity["cameras"]]
        scores = (np.stack([state["feature"] for _, state in waiting]) @
                  np.stack([self.globals[gid]["feature"] for gid in available]).T
                  if available else np.empty((len(waiting), 0), dtype=np.float32))
        confirmations = []
        for i, (person, state) in enumerate(waiting):
            key = cam, person["track_id"]
            ev = self.pending.setdefault(key, {"first_seen": self.batch_index,
                                               "candidate": None, "votes": 0,
                                               "last_vote_encode": -1})
            # Do not count old embeddings as fresh evidence.
            refreshed = state["last_encoded"] == self.batch_index
            if not available:
                self.decisions["no_candidate"] += 1
                self._finalize(None, cam, person, state, matched=False)
                continue
            order = sorted(range(len(available)), key=lambda j: (-float(scores[i, j]), available[j]))
            j = order[0]
            top = float(scores[i, j])
            second = float(scores[i, order[1]]) if len(order) > 1 else -1.0
            if top < self.policy.threshold or top-second < self.policy.margin:
                self.decisions["gate_rejected"] += 1
                self._finalize(None, cam, person, state, matched=False)
                continue
            # This is a *mutual rank-one* test of current mature unmatched rows
            # and IDs, not an independent second model or ground-truth check.
            if self.guard.require_reciprocal:
                favored_row = min(range(len(waiting)), key=lambda k: (-float(scores[k, j]), k))
                if favored_row != i:
                    self.decisions["reciprocal_rejected"] += 1
                    self._finalize(None, cam, person, state, matched=False)
                    continue
            if refreshed and ev["last_vote_encode"] != state["last_encoded"]:
                if ev["candidate"] == available[j]:
                    ev["votes"] += 1
                else:
                    ev["candidate"] = available[j]
                    ev["votes"] = 1
                ev["last_vote_encode"] = state["last_encoded"]
                self.decisions["fresh_votes"] += 1
            if ev["votes"] >= self.guard.independent_votes and ev["candidate"] == available[j]:
                confirmations.append((top, i, available[j]))
            elif self.batch_index - ev["first_seen"] >= self.guard.max_pending_batches:
                self.decisions["expired_pending"] += 1
                self._finalize(None, cam, person, state, matched=False)
            else:
                self.decisions["deferred_observation"] += 1
        # Same one-to-one rule as a camera-local Hungarian assignment when each
        # row has a single eligible candidate. Highest cosine wins the conflict.
        occupied = set()
        for _, i, gid in sorted(confirmations, key=lambda x: (-x[0], x[1])):
            person, state = waiting[i]
            if state["global_id"] is not None:
                continue
            if gid in occupied or cam in self.globals[gid]["cameras"]:
                self.decisions["assignment_conflict_deferred"] += 1
                continue
            occupied.add(gid)
            self._finalize(gid, cam, person, state, matched=True)


def guarded_replay(by_frame, policy, guard):
    tracks, labels = build_tracks(by_frame)
    lookup = FrozenEncoder(by_frame).lookup
    matcher = EvidenceMatcher(policy, guard)
    predictions = {}
    for fid, cameras in tracks.items():
        predictions[fid] = matcher.update(fid, cameras, lookup)
    return predictions, labels, dict(matcher.decisions)


def check_prior_manifest(previous, manifest, by_frame):
    selected = previous.get("selection", {}).get("selected")
    if not selected:
        raise ValueError("Step 14.21 has no selected policy to investigate")
    for key, source_key in (("frames_sha256", "ground_truth_sha256"),
                            ("checkpoint_sha256", "checkpoint_sha256"),
                            ("embeddings_sha256", "features_sha256")):
        if previous.get("source", {}).get(key) != manifest[source_key]:
            raise ValueError(f"Step 14.21 source fingerprint mismatch: {key}")
    if selected != {"name": "camera_balanced_batch_hungarian_t80_m05", "threshold": 0.8,
                    "margin": 0.05, "prototype": "camera_balanced", "assignment": "batch_hungarian"}:
        raise ValueError("Step 14.21 selected policy differs from expected diagnostic configuration")
    dev, heldout = partition_ids(by_frame, previous.get("split", {}).get("seed", 1421))
    if len(dev) != previous["split"]["dev_identity_count"] or len(heldout) != previous["split"]["heldout_identity_count"]:
        raise ValueError("Step 14.21 identity partition differs from prior results")
    return Policy(**selected), dev, heldout


def verify_replay(predictions, labels, gt, previous, dev, heldout):
    metrics = {"development": score_subset(predictions, labels, gt, dev),
               "previously_inspected_heldout": score_subset(predictions, labels, gt, heldout)}
    for split, prior_name in (("development", "dev"), ("previously_inspected_heldout", "heldout")):
        expected = previous["selected"][prior_name]
        for key in ("tp", "fp", "true_pairs", "predicted_pairs"):
            if metrics[split][key] != expected[key]:
                raise ValueError(f"Selected-policy replay mismatch: {split} {key}: "
                                 f"{metrics[split][key]} != {expected[key]}")
    return metrics


def run(features_dir, oracle_dir, data_dir, prior_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = Path(prior_dir) / "comparison.json"
    if not prior_path.is_file():
        raise FileNotFoundError(f"Missing Step 14.21 comparison: {prior_path}")
    previous = json.loads(prior_path.read_text(encoding="utf-8"))
    manifest, gt, by_frame = read_features(features_dir, data_dir, oracle_dir)
    policy, dev, heldout = check_prior_manifest(previous, manifest, by_frame)
    selected_preds, labels, decisions = traced_replay(by_frame, policy)
    original_metrics = verify_replay(selected_preds, labels, gt, previous, dev, heldout)
    false_rows = false_pair_timeline(selected_preds, labels)
    cases = reconstruct_cases(false_rows, decisions)
    guard_list = [Guard("fresh2", 2), Guard("fresh3", 3),
                  Guard("reciprocal_fresh2", 2, require_reciprocal=True)]
    tested = []
    for guard in guard_list:
        preds, gt_labels, counts = guarded_replay(by_frame, policy, guard)
        if gt_labels != labels:
            raise ValueError("Ground-truth observation ordering changed")
        dev_scores = score_subset(preds, labels, gt, dev)
        heldout_scores = score_subset(preds, labels, gt, heldout)
        tested.append({"guard": guard.__dict__, "development": dev_scores,
                       "previously_inspected_heldout": heldout_scores,
                       "decision_counters": counts,
                       "false_merge_cases": reconstruct_cases(false_pair_timeline(preds, labels), [])})
        print(f"{guard.name}: dev TP/FP={dev_scores['tp']}/{dev_scores['fp']}, "
              f"previously-inspected heldout TP/FP={heldout_scores['tp']}/{heldout_scores['fp']}", flush=True)
    summary = {
        "purpose": "forensic reconstruction and predeclared temporal confirmation counterfactuals",
        "source": {"frames_sha256": manifest["ground_truth_sha256"],
                   "embeddings_sha256": manifest["features_sha256"],
                   "checkpoint_sha256": manifest["checkpoint_sha256"],
                   "previous_comparison_path": str(prior_path)},
        "selected_policy": policy.__dict__,
        "selected_policy_replay_parity": True,
        "selected_policy_scores": original_metrics,
        "false_pair_observations": len(false_rows),
        "distinct_wrong_person_pairs": len({r["person_pair"] for r in false_rows}),
        "false_merge_cases": cases,
        "counterfactuals": tested,
        "interpretation_limits": [
            "All runs use the SAME annotated 100 frames and pretrained cached embeddings; no new generalization test.",
            "Previously inspected held-out identities are FOR DIAGNOSTICS ONLY, not a blind evaluation.",
            "Guard votes count distinct refreshed encoder observations, not repeated reuse of one embedding.",
            "Reciprocal rank-1 is defined only within the current camera's unmatched mature rows and existing identity gallery.",
            "Deferred tracks have no global IDs until confirmed or assigned separate IDs; precision can rise while recall falls.",
            "Oracle ground-truth boxes and perfect camera-local tracks; no production CPU, RTSP, identity-event correctness or official IDF1/HOTA.",
            "GT labels only audit and score, never select association candidates.",
            "One incorrect association can produce repeated incorrect frame-pair observations; count distinct errors separately.",
            "Do not change production matcher on these results; validate on new independent data before deployment.",
        ]
    }
    dump_json(output_dir / "comparison.json", summary)
    dump_json(output_dir / "false_merge_cases.json", {"cases": cases, "count": len(cases)})
    write_csv(output_dir / "false_merge_timeline.csv", false_rows,
              ["frame_id", "camera_pair", "global_id", "gt_person_a", "gt_person_b", "person_pair"])
    write_csv(output_dir / "association_decisions.csv", decisions,
              ["frame_id", "camera", "track_id", "ground_truth_person", "global_id", "decision",
               "prior_cameras", "prior_gt_persons", "match_cosine",
               "best_available_cosine_at_assignment", "runner_up_cosine_at_assignment",
               "wrong_gt_merge_at_assignment"])
    write_csv(output_dir / "counterfactuals.csv", [
        {"name": "step14_21_selected", "split": split, **{k: metrics[k] for k in
         ("tp", "fp", "fn", "true_pairs", "precision", "recall", "f1",
          "unique_wrongly_merged_identity_pairs")}}
        for split, metrics in original_metrics.items()
    ] + [
        {"name": c["guard"]["name"], "split": split,
         **{k: metrics[k] for k in ("tp", "fp", "fn", "true_pairs", "precision", "recall", "f1",
                                   "unique_wrongly_merged_identity_pairs")}}
        for c in tested for split, metrics in
        (("development", c["development"]), ("previously_inspected_heldout", c["previously_inspected_heldout"]))
    ], ["name", "split", "tp", "fp", "fn", "true_pairs", "precision", "recall", "f1",
        "unique_wrongly_merged_identity_pairs"])
    print(f"Reconstructed {len(cases)} global-ID false merge cases from "
          f"{len(false_rows)} false pair-frame observations")
    return summary


def report(output_dir):
    data = json.loads((Path(output_dir) / "comparison.json").read_text(encoding="utf-8"))
    print("Selected 14.21 replay parity:", data["selected_policy_replay_parity"])
    print("False pair-frame observations:", data["false_pair_observations"])
    print("Unique wrong-person pairs:", data["distinct_wrong_person_pairs"])
    for row in data["counterfactuals"]:
        print(row["guard"]["name"], "dev:", row["development"]["tp"], row["development"]["fp"],
              "previous heldout:", row["previously_inspected_heldout"]["tp"],
              row["previously_inspected_heldout"]["fp"])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="Diagnose selected merge and test 3 offline evidence guards")
    r.add_argument("--features-dir", type=Path, default=DEFAULT_DIAG)
    r.add_argument("--oracle-dir", type=Path, default=DEFAULT_OUT / "oracle_100")
    r.add_argument("--data-dir", type=Path, default=DATA)
    r.add_argument("--prior-dir", type=Path, default=PREVIOUS_OUT)
    r.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    p = commands.add_parser("report")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.features_dir, args.oracle_dir, args.data_dir, args.prior_dir, args.output_dir)
    else:
        report(args.output_dir)


if __name__ == "__main__":
    main()
