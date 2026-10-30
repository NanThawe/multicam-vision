"""Step 14.25: offline source-identity appearance consistency diagnostics.

Existing Step 14.21 experimental matcher is unchanged. Tests whether incoming
appearance corroborates *individual fresh features* from an already-existing
identity, rather than just its accumulated prototype. Does not train or deploy.

  python -m src.source_consistency run
  python -m src.source_consistency report
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .association_ablation import (ExperimentalMatcher, Policy, build_tracks,
                                   partition_ids, score_subset)
from .false_merge_forensics import (traced_replay, false_pair_timeline,
                                    reconstruct_cases, local_label_map)
from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json
from .reid_diagnostics import DEFAULT_DIAG, FrozenEncoder, read_features
from .reid import unit

DEFAULT_PREVIOUS = DEFAULT_OUT / 'false_merge_forensics_100'
DEFAULT_OUTPUT = DEFAULT_OUT / 'source_consistency_100'


@dataclass(frozen=True)
class SourceGuard:
    name: str
    rule: str  # latest_pair or two_of_three
    cosine_floor: float
    history_count: int = 3
    min_agree: int = 2


GUARDS = (
    SourceGuard('latest_pair_t60', 'latest_pair', 0.60),
    SourceGuard('source_2of3_t60', 'two_of_three', 0.60),
    SourceGuard('source_2of3_t70', 'two_of_three', 0.70),
)


class SourceConsistencyMatcher(ExperimentalMatcher):
    """Retains original candidate-ranking/threshold/Hungarian semantics.

    Adds a post-eligibility veto ONLY to the original top-ranked candidate.
    No GT identity is used in feature processing, gallery, scoring or vetoes.
    Each source sample is from a distinct encoder refresh, never from a cache
    hit. If a source has insufficient fresh evidence the veto fails closed.
    """
    def __init__(self, policy, guard=None):
        if policy.prototype != 'camera_balanced' or policy.assignment != 'batch_hungarian':
            raise ValueError('Requires selected Step 14.21 camera-balanced Hungarian policy')
        if guard is not None and (guard.rule not in ('latest_pair', 'two_of_three')
                                  or not -1 <= guard.cosine_floor <= 1
                                  or guard.history_count < 2 or guard.min_agree < 1
                                  or guard.min_agree > guard.history_count):
            raise ValueError('Invalid source guard')
        super().__init__(policy)
        self.guard = guard
        self.fid = None
        self.fresh = defaultdict(list)  # (cam, local_id) -> [(batch index, L2 vector)]
        self.attempts = []
        self._coords_to_local = {}

    def update(self, fid, tracks, feature_lookup):
        self.fid = int(fid)
        mapping = {}
        for cam in CAMERAS:
            for person in tracks[cam]:
                coords = tuple(map(float, person['xyxy']))
                key = (int(fid), cam, coords)
                if key in mapping:
                    raise ValueError('Ambiguous crop coordinates for fresh-feature history')
                mapping[key] = int(person['track_id'])
        self._coords_to_local = mapping

        # The original matcher expects lookup[key] (not a callable). A proxy
        # preserves that exact contract while recording only true cache misses.
        owner = self
        class ObservedLookup:
            def __getitem__(self, key):
                vec = feature_lookup[key]
                track_id = owner._coords_to_local[key]
                owner.fresh[(key[1], track_id)].append((owner.batch_index, unit(vec).copy()))
                return vec
        # ExperimentalMatcher.update increments batch_index before lookup.
        return super().update(fid, tracks, ObservedLookup())

    def source_evidence(self, gid, cam, incoming_track_id, incoming_local):
        """Return guard-specific evidence from EXISTING camera assignments.

        If a global ID contains multiple source cameras, at least one camera
        must independently corroborate the incoming track. Source history is
        never updated using current incoming observations before this gate.
        """
        incoming = self.fresh.get((cam, incoming_track_id), ())
        by_camera = {}
        for source_cam, source_id in sorted(self.globals[gid]['cameras'].items()):
            if source_cam == cam:
                continue
            hist = self.fresh.get((source_cam, source_id), ())
            # source batches strictly increase: each entry is a unique encoder call
            if any(a[0] >= b[0] for a, b in zip(hist, hist[1:])):
                raise ValueError('Repeated source refresh counted as independent')
            if not hist or not incoming:
                by_camera[source_cam] = {'source_fresh': len(hist), 'available': False,
                                         'pass': False, 'scores': []}
                continue
            if self.guard is None or self.guard.rule == 'latest_pair':
                score = float(np.dot(incoming[-1][1], hist[-1][1]))
                by_camera[source_cam] = {'source_fresh': len(hist), 'available': True,
                                         'pass': self.guard is None or score >= self.guard.cosine_floor,
                                         'scores': [round(score, 6)]}
            else:
                samples = hist[-self.guard.history_count:]
                sims = [float(np.dot(incoming_local, v)) for _, v in samples]
                enough = len(samples) >= self.guard.min_agree
                votes = sum(v >= self.guard.cosine_floor for v in sims)
                by_camera[source_cam] = {'source_fresh': len(hist), 'available': enough,
                                         'pass': enough and votes >= self.guard.min_agree,
                                         'votes': votes,
                                         'scores': [round(v, 6) for v in sims]}
        return (True if self.guard is None else any(s['pass'] for s in by_camera.values()), by_camera)

    def _batch_camera(self, cam, people):
        pending = []
        for person in people:
            state = self.local.get((cam, person['track_id']))
            if state is None or state['count'] < 3:
                continue
            if state['global_id'] is None:
                pending.append((person, state))
            else:
                self._update_global(state['global_id'], cam, state)
        if not pending:
            return
        from scipy.optimize import linear_sum_assignment
        available = [gid for gid, ident in sorted(self.globals.items())
                     if cam not in ident['cameras']]
        n, m = len(pending), len(available)
        costs = np.full((n, m+n), 1e6, dtype=np.float64)
        attempts_index = {}
        for i, (person, state) in enumerate(pending):
            costs[i, m+i] = -self.policy.threshold + 1e-6
            if not available:
                self.decisions['no_candidate'] += 1
                continue
            scores = np.asarray([float(np.dot(state['feature'], self.globals[gid]['feature']))
                                 for gid in available], dtype=np.float64)
            order = np.argsort(-scores, kind='stable')
            j = int(order[0]); top = float(scores[j])
            second = float(scores[int(order[1])]) if m > 1 else -1.
            base_pass = top >= self.policy.threshold and top - second >= self.policy.margin
            source_pass, source = self.source_evidence(available[j], cam,
                                                       int(person['track_id']), state['feature'])
            # For zero margin, multiple original candidates can compete for an
            # assignment. Each eligible alternate must satisfy its own source veto.
            accepted_candidates = []
            for k, gid in enumerate(available):
                score = float(scores[k])
                if score < self.policy.threshold:
                    continue
                if self.policy.margin > 0 and (k != j or top-second < self.policy.margin):
                    continue
                accept, _ = (source_pass, source) if k == j else self.source_evidence(
                    gid, cam, int(person['track_id']), state['feature'])
                if accept:
                    costs[i, k] = -score
                    accepted_candidates.append(gid)
            if base_pass and not source_pass:
                self.decisions['source_veto'] += 1
            row = {'frame_id': self.fid, 'camera': cam, 'track_id': int(person['track_id']),
                   'candidate_gid': int(available[j]),
                   'prototype_cosine': round(top, 6), 'runner_up_cosine': round(second, 6),
                   'baseline_gates_pass': base_pass, 'source_pass': bool(source_pass),
                   'source_details': source, 'eligible_after_source': len(accepted_candidates),
                   'assigned_gid': None, 'decision': None}
            attempts_index[i] = len(self.attempts)
            self.attempts.append(row)
        rows, cols = linear_sum_assignment(costs)
        chosen = {int(i): available[int(j)] for i, j in zip(rows, cols)
                  if j < m and costs[i, j] < 1e5}
        for i, (person, state) in enumerate(pending):
            gid = chosen.get(i)
            if i in attempts_index:
                attempt = self.attempts[attempts_index[i]]
                attempt['assigned_gid'] = int(gid) if gid is not None else None
                attempt['decision'] = 'match_existing' if gid is not None else 'new_identity'
            if gid is None:
                gid = self._new_global(state['feature'])
            else:
                self.decisions['cross_camera_match'] += 1
            self._record_identity(gid, cam, person, state)
            self._update_global(gid, cam, state)


def replay_source(by_frame, policy, guard):
    tracks, labels = build_tracks(by_frame)
    matcher = SourceConsistencyMatcher(policy, guard)
    lookup = FrozenEncoder(by_frame).lookup
    preds = {}
    for fid, cams in tracks.items():
        preds[fid] = matcher.update(fid, cams, lookup)
    return matcher, preds, labels


def score_with_context(matcher, predictions, labels, gt, dev, inspected):
    return {'development': score_subset(predictions, labels, gt, dev),
            'previously_inspected_heldout': score_subset(predictions, labels, gt, inspected),
            'all_identities_descriptive': score_subset(predictions, labels, gt, dev | inspected)}


def verify_step14_22_replay(predictions, labels, gt, prior, dev, inspected):
    """Verify against Step 14.22's *selected_policy_scores* schema.

    The Step 14.21 report uses ``selected.dev`` / ``selected.heldout``;
    the Step 14.22 forensic report instead uses ``selected_policy_scores``
    with ``development`` / ``previously_inspected_heldout``.  Do not pass
    the latter into false_merge_forensics.verify_replay, whose input is
    specifically the earlier Step 14.21 report.
    """
    prior_scores = prior.get('selected_policy_scores')
    if not isinstance(prior_scores, dict):
        raise ValueError('Expected Step 14.22 report with selected_policy_scores; '
                         'check --prior-dir')
    metrics = {
        'development': score_subset(predictions, labels, gt, dev),
        'previously_inspected_heldout': score_subset(predictions, labels, gt, inspected),
    }
    for split in metrics:
        expected = prior_scores.get(split)
        if not isinstance(expected, dict):
            raise ValueError(f'Step 14.22 missing reference scores for {split}')
        for key in ('tp', 'fp', 'true_pairs', 'predicted_pairs'):
            if key not in expected:
                raise ValueError(f'Step 14.22 missing {split}.{key}')
            if metrics[split][key] != expected[key]:
                raise ValueError(f'Step 14.22 replay mismatch: {split}.{key}: '
                                 f'{metrics[split][key]} != {expected[key]}')
    return metrics


def save_rows(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run(features_dir=DEFAULT_DIAG, oracle_dir=DEFAULT_OUT / 'oracle_100', data_dir=DATA,
        prior_dir=DEFAULT_PREVIOUS, output_dir=DEFAULT_OUTPUT):
    out = Path(output_dir)
    prior = json.loads((Path(prior_dir) / 'comparison.json').read_text(encoding='utf-8'))
    manifest, gt, by_frame = read_features(features_dir, data_dir, oracle_dir)
    if len(manifest['frame_ids']) != 100:
        raise ValueError('Use same complete 100 annotated frame sets as Step 14.24')
    if not prior['selected_policy_replay_parity']:
        raise ValueError('Prior reference policy failed parity validation')
    for k, old in [('frames_sha256', prior['source']['frames_sha256']),
                   ('embeddings_sha256', prior['source']['embeddings_sha256']),
                   ('checkpoint_sha256', prior['source']['checkpoint_sha256'])]:
        new = manifest[{'frames_sha256': 'ground_truth_sha256',
                        'embeddings_sha256': 'features_sha256',
                        'checkpoint_sha256': 'checkpoint_sha256'}[k]]
        if old != new:
            raise ValueError(f'Step14.22/14.25 source fingerprint differs: {k}')
    cfg = prior['selected_policy']
    policy = Policy(**cfg)
    if policy.prototype != 'camera_balanced' or policy.assignment != 'batch_hungarian':
        raise ValueError('Unexpected selected policy; stop rather than auditing the wrong matcher')
    dev, inspected = partition_ids(by_frame, seed=1421)
    # Read GT only to score after all matching, NEVER in source evidence decisions.
    base_predictions, labels, traced_decisions = traced_replay(by_frame, policy)
    base_scores = verify_step14_22_replay(base_predictions, labels, gt, prior, dev, inspected)
    old_cases = reconstruct_cases(false_pair_timeline(base_predictions, labels), traced_decisions)
    if old_cases != prior['false_merge_cases']:
        raise ValueError('Baseline wrong-merge reconstruction differs from Step14.22')
    no_guard, parity_predictions, parity_labels = replay_source(by_frame, policy, None)
    if parity_predictions != base_predictions or parity_labels != labels:
        raise ValueError('Source consistency instrumentation changed baseline assignments')
    experiments = []
    attempt_by_guard = {}
    baseline_attempts = no_guard.attempts
    for g in GUARDS:
        matcher, predictions, got_labels = replay_source(by_frame, policy, g)
        if got_labels != labels:
            raise ValueError(f'Label ordering differs under {g.name}')
        metrics = score_with_context(matcher, predictions, labels, gt, dev, inspected)
        false_cases = reconstruct_cases(false_pair_timeline(predictions, labels), [])
        attempt_by_guard[g.name] = matcher.attempts
        experiments.append({'guard': g.__dict__, 'scores': metrics,
                            'decisions': dict(matcher.decisions), 'false_merges': false_cases,
                            'source_attempts': len(matcher.attempts),
                            'source_vetoes': matcher.decisions['source_veto']})
        out.mkdir(parents=True, exist_ok=True)
        save_rows(out / f'attempts_{g.name}.csv', [
            {**{k: v for k, v in x.items() if k != 'source_details'},
             'source_details_json': json.dumps(x['source_details'], sort_keys=True)}
            for x in matcher.attempts],
            ['frame_id', 'camera', 'track_id', 'candidate_gid', 'prototype_cosine',
             'runner_up_cosine', 'baseline_gates_pass', 'source_pass',
             'eligible_after_source', 'assigned_gid', 'decision', 'source_details_json'])
        print(f'{g.name}: dev TP/FP={metrics["development"]["tp"]}/{metrics["development"]["fp"]}, '
              f'previous inspected TP/FP={metrics["previously_inspected_heldout"]["tp"]}/'
              f'{metrics["previously_inspected_heldout"]["fp"]}', flush=True)
    # Original source score at the exact frame/track identity error, which was
    # under experiment selected policy, not the production EMA matcher.
    original = next((r for r in baseline_attempts
                     if r['frame_id'] == 405 and r['camera'] == 'C2'
                     and r['candidate_gid'] == 112 and r['baseline_gates_pass']), None)
    if original is None:
        raise ValueError('Cannot locate original C2 frame 405 candidate; abort causal narrative')
    hard_case = {'frame_id': 405, 'candidate_gid': 112,
                 'source_vs_incoming': original,
                 'counterfactual_attempts': {}}
    # Global ID allocations can change after a veto; filter by frame/camera
    # rather than assuming all counterfactuals still use global ID 112.
    for item in experiments:
        name = item['guard']['name']
        hard_case['counterfactual_attempts'][name] = [
            x for x in attempt_by_guard[name]
            if x['frame_id'] == 405 and x['camera'] == 'C2'
        ]
    baseline = score_with_context(no_guard, parity_predictions, labels, gt, dev, inspected)
    report = {
      'purpose': 'Check source-camera appearance corroboration for Step14.21 experimental association',
      'source': {'frames_sha256': manifest['ground_truth_sha256'],
                 'embeddings_sha256': manifest['features_sha256'],
                 'checkpoint_sha256': manifest['checkpoint_sha256'],
                 'frame_sets': len(manifest['frame_ids'])},
      'base_policy': cfg, 'baseline_parity': True,
      'baseline_scores': baseline,
      'baseline_false_merges': old_cases,
      'baseline_attempts': len(baseline_attempts),
      'hard_case': hard_case,
      'experiments': experiments,
      'limits': [
        'All 100 previously inspected annotated frame sets are reused; no untouched held-out accuracy.',
        'The original false merge belongs to an undeployed Step14.21 matcher, not production EMA.',
        'GT identities are used solely for audit and scoring, not for candidate ranking or source guards.',
        'One wrong global-ID merge can create repeated false frame-pair observations.',
        'Source history counts distinct model encodings, NOT independent camera conditions or independent people.',
        'When a guard vetoes the best candidate, the track gets a new ID; this may lose true cross-camera matches.',
        'This diagnostic does not measure live RTSP FPS, MQTT, IDF1/HOTA or new encoder quality.',
        'A source-consistency guard must be validated on new independent data before deployment.'
      ]
    }
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / 'comparison.json', report)
    save_rows(out / 'comparison.csv', [
       {'configuration': 'unguarded_selected', 'split': split, **{k: score[k] for k in
         ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'unique_correctly_linked_gt_identities',
          'unique_wrongly_merged_identity_pairs')}}
       for split, score in baseline.items()
    ] + [
       {'configuration': ex['guard']['name'], 'split': split, **{k: score[k] for k in
         ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'unique_correctly_linked_gt_identities',
          'unique_wrongly_merged_identity_pairs')}}
       for ex in experiments for split, score in ex['scores'].items()],
      ['configuration', 'split', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
       'unique_correctly_linked_gt_identities', 'unique_wrongly_merged_identity_pairs'])
    dump_json(out / 'hard_case.json', hard_case)
    return report


def print_report(out=DEFAULT_OUTPUT):
    d = json.loads((Path(out) / 'comparison.json').read_text(encoding='utf-8'))
    print('Baseline replay parity:', d['baseline_parity'])
    for split in ('development', 'previously_inspected_heldout'):
        score = d['baseline_scores'][split]
        print('Experimental selected baseline', split, 'TP/FP', score['tp'], score['fp'])
    for ex in d['experiments']:
        s = ex['scores']['previously_inspected_heldout']
        print(ex['guard']['name'], 'TP/FP', s['tp'], s['fp'],
              'precision', s['precision'], 'recall', s['recall'],
              'wrong-person merges', s['unique_wrongly_merged_identity_pairs'])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Replay existing matcher and compare source-consistency guards')
    p.add_argument('--features-dir', type=Path, default=DEFAULT_DIAG)
    p.add_argument('--oracle-dir', type=Path, default=DEFAULT_OUT / 'oracle_100')
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--prior-dir', type=Path, default=DEFAULT_PREVIOUS)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    r = sub.add_parser('report')
    r.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)
    if args.command == 'run':
        run(args.features_dir, args.oracle_dir, args.data_dir, args.prior_dir, args.output_dir)
    else:
        print_report(args.output_dir)


if __name__ == '__main__':
    main()
