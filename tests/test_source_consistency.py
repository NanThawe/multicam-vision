"""Synthetic unit tests for Step 14.25 (no annotated WILDTRACK files needed)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import source_consistency as s
from src.association_ablation import Policy, build_tracks, replay


P = Policy('camera_balanced_batch_hungarian_t80_m05', .8, .05,
           'camera_balanced', 'batch_hungarian')
BOX_A = (20., 20., 60., 120.)
BOX_B = (200., 20., 240., 120.)


def vector(x, y):
    v = np.asarray([x, y, 0.01], dtype=np.float32)
    return v / np.linalg.norm(v)


def frames(scenario='impostor'):
    """Source has multiple refreshes and a view change before C2 arrives."""
    out = {}
    for fid in range(0, 100, 5):
        a = vector(1, 0) if (scenario == 'genuine' or fid < 35) else vector(0, 1)
        out[fid] = {'C1': [(1, BOX_A, a)], 'C2': [], 'C3': [], 'C4': []}
        if fid >= 45:
            if scenario == 'genuine':
                incoming = vector(1, 0)
                who = 1
            else:
                incoming = vector(.9, .4359)
                who = 2
            out[fid]['C2'] = [(who, BOX_B, incoming)]
    return out


class SourceGuardTests(unittest.TestCase):
    def test_fresh_history_only_includes_actual_encoding(self):
        matcher, _, _ = s.replay_source(frames('genuine'), P, None)
        fresh = matcher.fresh[('C1', 1)]
        self.assertLess(len(fresh), 20)
        self.assertEqual(len(fresh), len({x[0] for x in fresh}))
        self.assertEqual([x[0] for x in fresh[:3]], [0, 1, 2])
        self.assertEqual(len(matcher.fresh[('C2', 1)]), 4)

    def test_instrumented_no_guard_matches_original(self):
        for scenario in ('impostor', 'genuine'):
            orig, labels, _ = replay(frames(scenario), P)
            matcher, actual, got_labels = s.replay_source(frames(scenario), P, None)
            self.assertEqual(orig, actual)
            self.assertEqual(labels, got_labels)
            self.assertGreater(len(matcher.attempts), 0)

    def test_latest_source_guard_vetoes_wrong_merge(self):
        original, _, _ = s.replay_source(frames('impostor'), P, None)
        guard, predictions, labels = s.replay_source(
            frames('impostor'), P, s.SourceGuard('test', 'latest_pair', .6))
        self.assertGreater(original.decisions['cross_camera_match'], 0)
        self.assertEqual(guard.decisions['cross_camera_match'], 0)
        self.assertGreater(guard.decisions['source_veto'], 0)
        self.assertTrue(any(a['baseline_gates_pass'] and not a['source_pass']
                            for a in guard.attempts))
        # Different GT people have different assigned global IDs at maturity.
        self.assertNotEqual(predictions[60]['C1'], predictions[60]['C2'])

    def test_guard_keeps_clear_genuine(self):
        guard, pred, _ = s.replay_source(frames('genuine'), P,
            s.SourceGuard('test', 'two_of_three', .8))
        self.assertGreater(guard.decisions['cross_camera_match'], 0)
        self.assertEqual(pred[60]['C1'], pred[60]['C2'])

    def test_gate_source_history_no_identity_gt_used(self):
        guard = s.SourceConsistencyMatcher(P, s.SourceGuard('test', 'latest_pair', .6))
        self.assertFalse(hasattr(guard, 'track_gt'))
        guard.globals[1] = {'feature': vector(1, 0), 'cameras': {'C1': 2}, 'per_camera': {}}
        # No source embedding: fail closed.
        self.assertFalse(guard.source_evidence(1, 'C2', 3, vector(1, 0))[0])
        guard.fresh[('C1', 2)] = [(2, vector(1, 0))]
        guard.fresh[('C2', 3)] = [(2, vector(1, 0))]
        self.assertTrue(guard.source_evidence(1, 'C2', 3, vector(1, 0))[0])
        # Unguarded parity must not depend on history completeness.
        unguarded = s.SourceConsistencyMatcher(P, None)
        unguarded.globals[1] = guard.globals[1]
        self.assertTrue(unguarded.source_evidence(1, 'C2', 3, vector(1, 0))[0])

    def test_source_votes_count_distinct_encoded_observations(self):
        guard = s.SourceConsistencyMatcher(P,
            s.SourceGuard('test', 'two_of_three', .70))
        guard.globals[1] = {'feature': vector(1, 0), 'cameras': {'C1': 2}, 'per_camera': {}}
        guard.fresh[('C1', 2)] = [(0, vector(1, 0)), (2, vector(.99, .1)),
                                  (7, vector(0, 1))]
        guard.fresh[('C2', 3)] = [(8, vector(1, 0))]
        accepted, evidence = guard.source_evidence(1, 'C2', 3, vector(1, 0))
        self.assertTrue(accepted)
        self.assertEqual(evidence['C1']['votes'], 2)
        guard.fresh[('C1', 2)] = [(0, vector(0, 1)), (2, vector(.65, .76)),
                                  (7, vector(1, 0))]
        accepted, evidence = guard.source_evidence(1, 'C2', 3, vector(1, 0))
        self.assertFalse(accepted)
        self.assertEqual(evidence['C1']['votes'], 1)

    def test_source_duplicate_batch_detected(self):
        m = s.SourceConsistencyMatcher(P,
            s.SourceGuard('test', 'two_of_three', .6))
        m.globals[1] = {'feature': vector(1, 0), 'cameras': {'C1': 2}, 'per_camera': {}}
        m.fresh[('C1', 2)] = [(2, vector(1, 0)), (2, vector(1, 0))]
        m.fresh[('C2', 3)] = [(3, vector(1, 0))]
        with self.assertRaisesRegex(ValueError, 'Repeated source refresh'):
            m.source_evidence(1, 'C2', 3, vector(1, 0))

    def test_invalid_guard_fails(self):
        for bad in [s.SourceGuard('x', 'nope', .5),
                    s.SourceGuard('x', 'two_of_three', 1.5),
                    s.SourceGuard('x', 'two_of_three', .5, 1, 2)]:
            with self.assertRaises(ValueError):
                s.SourceConsistencyMatcher(P, bad)
        with self.assertRaises(ValueError):
            s.SourceConsistencyMatcher(Policy('unsupported'), None)

    def test_duplicate_crop_key_fails(self):
        m = s.SourceConsistencyMatcher(P, None)
        sample = {c: [] for c in s.CAMERAS}
        sample['C1'] = [{'track_id': 1, 'xyxy': BOX_A},
                        {'track_id': 2, 'xyxy': BOX_A}]
        with self.assertRaisesRegex(ValueError, 'Ambiguous crop'):
            m.update(0, sample, lambda x: vector(1, 0))

    def test_step14_22_report_schema_reference_scores(self):
        """Regression: Step 14.22 has selected_policy_scores, not selected."""
        prior = {'selected_policy_scores': {
            'development': {'tp': 120, 'fp': 0, 'true_pairs': 2734,
                            'predicted_pairs': 120},
            'previously_inspected_heldout': {'tp': 22, 'fp': 18,
                                            'true_pairs': 1672,
                                            'predicted_pairs': 40},
        }}
        dev_expected = prior['selected_policy_scores']['development']
        held_expected = prior['selected_policy_scores']['previously_inspected_heldout']
        with patch.object(s, 'score_subset', side_effect=[dev_expected, held_expected]):
            result = s.verify_step14_22_replay({}, {}, {}, prior, {1}, {2})
        self.assertEqual(result['development']['tp'], 120)
        self.assertEqual(result['previously_inspected_heldout']['fp'], 18)

    def test_step14_22_replay_rejects_reference_metric_drift(self):
        prior = {'selected_policy_scores': {
            'development': {'tp': 120, 'fp': 0, 'true_pairs': 2734,
                            'predicted_pairs': 120},
            'previously_inspected_heldout': {'tp': 22, 'fp': 18,
                                            'true_pairs': 1672,
                                            'predicted_pairs': 40},
        }}
        dev_wrong = dict(prior['selected_policy_scores']['development'], tp=119)
        with patch.object(s, 'score_subset', side_effect=[dev_wrong,
                        prior['selected_policy_scores']['previously_inspected_heldout']]):
            with self.assertRaisesRegex(ValueError, r'development\.tp'):
                s.verify_step14_22_replay({}, {}, {}, prior, {1}, {2})

    def test_step14_22_reference_requires_correct_report_type(self):
        with self.assertRaisesRegex(ValueError, 'selected_policy_scores'):
            s.verify_step14_22_replay({}, {}, {}, {'selected': {}}, {1}, {2})

    def test_csv_and_report_helpers(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'attempt.csv'
            s.save_rows(f, [{'x': 1, 'z': 2}], ['x'])
            self.assertEqual(f.read_text().splitlines(), ['x', '1'])
            out = Path(td) / 'report'
            out.mkdir()
            (out / 'comparison.json').write_text(json.dumps({
                'baseline_parity': True,
                'baseline_scores': {'development': {'tp': 0, 'fp': 0},
                    'previously_inspected_heldout': {'tp': 0, 'fp': 0}},
                'experiments': []}))
            self.assertTrue(s.print_report(out)['baseline_parity'])

    def test_scores_count_fp_even_when_split_mixed(self):
        gt = {0: {'C1': [{'gt_id': 1, 'xyxy': BOX_A}],
                  'C2': [{'gt_id': 2, 'xyxy': BOX_B}], 'C3': [], 'C4': []}}
        labels = {0: {'C1': [1], 'C2': [2], 'C3': [], 'C4': []}}
        predictions = {0: {'C1': [3], 'C2': [3], 'C3': [], 'C4': []}}
        result = s.score_subset(predictions, labels, gt, {2})
        self.assertEqual(result['fp'], 1)
        self.assertEqual(result['precision'], 0.0)


if __name__ == '__main__':
    unittest.main()
