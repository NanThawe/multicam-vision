"""Step 14.22 unit/integration tests; no weights, RTSP or network required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import false_merge_forensics as f
from src import association_ablation as a


def vec(x, y):
    return a.unit(np.array([x, y, .01], dtype=np.float32))


def fixture(identical=False, frames=10):
    truth, embeddings = {}, {}
    for k in range(frames):
        fid = 5*k
        truth[fid] = {cam: [] for cam in a.CAMERAS}
        embeddings[fid] = {cam: [] for cam in a.CAMERAS}
        profiles = {
            'C1': [(1, vec(1, 0))],
            'C2': [(2 if identical else 1, vec(1, 0))],
            'C3': [(3, vec(0, 1))],
            'C4': [(4, vec(-1, 0))],
        }
        for cam, people in profiles.items():
            for pid, feature in people:
                box = (float(10*pid), 0., float(10*pid+30), 60.)
                embeddings[fid][cam].append((pid, box, feature))
                truth[fid][cam].append({'gt_id': pid, 'xyxy': list(box)})
    return truth, embeddings


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.policy = a.Policy('camera_balanced_batch_hungarian_t80_m05',
                               0.8, 0.05, 'camera_balanced', 'batch_hungarian')

    def test_traced_replay_matches_unmodified_selected_replay(self):
        _, emb = fixture()
        expected, labels, _ = a.replay(emb, self.policy)
        got, traced_labels, audit = f.traced_replay(emb, self.policy)
        self.assertEqual(got, expected)
        self.assertEqual(labels, traced_labels)
        self.assertTrue(audit)
        self.assertEqual(sum(x['decision'] == 'match_existing' for x in audit), 1)
        self.assertFalse(any(x['wrong_gt_merge_at_assignment'] for x in audit))

    def test_reconstructs_one_wrong_merge_and_repeated_false_observations(self):
        _, emb = fixture(identical=True)
        predictions, labels, decisions = f.traced_replay(emb, self.policy)
        timeline = f.false_pair_timeline(predictions, labels)
        self.assertEqual(len(timeline), 8)
        cases = f.reconstruct_cases(timeline, decisions)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case['wrong_gt_pair'], '1:2')
        self.assertEqual(case['false_pair_observations'], 8)
        self.assertEqual(case['camera_pairs'], {'C1-C2': 8})
        self.assertEqual(case['first_wrong_assignment']['frame_id'], 10)
        self.assertEqual(case['first_wrong_assignment']['prior_gt_persons'], '1')
        self.assertGreaterEqual(case['first_wrong_assignment']['match_cosine'], 0.8)

    def test_no_false_merges_when_correct_identities(self):
        _, emb = fixture()
        preds, labels, decisions = f.traced_replay(emb, self.policy)
        self.assertEqual(f.reconstruct_cases(f.false_pair_timeline(preds, labels), decisions), [])

    def test_case_handles_missing_assignment_trace(self):
        rows = [{'frame_id': 10, 'camera_pair': 'C1-C2', 'global_id': 7,
                 'gt_person_a': 1, 'gt_person_b': 2, 'person_pair': '1:2'}]
        case = f.reconstruct_cases(rows, [])[0]
        self.assertIsNone(case['first_wrong_assignment'])

    def test_gt_map_catches_reused_local_id(self):
        tracks = {0: {'C1': [{'track_id':1}], 'C2': [], 'C3':[], 'C4':[]},
                  5: {'C1': [{'track_id':1}], 'C2': [], 'C3':[], 'C4':[]}}
        labels = {0: {'C1': [1], 'C2': [], 'C3':[], 'C4':[]},
                  5: {'C1': [2], 'C2': [], 'C3':[], 'C4':[]}}
        with self.assertRaisesRegex(ValueError, 'changed GT person'):
            f.local_label_map(tracks, labels)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.policy = a.Policy('base', 0.8, 0.05, 'camera_balanced', 'batch_hungarian')

    def test_single_vote_is_not_enough(self):
        _, emb = fixture(frames=3)
        preds, labels, counts = f.guarded_replay(emb, self.policy, f.Guard('two', 2))
        self.assertIsNotNone(preds[10]['C1'][0])
        self.assertIsNone(preds[10]['C2'][0])
        self.assertEqual(counts.get('confirmed_matches', 0), 0)
        self.assertGreater(counts.get('deferred_observation', 0), 0)

    def test_two_fresh_encodings_confirm(self):
        _, emb = fixture(frames=10)
        preds, _, counts = f.guarded_replay(emb, self.policy, f.Guard('two', 2))
        self.assertEqual(preds[35]['C1'][0], preds[35]['C2'][0])
        self.assertGreaterEqual(counts['confirmed_matches'], 1)

    def test_cached_reuse_does_not_count_as_fresh_vote(self):
        _, emb = fixture(frames=7)
        preds, _, counts = f.guarded_replay(emb, self.policy, f.Guard('two', 2))
        self.assertIsNone(preds[30]['C2'][0])
        self.assertEqual(counts.get('confirmed_matches', 0), 0)

    def test_three_votes_require_more_independent_observations(self):
        _, emb = fixture(frames=14)
        preds, _, counts = f.guarded_replay(emb, self.policy, f.Guard('three', 3))
        self.assertIsNone(preds[35]['C2'][0])
        self.assertEqual(preds[60]['C1'][0], preds[60]['C2'][0])
        self.assertGreaterEqual(counts['confirmed_matches'], 1)

    def test_persistent_impostor_can_still_pass_confirmation(self):
        _, emb = fixture(identical=True, frames=10)
        preds, labels, counts = f.guarded_replay(emb, self.policy, f.Guard('two', 2))
        timeline = f.false_pair_timeline(preds, labels)
        self.assertGreater(len(timeline), 0)
        self.assertGreater(counts['confirmed_matches'], 0)

    def test_one_vote_matches_selected_policy_on_simple_fixture(self):
        _, emb = fixture(frames=10)
        original, _, _ = a.replay(emb, self.policy)
        guarded, _, _ = f.guarded_replay(emb, self.policy, f.Guard('one', 1))
        self.assertEqual(original, guarded)

    def test_mutual_rank_filter_never_uses_gt(self):
        _, emb = fixture(frames=10)
        preds, _, counts = f.guarded_replay(emb, self.policy,
                                            f.Guard('mutual', 2, require_reciprocal=True))
        self.assertEqual(preds[35]['C1'][0], preds[35]['C2'][0])
        self.assertGreater(counts['fresh_votes'], 0)

    def test_invalid_guard_rejected(self):
        with self.assertRaises(ValueError):
            f.EvidenceMatcher(self.policy, f.Guard('invalid', 0))
        with self.assertRaises(ValueError):
            f.EvidenceMatcher(a.Policy('wrong'), f.Guard('two', 2))

    def test_different_gallery_only_creates_separate_ids(self):
        _, emb = fixture(frames=10)
        for fid in emb:
            pid, box, _ = emb[fid]['C2'][0]
            emb[fid]['C2'][0] = (pid, box, vec(.0, -1.))
        preds, _, counts = f.guarded_replay(emb, self.policy, f.Guard('two', 2))
        self.assertNotEqual(preds[35]['C1'][0], preds[35]['C2'][0])
        self.assertEqual(counts.get('confirmed_matches', 0), 0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.truth, self.emb = fixture(frames=20)
        for fid in self.emb:
            box = (200., 0., 230., 60.)
            self.emb[fid]['C4'].append((5, box, vec(-.7, -.7)))
            self.truth[fid]['C4'].append({'gt_id': 5, 'xyxy': list(box)})
        self.policy = a.Policy('camera_balanced_batch_hungarian_t80_m05',
                               0.8, 0.05, 'camera_balanced', 'batch_hungarian')
        self.manifest = {'frame_ids': sorted(self.emb), 'ground_truth_sha256': 'G',
                         'checkpoint_sha256': 'C', 'features_sha256': 'E'}
        self.dev, self.heldout = a.partition_ids(self.emb, seed=1421)
        preds, labels, _ = a.replay(self.emb, self.policy)
        self.prior = {
            'source': {'frames_sha256': 'G', 'checkpoint_sha256': 'C',
                       'embeddings_sha256': 'E'},
            'selection': {'selected': self.policy.__dict__},
            'split': {'seed': 1421, 'dev_identity_count': len(self.dev),
                      'heldout_identity_count': len(self.heldout)},
            'selected': {'dev': a.score_subset(preds, labels, self.truth, self.dev),
                         'heldout': a.score_subset(preds, labels, self.truth, self.heldout)},
        }

    def test_fingerprint_enforced(self):
        self.prior['source']['embeddings_sha256'] = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            f.check_prior_manifest(self.prior, self.manifest, self.emb)

    def test_policy_enforced(self):
        self.prior['selection']['selected']['threshold'] = .70
        with self.assertRaisesRegex(ValueError, 'differs from expected'):
            f.check_prior_manifest(self.prior, self.manifest, self.emb)

    def test_scoring_parity_enforced(self):
        preds, labels, _ = f.traced_replay(self.emb, self.policy)
        self.prior['selected']['heldout']['tp'] += 1
        with self.assertRaisesRegex(ValueError, 'replay mismatch'):
            f.verify_replay(preds, labels, self.truth, self.prior, self.dev, self.heldout)

    def test_complete_run_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp, 'prior')
            prior.mkdir()
            (prior / 'comparison.json').write_text(json.dumps(self.prior), encoding='utf-8')
            with patch.object(f, 'read_features', return_value=(self.manifest, self.truth, self.emb)):
                result = f.run('features', 'oracle', 'data', prior, Path(tmp,'out'))
            self.assertTrue(result['selected_policy_replay_parity'])
            self.assertEqual(len(result['counterfactuals']), 3)
            for name in ('comparison.json', 'false_merge_cases.json', 'false_merge_timeline.csv',
                         'association_decisions.csv', 'counterfactuals.csv'):
                self.assertTrue((Path(tmp, 'out', name)).is_file(), name)
            self.assertEqual(f.report(Path(tmp, 'out'))['source']['checkpoint_sha256'], 'C')


if __name__ == '__main__':
    unittest.main()
