"""Unit and synthetic parity checks for Step 14.21. No OSNet weights required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from src import association_ablation as a


def feature(x, y):
    return a.unit(np.array([x, y, .01], dtype=np.float32))


def synth(ids=(0, 5, 10, 15, 20, 25, 30, 35), include_negative=True):
    truth, emb = {}, {}
    for fid in ids:
        truth[fid] = {cam: [] for cam in a.CAMERAS}
        emb[fid] = {cam: [] for cam in a.CAMERAS}
        for cam, people in {"C1": [1, 2], "C2": [1, 2], "C3": [3], "C4": [3]}.items():
            if not include_negative and cam not in ("C1", "C2"):
                people = []
            for idx, pid in enumerate(people):
                box = (float(60 * idx), 0., float(60 * idx + 40), 70.)
                vec = feature(1, 0) if pid in (1, 3) else feature(0, 1)
                emb[fid][cam].append((pid, box, vec))
                truth[fid][cam].append({"gt_id": pid, "xyxy": list(box)})
    return truth, emb


class PartitionTests(unittest.TestCase):
    def test_split_determinism_and_no_identity_overlap(self):
        _, emb = synth()
        for fid in emb:
            emb[fid]['C4'].append((4, (90., 0., 130., 70.), feature(.1, .9)))
        x, y = a.partition_ids(emb, seed=11)
        x2, y2 = a.partition_ids(emb, seed=11)
        self.assertEqual((x, y), (x2, y2))
        self.assertFalse(x & y)
        self.assertEqual(x | y, {1, 2, 3, 4})

    def test_split_needs_four_ids(self):
        _, emb = synth()
        with self.assertRaises(ValueError):
            a.partition_ids(emb)

    def test_split_fraction_rejected(self):
        _, emb = synth()
        with self.assertRaises(ValueError):
            a.partition_ids(emb, dev_fraction=1)

    def test_split_with_several_ids(self):
        _, emb = synth()
        for fid in emb:
            for pid in range(4, 20):
                emb[fid]['C4'].append((pid, (pid*45., 0., pid*45.+40, 70.), feature(1, 0)))
        dev, test = a.partition_ids(emb, seed=7)
        self.assertEqual(len(dev), round(.6 * 19))
        self.assertEqual(len(test), 19-len(dev))
        self.assertFalse(dev & test)


class ReplayTests(unittest.TestCase):
    def test_deployed_parity_with_ema_greedy(self):
        _, emb = synth()
        expected, labels = a.baseline_replay(emb)
        got, got_labels, _ = a.replay(emb, a.Policy('same', 0.85, 0.05, 'ema', 'greedy'))
        self.assertEqual(expected, got)
        self.assertEqual(labels, got_labels)

    def test_deployed_parity_when_threshold_loose(self):
        _, emb = synth()
        # Both methods should follow reference track ID order and cache warmup.
        p = a.Policy('alt', .2, 0., 'ema', 'greedy')
        got, _, _ = a.replay(emb, p)
        ref = a.CrossCameraMatcher(a.FrozenEncoder(emb), min_observations=3,
                                    threshold=.2, margin=0, refresh_every=5,
                                    batch_across_cameras=True)
        tracks, _ = a.build_tracks(emb)
        for fid, t in tracks.items():
            pred, _ = ref.update(fid, {cam: None for cam in a.CAMERAS}, t)
            self.assertEqual(got[fid], {cam: [p['global_id'] for p in pred[cam]]
                                        for cam in a.CAMERAS})

    def test_similarity_gate_stops_same_person(self):
        _, emb = synth()
        for fid in emb:
            for i, (pid, box, feat) in enumerate(emb[fid]['C2']):
                emb[fid]['C2'][i] = (pid, box, feature(.4, .9) if pid == 1 else feat)
        no_match, _, _ = a.replay(emb, a.Policy('strict', .99, .05))
        permissive, _, _ = a.replay(emb, a.Policy('loose', .3, 0.))
        self.assertNotEqual(no_match, permissive)

    def test_prototype_mean_and_batch_do_not_need_gt_labels(self):
        _, emb = synth()
        for proto, ass in [('camera_balanced', 'greedy'), ('camera_balanced', 'batch_hungarian')]:
            actual, labels, decisions = a.replay(emb, a.Policy('alt', .7, 0, proto, ass))
            self.assertEqual(set(actual), set(emb))
            self.assertEqual(len(actual[20]['C1']), len(labels[20]['C1']))
            self.assertGreater(decisions['new_id'], 0)

    def test_invalid_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown prototype'):
            a.ExperimentalMatcher(a.Policy('bad', prototype='unknown'))
        with self.assertRaisesRegex(ValueError, 'Unknown assignment'):
            a.ExperimentalMatcher(a.Policy('bad', assignment='unknown'))


class ScoringTests(unittest.TestCase):
    def test_simple_tp_and_wrong_merge_with_nontarget(self):
        gt, emb = synth(ids=(0,))
        _, labels = a.build_tracks(emb)
        preds = {0: {c: [None] * len(labels[0][c]) for c in a.CAMERAS}}
        preds[0]['C1'] = [10, 20]
        preds[0]['C2'] = [10, 10]
        dev = a.score_subset(preds, labels, gt, {1})
        self.assertEqual(dev['true_pairs'], 1)
        self.assertEqual(dev['tp'], 1)
        self.assertEqual(dev['fp'], 1)
        self.assertEqual(dev['precision'], .5)
        self.assertEqual(dev['recall'], 1.)
        heldout = a.score_subset(preds, labels, gt, {2})
        self.assertEqual(heldout['true_pairs'], 1)
        self.assertEqual(heldout['tp'], 0)
        self.assertEqual(heldout['fp'], 1)

    def test_no_predictions_precision_not_one(self):
        gt, emb = synth(ids=(0,))
        _, labels = a.build_tracks(emb)
        preds = {0: {c: [None]*len(labels[0][c]) for c in a.CAMERAS}}
        result = a.score_subset(preds, labels, gt, {1})
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['f1'], 0)

    def test_denominator_includes_ineligible_boxes(self):
        gt, emb = synth(ids=(0,))
        gt[0]['C2'].append({'gt_id': 17, 'xyxy':[0,0,2,10]})
        gt[0]['C1'].append({'gt_id': 17, 'xyxy':[0,0,2,10]})
        _, labels = a.build_tracks(emb)
        preds = {0: {c: [None]*len(labels[0][c]) for c in a.CAMERAS}}
        result = a.score_subset(preds, labels, gt, {17})
        self.assertEqual(result['true_pairs'], 1)
        self.assertEqual(result['tp'], 0)

    def test_length_mismatch_fails(self):
        gt, emb = synth(ids=(0,))
        _, labels = a.build_tracks(emb)
        preds = {0: {c: [] for c in a.CAMERAS}}
        with self.assertRaisesRegex(ValueError, 'lengths differ'):
            a.score_subset(preds, labels, gt, {1})

    def test_candidate_selector_respects_precision_floor(self):
        rows = [
            {'config':{'name':'highrecall'}, 'dev':{'true_pairs':10, 'predicted_pairs':10, 'precision':.5, 'recall':.9, 'f1':.65, 'fp':5}},
            {'config':{'name':'safe'}, 'dev':{'true_pairs':10, 'predicted_pairs':5, 'precision':1., 'recall':.5, 'f1':.67, 'fp':0}},
        ]
        self.assertEqual(a.choose_candidate(rows, .9)['config']['name'], 'safe')
        self.assertIsNone(a.choose_candidate(rows, 1.1))

    def test_standard_grid_contains_expected_baseline(self):
        cs = a.candidates()
        self.assertEqual(cs[0].name, 'deployed_baseline')
        self.assertEqual(len(set(x.name for x in cs)), len(cs))
        self.assertGreater(len(cs), 20)


class IntegrationTest(unittest.TestCase):
    def test_mocked_pipeline_reports_split_and_files(self):
        gt, emb = synth(ids=tuple(range(0, 100, 5)))
        for fid in emb:
            box = (90., 0., 130., 70.)
            emb[fid]['C1'].append((4, box, feature(.1,.9)))
            emb[fid]['C2'].append((4, box, feature(.1,.9)))
            gt[fid]['C1'].append({'gt_id':4,'xyxy':list(box)})
            gt[fid]['C2'].append({'gt_id':4,'xyxy':list(box)})
        manifest = {'frame_ids':sorted(emb), 'ground_truth_sha256':'h1',
                    'checkpoint_sha256':'h2', 'features_sha256':'h3'}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(a, 'read_features', return_value=(manifest, gt, emb)):
                result = a.run('features','oracle','data', temp, seed=50, grid='smoke', precision_floor=.5)
            self.assertEqual(result['split']['dev_identity_count'] + result['split']['heldout_identity_count'], 4)
            self.assertTrue(Path(temp,'comparison.json').exists())
            self.assertTrue(Path(temp,'development_grid.csv').exists())
            self.assertTrue(Path(temp,'split_manifest.json').exists())
            if result['selected']:
                self.assertIn('heldout', result['selected'])
            self.assertIn('heldout', result['baseline'])
            self.assertEqual(a.report(temp)['selection'], result['selection'])


if __name__ == '__main__':
    unittest.main()

class AdditionalSafetyTests(unittest.TestCase):
    def test_duplicate_person_in_camera_rejected(self):
        _, emb = synth(ids=(0,))
        emb[0]['C1'].append(emb[0]['C1'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate GT person'):
            a.build_tracks(emb)

    def test_hungarian_can_allocate_two_competing_candidates(self):
        policy = a.Policy('joint', .6, 0, 'ema', 'batch_hungarian')
        matcher = a.ExperimentalMatcher(policy)
        matcher.globals = {
            1: {'feature': feature(1., 0.), 'cameras': {'C2': 1}, 'per_camera': {}},
            2: {'feature': feature(.8, .6), 'cameras': {'C2': 2}, 'per_camera': {}}
        }
        matcher.next_id = 3
        fa = feature(.95, .31)
        fb = feature(.8, -.6)
        matcher.local = {
            ('C1', 1): {'feature': fa, 'count': 3, 'global_id': None},
            ('C1', 2): {'feature': fb, 'count': 3, 'global_id': None},
        }
        people = [{'track_id':1},{'track_id':2}]
        matcher._batch_camera('C1', people)
        self.assertEqual(set(matcher.local[('C1',x)]['global_id'] for x in (1,2)), {1,2})
        self.assertEqual(matcher.decisions['cross_camera_match'], 2)
        self.assertEqual(matcher.decisions['new_id'], 0)

    def test_existing_diagnostic_parity_mismatch_rejected(self):
        gt, emb = synth(ids=tuple(range(0, 100, 5)))
        # Add a fourth unique person to support identity disjoint splitting.
        for fid in emb:
            b=(90., 0., 130., 70.)
            emb[fid]['C1'].append((4, b, feature(.3,.9)))
            gt[fid]['C1'].append({'gt_id': 4, 'xyxy': list(b)})
        manifest = {'frame_ids': sorted(emb), 'ground_truth_sha256':'h1',
                    'checkpoint_sha256':'h2', 'features_sha256':'h3'}
        with tempfile.TemporaryDirectory() as d:
            fdir=Path(d,'features'); fdir.mkdir()
            (fdir/'diagnostics.json').write_text(json.dumps({
                'input': {'ground_truth_sha256':'h1', 'checkpoint_sha256':'h2'},
                'association_gates': {'cross_camera_pair_totals': {
                    'true_gt_pairs': 999999, 'predicted_pairs': 0, 'correct_pairs': 0}}
            }))
            with patch.object(a, 'read_features', return_value=(manifest,gt,emb)):
                with self.assertRaisesRegex(ValueError, 'baseline parity failed'):
                    a.run(fdir,'oracle','data',Path(d,'output'),grid='smoke')

class SelectedCandidateIntegrationTest(unittest.TestCase):
    def test_selected_candidate_evaluated_on_heldout_only_after_selection(self):
        emb, gt = {}, {}
        eye = np.eye(4, dtype=np.float32)
        for fid in range(0, 100, 5):
            emb[fid] = {c: [] for c in a.CAMERAS}
            gt[fid] = {c: [] for c in a.CAMERAS}
            for cam in ('C1', 'C2'):
                for pid in range(4):
                    box=(float(pid*50),0.,float(pid*50+40),70.)
                    emb[fid][cam].append((pid, box, eye[pid]))
                    gt[fid][cam].append({'gt_id':pid,'xyxy':list(box)})
        manifest={'frame_ids':sorted(emb),'ground_truth_sha256':'h1',
                  'checkpoint_sha256':'h2','features_sha256':'h3'}
        with tempfile.TemporaryDirectory() as d:
            with patch.object(a, 'read_features', return_value=(manifest,gt,emb)):
                result = a.run('features','oracle','data',d,grid='smoke')
            self.assertIsNotNone(result['selection']['selected'])
            self.assertIsNotNone(result['selected']['heldout'])
            self.assertGreater(result['selected']['heldout']['tp'], 0)
            self.assertEqual(result['selected']['heldout']['fp'], 0)
            self.assertEqual(result['baseline']['heldout']['precision'], 1.0)
