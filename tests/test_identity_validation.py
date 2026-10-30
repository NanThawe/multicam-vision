"""Regression tests for Step 14.27 fixed-policy, later-frame validation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import identity_validation as mod
from src.identity_evaluation import source_hash


class IdentityValidationTests(unittest.TestCase):
    def test_full_late_frames_are_predetermined(self):
        f = mod.fixed_frame_ids()
        self.assertEqual(len(f), 60)
        self.assertEqual((f[0], f[-1]), (1400, 1990))
        self.assertEqual(len(set(f)), 60)
        self.assertTrue(all(x % 5 == 0 for x in f))

    def test_pilot_is_fixed_subset(self):
        self.assertEqual(mod.fixed_frame_ids(True), mod.fixed_frame_ids()[:15])

    def test_disjoint_check_accepts_late(self):
        mod.assert_later_frames([1400, 1410], list(range(0, 500, 5)))

    def test_disjoint_check_rejects_overlap(self):
        with self.assertRaises(ValueError):
            mod.assert_later_frames([0, 1400], [0, 5, 10])

    def test_disjoint_check_rejects_unsorted(self):
        with self.assertRaises(ValueError):
            mod.assert_later_frames([1410, 1400], [0, 5, 10])

    def test_disjoint_check_rejects_off_plan(self):
        with self.assertRaises(ValueError):
            mod.assert_later_frames([1395, 1400], [0, 5, 10])

    def test_assets_four_cameras_and_annotations(self):
        paths = mod.get_assets('/tmp/wildtrack', [1400, 1410])
        self.assertEqual(len(paths), 10)
        self.assertEqual(paths[0][0], 'annotations_positions/00001400.json')
        self.assertEqual(paths[4][0], 'Image_subsets/C4/00001400.png')

    def test_missing_assets_fail_with_download_hint(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, 'prepare --download'):
                mod.validate_assets(d, [1400])

    def test_identity_set_reads_gt_without_matcher(self):
        gt = {1400: {'C1': [{'gt_id': 10}], 'C2': [{'gt_id': 11}], 'C3': [], 'C4': []},
              1410: {'C1': [{'gt_id': 11}], 'C2': [], 'C3': [], 'C4': []}}
        self.assertEqual(mod.identities(gt), {10, 11})

    def test_fixed_policy_matches_prior_step25(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d:
            prior = Path(d) / 'comparison.json'
            prior.write_text(json.dumps({'baseline_parity': True,
                'base_policy': asdict(mod.POLICY),
                'experiments': [{'guard': asdict(mod.GUARD)}]}))
            mod.prior_policy_check(d)

    def test_guard_mutation_fails_closed(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d:
            guard = asdict(mod.GUARD)
            guard['cosine_floor'] = 0.60
            (Path(d) / 'comparison.json').write_text(json.dumps({
                'baseline_parity': True, 'base_policy': asdict(mod.POLICY),
                'experiments': [{'guard': guard}]}))
            with self.assertRaisesRegex(ValueError, '0.70'):
                mod.prior_policy_check(d)

    def test_prior_not_parity_rejected(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'comparison.json').write_text(json.dumps({
                'baseline_parity': False, 'base_policy': asdict(mod.POLICY),
                'experiments': [{'guard': asdict(mod.GUARD)}]}))
            with self.assertRaisesRegex(ValueError, 'parity'):
                mod.prior_policy_check(d)

    def test_prior_policy_mutation_fails_closed(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as d:
            policy = asdict(mod.POLICY)
            policy['threshold'] = .60
            (Path(d) / 'comparison.json').write_text(json.dumps({
                'baseline_parity': True, 'base_policy': policy,
                'experiments': [{'guard': asdict(mod.GUARD)}]}))
            with self.assertRaisesRegex(ValueError, 'frozen'):
                mod.prior_policy_check(d)

    def test_score_subset_counts_novel_without_removing_distractors(self):
        # A new person 9 must match in two cameras; person 3 remains in gallery.
        gt = {1400: {'C1': [{'gt_id': 9, 'xyxy': [0,0,20,40]}, {'gt_id': 3, 'xyxy': [50,0,70,40]}],
                     'C2': [{'gt_id': 9, 'xyxy': [0,0,20,40]}], 'C3': [], 'C4': []}}
        labels = {1400: {'C1': [9, 3], 'C2': [9], 'C3': [], 'C4': []}}
        pred = {1400: {'C1': [1, 2], 'C2': [1], 'C3': [], 'C4': []}}
        s = mod.summarize_score(pred, labels, gt, {9})
        self.assertEqual(s['tp'], 1)
        self.assertEqual(s['fp'], 0)
        self.assertEqual(s['true_pairs'], 1)
        # Wrong match to a distractor is still a false positive for the novel ID.
        pred[1400]['C2'] = [2]
        s = mod.summarize_score(pred, labels, gt, {9})
        self.assertEqual(s['fp'], 1)

    def test_empty_novel_set_never_implies_precision_100(self):
        gt = {1400: {cam: [] for cam in mod.CAMERAS}}
        pred = {1400: {cam: [] for cam in mod.CAMERAS}}
        s = mod.summarize_score(pred, {1400: {cam: [] for cam in mod.CAMERAS}}, gt, set())
        self.assertIsNone(s['precision'])
        self.assertEqual(s['true_pairs'], 0)

    def test_write_csv_preserves_zero_and_null(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'r.csv'
            mod.write_csv(dest, [{'a': 0, 'b': None}], ['a', 'b'])
            with dest.open(newline='') as f:
                self.assertEqual(next(csv.DictReader(f)), {'a': '0', 'b': ''})

    def test_prepare_manifest_declares_new_people(self):
        old_gt = {fid: {cam: [] for cam in mod.CAMERAS} for fid in range(0,500,5)}
        old_gt[0]['C1'] = [{'gt_id': 115}]
        late_gt = {fid: {cam: [] for cam in mod.CAMERAS} for fid in mod.fixed_frame_ids(True)}
        late_gt[1400]['C2'] = [{'gt_id': 115}, {'gt_id': 999}]
        with tempfile.TemporaryDirectory() as d:
            m = {'frame_ids': list(range(0,500,5)), 'gt_sha256': source_hash(old_gt)}
            old_oracle = Path(d) / 'old'
            old_oracle.mkdir()
            (old_oracle / 'oracle_manifest.json').write_text(json.dumps(m))
            with patch.object(mod, 'ground_truth', side_effect=[old_gt, late_gt]), \
                 patch.object(mod, 'validate_assets'):
                result = mod.prepare(d, old_oracle, Path(d) / 'out', pilot=True)
            self.assertEqual(result['novel_gt_ids'], [999])
            self.assertEqual(result['shared_gt_id_count'], 1)
            saved = json.loads((Path(d) / 'out/oracle/oracle_manifest.json').read_text())
            self.assertEqual(saved['gt_sha256'], source_hash(late_gt))

    def test_check_manifest_fails_if_old_annotations_change(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'validation_manifest.json').write_text(json.dumps({
                'previous_frame_ids_sha256': 'old', 'frame_ids': mod.fixed_frame_ids(),
                'mode': 'full'}))
            with patch.object(mod, 'previous_gt', return_value=(list(range(0,500,5)), {}, 'new')):
                with self.assertRaisesRegex(ValueError, 'Prior annotation SHA'):
                    mod.check_manifest(d, d, d)

    def test_report_handles_precision_undefined(self):
        s = {'tp':0, 'fp':0, 'recall':None, 'precision':None,
             'unique_wrongly_merged_identity_pairs':0}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'comparison.json').write_text(json.dumps({
                'mode':'pilot', 'evaluation_source':{'frame_count':15,'novel_gt_id_count':0},
                'results': {'p': {'scores': {'novel_identities':s}}}}))
            self.assertIsNotNone(mod.report(d))

    def test_production_threshold_is_not_tuned(self):
        self.assertEqual(mod.MATCH_THRESHOLD, .85)
        self.assertEqual(mod.MATCH_MARGIN, .05)
        self.assertEqual(mod.POLICY.threshold, .8)
        self.assertEqual(mod.GUARD.cosine_floor, .7)

    def test_run_mocked_pipeline_emits_all_three_locked_policies(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / 'old_features').mkdir()
            (d / 'old_features/embeddings_manifest.json').write_text(json.dumps({'checkpoint_sha256': 'abc'}))
            weight = d / 'model.pt'
            weight.write_bytes(b'weights')
            fids = [1400, 1410, 1420]
            one = {'C1': [(99, (0., 0., 20., 40.), [1., 0.])],
                   'C2': [(99, (0., 0., 20., 40.), [1., 0.])], 'C3': [], 'C4': []}
            frame = {fid: one for fid in fids}
            gt = {fid: {'C1': [{'gt_id': 99, 'xyxy': [0, 0, 20, 40]}],
                        'C2': [{'gt_id': 99, 'xyxy': [0, 0, 20, 40]}], 'C3': [], 'C4': []}
                  for fid in fids}
            labels = {fid: {'C1': [99], 'C2': [99], 'C3': [], 'C4': []} for fid in fids}
            p = {fid: {'C1': [1], 'C2': [1], 'C3': [], 'C4': []} for fid in fids}
            old_gt = {0: {'C1': [{'gt_id': 2}], 'C2': [], 'C3': [], 'C4': []}}
            matcher = Mock()
            matcher.attempts = []
            matcher.decisions = {}
            manifest = {'frame_ids': fids, 'ground_truth_sha256': 'gt',
                        'features_sha256': 'embedding', 'checkpoint_sha256': 'abc'}
            prepared = {'mode': 'pilot', 'frame_ids': fids, 'gt_sha256': 'gt',
                        'limitations': [], 'novel_gt_ids': [99]}
            with patch.object(mod, 'check_manifest', return_value=prepared), \
                 patch.object(mod, 'prior_policy_check', return_value={'source':{'checkpoint_sha256':'abc'}}), \
                 patch.object(mod, 'read_features', return_value=(manifest, gt, frame)), \
                 patch.object(mod, 'previous_gt', return_value=([0], old_gt, 'old')), \
                 patch.object(mod, 'MODEL_WEIGHTS', weight), \
                 patch.object(mod, 'sha256', return_value='abc'), \
                 patch.object(mod, 'baseline_replay', return_value=(p, labels)), \
                 patch.object(mod, 'replay', return_value=(p, labels)), \
                 patch.object(mod, 'replay_source', return_value=(matcher, p, labels)) as call:
                report = mod.run(output_dir=d, old_features_dir=d / 'old_features')
            self.assertEqual(call.call_count, 2)
            self.assertEqual(set(report['results']), set(mod.POLICIES))
            self.assertEqual(report['evaluation_source']['novel_gt_id_count'], 1)
            self.assertEqual(report['results'][mod.POLICIES[0]]['scores']['novel_identities']['tp'], 3)
            self.assertTrue((d / 'comparison.csv').is_file())
            self.assertTrue((d / 'camera_pair_metrics.csv').is_file())

    def test_run_aborts_on_modified_model_hash(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            w = d / 'model.pt'
            w.write_bytes(b'model')
            (d / 'old').mkdir()
            (d / 'old/embeddings_manifest.json').write_text(json.dumps({'checkpoint_sha256': 'old'}))
            manifest = {'ground_truth_sha256':'gt', 'frame_ids':[1400],
                        'checkpoint_sha256':'changed', 'features_sha256':'f'}
            prepared = {'gt_sha256':'gt','frame_ids':[1400]}
            with patch.object(mod, 'check_manifest', return_value=prepared), \
                 patch.object(mod, 'prior_policy_check', return_value={'source':{'checkpoint_sha256':'old'}}), \
                 patch.object(mod, 'read_features', return_value=(manifest, {}, {})), \
                 patch.object(mod, 'MODEL_WEIGHTS', w), \
                 patch.object(mod, 'sha256', return_value='old'):
                with self.assertRaisesRegex(ValueError, 'Checkpoint differs'):
                    mod.run(output_dir=d, old_features_dir=d / 'old')


if __name__ == '__main__':
    unittest.main()
