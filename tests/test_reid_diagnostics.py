"""Step 14.20 pure-CPU tests. Real OSNet/WILDTRACK test runs on Windows."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from unittest.mock import MagicMock

from src import reid_diagnostics as d


def feat(a, b):
    return d.normalize([a, b, 0.0])


def sample(fid, cam, person, feature, x=0):
    return (int(person), (float(x), 0., float(x + 40), 70.), feature)


def fixture(ids=(0, 5, 10, 15)):
    gt, emb = {}, {}
    for fid in ids:
        gt[fid] = {c: [] for c in d.CAMERAS}
        emb[fid] = {c: [] for c in d.CAMERAS}
        for cam, entries in {
            "C1": [sample(fid, "C1", 7, feat(1, 0))],
            "C2": [sample(fid, "C2", 7, feat(1, 0))],
        }.items():
            for pid, bb, v in entries:
                emb[fid][cam].append((pid, bb, v))
                gt[fid][cam].append({"gt_id": pid, "xyxy": list(bb)})
    return gt, emb


class RawSimilarityTests(unittest.TestCase):
    def test_normalize_unit(self):
        v = d.normalize([3, 4])
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1)
        self.assertAlmostEqual(float(v[0]), 0.6)

    def test_normalize_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            d.normalize([0, 0])

    def test_normalize_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            d.normalize([float('nan'), 0])

    def test_auc_perfect_separation(self):
        self.assertEqual(d.auc_score([.8, .9], [-.2, .1]), 1.0)

    def test_auc_reversed(self):
        self.assertEqual(d.auc_score([.0], [1.]), 0.0)

    def test_auc_tie(self):
        self.assertEqual(d.auc_score([1., 1.], [1., 1.]), 0.5)

    def test_auc_missing_class(self):
        self.assertIsNone(d.auc_score([], [0.4]))

    def test_quantiles_empty(self):
        self.assertEqual(d.quantiles([])["n"], 0)
        self.assertIsNone(d.quantiles([])["p50"])

    def test_pair_metrics_for_two_persons(self):
        frame = {c: [] for c in d.CAMERAS}
        frame['C1'] = [sample(0, 'C1', 1, feat(1, 0)), sample(0, 'C1', 2, feat(0, 1), 60)]
        frame['C2'] = [sample(0, 'C2', 1, feat(1, 0)), sample(0, 'C2', 2, feat(0, 1), 60)]
        report, hard, low = d.score_pairs({0: frame})
        pair = report['camera_pairs']['C1-C2']
        self.assertEqual(pair['genuine']['n'], 2)
        self.assertEqual(pair['impostor']['n'], 2)
        self.assertEqual(pair['roc_auc'], 1.0)
        self.assertEqual(pair['retrieval']['eligible_queries'], 4)
        self.assertEqual(pair['retrieval']['top1'], 1.0)
        self.assertEqual(len(hard), 2)
        self.assertEqual(len(low), 2)
        self.assertEqual(report['threshold_sensitivity'][0]['genuine_above_threshold'], 1.0)

    def test_no_matching_gallery_is_ineligible(self):
        frame = {c: [] for c in d.CAMERAS}
        frame['C1'] = [sample(0, 'C1', 1, feat(1, 0))]
        frame['C2'] = [sample(0, 'C2', 2, feat(0, 1))]
        report, _, _ = d.score_pairs({0: frame})
        self.assertEqual(report['camera_pairs']['C1-C2']['retrieval']['eligible_queries'], 0)
        self.assertIsNone(report['camera_pairs']['C1-C2']['retrieval']['top1'])
        self.assertIsNone(report['roc_auc'])

    def test_negative_harder_than_positive(self):
        frame = {c: [] for c in d.CAMERAS}
        frame['C1'] = [sample(0, 'C1', 1, feat(1, 0)), sample(0, 'C1', 2, feat(1, 0), 60)]
        frame['C2'] = [sample(0, 'C2', 1, feat(1, 0)), sample(0, 'C2', 2, feat(1, 0), 60)]
        report, _, _ = d.score_pairs({0: frame})
        self.assertEqual(report['roc_auc'], .5)

    def test_frozen_encoder_does_not_pass_labels(self):
        _, emb = fixture()
        encoder = d.FrozenEncoder(emb)
        x = encoder.encode_batch([('C1', 0, None, [{'xyxy': [0, 0, 40, 70], 'confidence': 1.0}])])
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0].shape, (1, 3))
        with self.assertRaises(KeyError):
            encoder.encode_batch([('C1', 0, None, [{'xyxy': [5, 0, 40, 70]}])])


class AssociationAuditTests(unittest.TestCase):
    def test_oracle_replay_consistent_match(self):
        gt, emb = fixture()
        report, decisions = d.shadow_audit(emb, gt)
        self.assertEqual(report['reference_matcher_parity'], 'verified for every global ID and association decision')
        self.assertTrue(report['gate_reasons']['accepted'] >= 1)
        self.assertEqual(report['pairwise_diagnostics']['C1-C2']['correct_same_identity_pairs'], 2)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[1]['reason'], 'accepted')
        self.assertTrue(decisions[1]['best_candidate_correct'])

    def test_below_threshold(self):
        gt, emb = fixture()
        for fid in emb:
            emb[fid]['C2'][0] = sample(fid, 'C2', 7, feat(0, 1))
        result, decisions = d.shadow_audit(emb, gt)
        self.assertEqual(result['gate_reasons']['below_threshold'], 1)
        self.assertTrue(any(x['reason'] == 'below_threshold' for x in decisions))

    def test_below_margin(self):
        ids = [0, 5, 10]
        gt = {fid: {c: [] for c in d.CAMERAS} for fid in ids}
        emb = {fid: {c: [] for c in d.CAMERAS} for fid in ids}
        for fid in ids:
            entries = {
                'C1': [sample(fid, 'C1', 1, feat(1, 0)),
                       sample(fid, 'C1', 2, feat(.98, .2), 60)],
                'C2': [sample(fid, 'C2', 3, feat(.99, .1))],
            }
            for cam, rows in entries.items():
                emb[fid][cam] = rows
                gt[fid][cam] = [{'gt_id': pid, 'xyxy': list(bb)} for pid, bb, _ in rows]
        report, decisions = d.shadow_audit(emb, gt)
        self.assertEqual(report['gate_reasons']['below_margin'], 1)
        self.assertEqual(sum(x['reason'] == 'below_margin' for x in decisions), 1)

    def test_no_candidate(self):
        gt, emb = fixture()
        report, _ = d.shadow_audit(emb, gt)
        self.assertEqual(report['gate_reasons']['no_candidate'], 1)

    def test_wrong_input_order_rejected(self):
        gt, emb = fixture()
        gt[0]['C1'][0]['xyxy'] = [3, 0, 43, 70]
        with self.assertRaisesRegex(ValueError, 'order mismatch'):
            d.shadow_audit(emb, gt)

    def test_decisions_written_with_labels_only_in_audit(self):
        gt, emb = fixture()
        _, decisions = d.shadow_audit(emb, gt)
        self.assertIn('gt_id_for_scoring_only', decisions[0])
        self.assertNotIn('gt_id', d.FrozenEncoder(emb).lookup)


class PersistedDataTests(unittest.TestCase):
    def test_read_features_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt, emb = fixture([0])
            rows = []
            feats = []
            for cam in d.CAMERAS:
                for pid, box, v in emb[0][cam]:
                    rows.append([0, d.CAM_TO_INDEX[cam], pid, *box])
                    feats.append(v)
            path = out / 'embeddings.npz'
            with path.open('wb') as f:
                np.savez_compressed(f, features=np.stack(feats), metadata=np.array(rows))
            manifest = {'frame_ids': [0], 'ground_truth_sha256': d.source_hash(gt),
                        'feature_file': path.name, 'features_sha256': d.sha256(path), 'crop_count': 2, 'feature_dimension': 3}
            d.dump_json(out / 'embeddings_manifest.json', manifest)
            with patch.object(d, 'get_gt', return_value=([0], gt, {})):
                _, parsed_gt, parsed_emb = d.read_features(out, out, out)
                self.assertEqual(parsed_gt, gt)
                self.assertEqual(parsed_emb[0]['C1'][0][0], 7)
                # Changed bytes fail validation, not merely a warning.
                with path.open('ab') as f:
                    f.write(b'altered')
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    d.read_features(out, out, out)

    def test_reader_rejects_wrong_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt, emb = fixture([0])
            path = out / 'embeddings.npz'
            with path.open('wb') as f:
                np.savez_compressed(f, features=np.stack([feat(1, 0)] * 2),
                                    metadata=np.asarray([[0, 0, 7, 1, 0, 41, 70],
                                                         [0, 1, 7, 0, 0, 40, 70]], dtype=float))
            d.dump_json(out / 'embeddings_manifest.json', {
                'frame_ids': [0], 'ground_truth_sha256': d.source_hash(gt),
                'feature_file': path.name, 'features_sha256': d.sha256(path),
                'crop_count': 2, 'feature_dimension': 3,
            })
            with patch.object(d, 'get_gt', return_value=([0], gt, {})):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    d.read_features(out, out, out)

    def test_analyze_creates_report(self):
        gt, emb = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(d, 'read_features', return_value=({'frame_ids': list(gt), 'crop_count': 8,
                                                                 'ground_truth_sha256': 'g',
                                                                 'checkpoint_sha256': 'm',
                                                                 'feature_dimension': 3}, gt, emb)):
                report = d.analyze(out, out, out)
            self.assertEqual(report['embedding_quality']['roc_auc'], None)  # no impostors in fixture
            self.assertTrue((out / 'diagnostics.json').is_file())
            self.assertTrue((out / 'gate_decisions.csv').is_file())
            self.assertTrue((out / 'camera_pair_retrieval.csv').is_file())


class ExtractionIntegrationTests(unittest.TestCase):
    def test_frozen_extraction_and_analysis(self):
        import cv2
        from src import reid, identity_evaluation
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            weights = root / 'osnet.pt'
            weights.write_bytes(b'synthetic-mock-weights')
            gt, _ = fixture([0, 5, 10])
            out = root / 'out'
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            encoder = MagicMock()
            encoder.encode_batch.side_effect = lambda requests: [
                np.stack([feat(1, 0) for _ in people])
                for _, _, _, people in requests
            ]
            with patch.object(d, 'get_gt', return_value=([0, 5, 10], gt, {'gt_sha256': d.source_hash(gt)})), \
                 patch.object(identity_evaluation, '_configure_execution', return_value=([0, 2, 4, 6], 4)), \
                 patch.object(reid, 'AppearanceEncoder', return_value=encoder), \
                 patch.object(reid, 'MODEL_WEIGHTS', weights), \
                 patch.object(cv2, 'imread', return_value=image):
                manifest = d.extract(root, root, out, 4, '0,2,4,6')
            self.assertEqual(manifest['crop_count'], 6)
            self.assertEqual(encoder.encode_batch.call_count, 3)
            self.assertTrue((out / 'embeddings.npz').is_file())
            with patch.object(d, 'get_gt', return_value=([0, 5, 10], gt, {})):
                report = d.analyze(root, root, out)
            self.assertTrue((out / 'diagnostics.json').is_file())
            self.assertEqual(report['association_gates']['gate_reasons']['accepted'], 1)
            self.assertEqual(report['association_gates']['pairwise_diagnostics']['C1-C2']['correct_same_identity_pairs'], 1)

    def test_reference_parity_catches_changed_matcher(self):
        from src import reid
        gt, emb = fixture()
        original = reid.CrossCameraMatcher.update
        def corrupt(self, *args, **kwargs):
            tracks, decisions = original(self, *args, **kwargs)
            if decisions:
                decisions[0]['decision'] = 'corrupted'
            return tracks, decisions
        with patch.object(reid.CrossCameraMatcher, 'update', corrupt):
            with self.assertRaisesRegex(RuntimeError, 'decision parity'):
                d.shadow_audit(emb, gt)


if __name__ == '__main__':
    unittest.main()
