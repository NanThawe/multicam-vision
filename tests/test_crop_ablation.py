"""Step 14.23 synthetic tests: not a real OSNet/WILDTRACK benchmark."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import crop_ablation as c


def sample_data():
    u = np.array([1., 0., 0.], np.float32)
    v = np.array([0., 1., 0.], np.float32)
    bb1 = (25., 10., 75., 90.)
    bb2 = (120., 10., 170., 90.)
    frames = {
        405: {"C1": [(116, bb1, u), (115, bb2, v)],
              "C2": [(115, bb1, v), (116, bb2, u)], "C3": [], "C4": []},
        410: {"C1": [(116, bb1, u), (115, bb2, v)],
              "C2": [(115, bb1, v), (116, bb2, u)], "C3": [], "C4": []},
    }
    return frames


class BboxTests(unittest.TestCase):
    def test_no_padding_identity(self):
        box = (20., 30., 60., 90.)
        self.assertEqual(c.box_padding(box, 0.), box)

    def test_padding_expand_and_clip(self):
        self.assertEqual(c.box_padding((10, 5, 30, 45), .5, 200, 100),
                         (0., 0., 40., 65.))

    def test_borders_not_wrapped(self):
        self.assertEqual(c.box_padding((190, 90, 200, 100), .2, 200, 100),
                         (188., 88., 200., 100.))

    def test_bad_padding(self):
        for p in (-.1, .51, float("nan"), float("inf")):
            with self.assertRaises(ValueError): c.box_padding((1, 1, 40, 50), p)

    def test_invalid_bounds(self):
        for box in ((-1, 1, 10, 20), (1, 1, 1, 10), (1, 1, 999999, 20)):
            with self.assertRaises(ValueError): c.box_padding(box, .1)

    def test_overlaps(self):
        self.assertAlmostEqual(c.overlap_fraction((0, 0, 10, 10), (5, 0, 20, 10)), .5)
        self.assertEqual(c.overlap_fraction((0, 0, 10, 10), (10, 0, 20, 10)), 0)

    def test_geometry_stats_detect_context_contamination(self):
        gt = {1: {"C1": [{"xyxy": (10, 10, 30, 50)},
                           {"xyxy": (33, 10, 53, 50)}]}}
        report = c.summarize_boxes(gt)
        self.assertEqual(report["pad_0"]["overlap_other_gt"], 0)
        self.assertEqual(report["pad_20"]["overlap_other_gt"], 2)
        self.assertEqual(report["pad_20"]["crops"], 2)


class FeatureCacheTests(unittest.TestCase):
    def test_feature_rows_deterministic(self):
        frames = sample_data()
        r = c._expected_rows(frames)
        self.assertEqual(r.shape, (8, 7))
        self.assertEqual(tuple(r[0, :3]), (405., 0., 116.))
        self.assertEqual(tuple(r[3, :3]), (405., 1., 116.))
        self.assertEqual(tuple(r[-1, :3]), (410., 1., 116.))

    def test_rebuild_features_exact_order(self):
        frames = sample_data()
        rows = c._expected_rows(frames)
        vectors = np.zeros((len(rows), 3), dtype=np.float32)
        vectors[:, 2] = 1
        rebuilt = c.rebuild_by_frame(frames, vectors)
        self.assertEqual(rebuilt[405]["C1"][0][0], 116)
        self.assertEqual(rebuilt[405]["C2"][1][0], 116)
        self.assertTrue(np.allclose(rebuilt[410]["C2"][1][2], [0, 0, 1]))
        with self.assertRaises(ValueError): c.rebuild_by_frame(frames, vectors[:-1])

    def test_invalid_meta_rejected(self):
        expected = c._expected_rows(sample_data())
        feats = np.tile(np.array([1., 0.], np.float32), (len(expected), 1))
        c.validate_meta(feats, expected, expected)
        altered = expected.copy(); altered[0, 2] = 999
        with self.assertRaisesRegex(ValueError, "metadata"): c.validate_meta(feats, altered, expected)
        with self.assertRaisesRegex(ValueError, "normalized"): c.validate_meta(feats*2, expected, expected)

    def test_save_then_load_and_tamper(self):
        base = sample_data(); exp = c._expected_rows(base)
        feats = np.tile(np.array([1., 0., 0.], np.float32), (len(exp), 1))
        man = {"features_sha256": "base123", "ground_truth_sha256": "gt123", "checkpoint_sha256": "model123"}
        with tempfile.TemporaryDirectory() as td:
            c.save_variant(td, .1, feats, exp, man, exp)
            recovered = c.load_variant(td, .1, man, exp)
            self.assertTrue(np.array_equal(feats, recovered))
            mf = Path(td) / 'manifest_pad10.json'
            x = json.loads(mf.read_text()); x['checkpoint_sha256'] = 'wrong'; mf.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError, "mismatch"): c.load_variant(td, .1, man, exp)

    def test_corrupted_feature_hash_rejected(self):
        exp = c._expected_rows(sample_data()); feats = np.tile(np.array([1., 0.], np.float32), (len(exp), 1))
        m = {"features_sha256": "abc", "ground_truth_sha256": "gt", "checkpoint_sha256": "model"}
        with tempfile.TemporaryDirectory() as td:
            c.save_variant(td, .2, feats, exp, m, exp)
            with (Path(td) / "features_pad20.npz").open("ab") as f: f.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash"): c.load_variant(td, .2, m, exp)

    def test_frac_parse(self):
        self.assertEqual(c.parse_fractions("0.10,0.20"), [.1, .2])
        for v in ("0", ".1,.1", "0.101", "0.6", "nan", "", "0.1,wat"):
            with self.assertRaises(ValueError): c.parse_fractions(v)

    def test_case_pair_cosines(self):
        data = sample_data()
        results = c.pair_cosines(data, frames=[405, 410, 420])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["gt_id_a"], 116)
        self.assertAlmostEqual(results[0]["cosine"], 0.0)


class MockEncoder:
    def encode_batch(self, requests):
        # Deterministic toy function of input box coordinates, not GT labels.
        batches = []
        for _, _, _, people in requests:
            out = []
            for p in people:
                b = p['xyxy']
                vec = np.array([b[0]+1, b[1]+1, 1.], dtype=np.float32)
                out.append(c.normalize(vec))
            batches.append(np.stack(out))
        return batches


class MockCV2:
    def imread(self, path):
        return np.zeros((c.IMAGE_HEIGHT, c.IMAGE_WIDTH, 3), np.uint8)


class IntegrationTests(unittest.TestCase):
    def test_mock_inference_only_changes_boxes(self):
        frames = sample_data(); expected = c._expected_rows(frames)
        features0, meta0 = c.encode_padding(frames, "irrelevant", MockEncoder(), 0., MockCV2())
        features1, meta1 = c.encode_padding(frames, "irrelevant", MockEncoder(), .2, MockCV2())
        self.assertTrue(np.array_equal(meta0, expected))
        self.assertTrue(np.array_equal(meta1, expected))
        self.assertTrue(np.all(np.isfinite(features1)))
        self.assertFalse(np.array_equal(features0, features1))

    def test_mock_inference_invalid_image_fails(self):
        class Missing:
            def imread(self, path): return None
        with self.assertRaises(FileNotFoundError):
            c.encode_padding(sample_data(), "missing", MockEncoder(), .1, Missing())

    def test_encoder_batch_count_checked(self):
        class Empty:
            def encode_batch(self, requests): return []
        with self.assertRaisesRegex(ValueError, "batch count"):
            c.encode_padding(sample_data(), "irrelevant", Empty(), .1, MockCV2())

    def test_retrieval_score_baseline(self):
        base = sample_data()
        s, _, _ = c.score_pairs(base)
        self.assertEqual(s['camera_pairs']['C1-C2']['retrieval']['top1'], 1.0)

    def test_report_compares_same_frozen_crops(self):
        base = sample_data()
        expected = c._expected_rows(base)
        baseline_manifest = {"features_sha256": "ref", "ground_truth_sha256": "gt", "checkpoint_sha256": "model", "frame_ids": list(range(100))}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "out"; out.mkdir()
            (td / "features").mkdir()
            same = np.stack([entry[2] for fid in sorted(base) for cam in c.CAMERAS
                             for entry in base[fid].get(cam, [])])
            c.save_variant(out, .1, same, expected, baseline_manifest, expected)
            c.save_variant(out, .2, same, expected, baseline_manifest, expected)
            with patch.object(c, "read_features", return_value=(baseline_manifest, {}, base)):
                report = c.compare(td, td, td / "features", out, [.1, .2])
            self.assertTrue(report["no_selected_winner"])
            self.assertEqual(set(report["variants"]), {"pad00", "pad10", "pad20"})
            self.assertEqual(report["variants"]["pad00"]["roc_auc"], report["variants"]["pad10"]["roc_auc"])
            self.assertTrue((out / "camera_pair_comparison.csv").is_file())
            self.assertEqual(len((out / "camera_pair_comparison.csv").read_text().splitlines()), 19)

    def test_contact_sheet_renders_observed_and_missing_crops(self):
        import cv2
        class CV2Proxy:
            def __getattr__(self, name):
                return getattr(cv2, name)
            def imread(self, path):
                return np.zeros((c.IMAGE_HEIGHT, c.IMAGE_WIDTH, 3), np.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "case.png"
            count = c.create_contact_sheet(td, sample_data(), path, [405, 410], CV2Proxy())
            self.assertEqual(count, 8)
            self.assertTrue(path.is_file())
            rendered = cv2.imread(str(path))
            self.assertEqual(rendered.shape[:2], (688, 1040))



if __name__ == '__main__':
    unittest.main()
