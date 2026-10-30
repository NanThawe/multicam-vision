"""Dependency-light tests; do not require torch, BoxMOT, videos or WILDTRACK."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import identity_evaluation as m


def box(pid, x1=0, y1=0, x2=50, y2=100):
    return {"gt_id": pid, "xyxy": [x1, y1, x2, y2]}


def pred(tid, gid, x1=0, y1=0, x2=50, y2=100):
    return {"track_id": tid, "global_id": gid, "xyxy": [x1, y1, x2, y2]}


def blank():
    return {cam: [] for cam in m.CAMERAS}


class IdentityEvaluationTests(unittest.TestCase):
    def test_eligible_threshold_exact(self):
        self.assertTrue(m.eligible([0, 0, 16, 32]))
        self.assertFalse(m.eligible([0, 0, 15, 32]))
        self.assertFalse(m.eligible([0, 0, 16, 31]))

    def test_tracks_opaque_and_camera_scoped(self):
        gt = blank()
        gt["C1"] = [box(50), box(200, 10, 10, 12, 30)]
        gt["C2"] = [box(50)]
        ids = {cam: {} for cam in m.CAMERAS}
        tracks = m.oracle_tracks(gt, ids)
        self.assertEqual(tracks["C1"][0]["track_id"], 1)
        self.assertEqual(tracks["C2"][0]["track_id"], 1)
        self.assertNotIn("gt_id", tracks["C1"][0])
        self.assertEqual(len(tracks["C1"]), 1)
        gt2 = blank()
        gt2["C1"] = [box(50), box(300)]
        next_tracks = m.oracle_tracks(gt2, ids)
        self.assertEqual([x["track_id"] for x in next_tracks["C1"]], [1, 2])

    def test_tracks_ids_are_not_derived_from_ground_truth_person_id(self):
        gt = blank()
        gt["C1"] = [box(999), box(2)]
        gt["C2"] = [box(2), box(999)]
        ids = {cam: {} for cam in m.CAMERAS}
        tracks = m.oracle_tracks(gt, ids)
        self.assertEqual([p["track_id"] for p in tracks["C1"]], [1, 2])
        self.assertEqual([p["track_id"] for p in tracks["C2"]], [1, 2])
        self.assertEqual(ids["C1"][999] + 1, ids["C2"][999])

    def test_source_hash_order_independent_dicts(self):
        a = {0: {"C1": [box(1)], "C2": []}}
        b = {0: {"C2": [], "C1": [box(1)]}}
        self.assertEqual(m.source_hash(a), m.source_hash(b))
        b[0]["C1"][0]["gt_id"] = 2
        self.assertNotEqual(m.source_hash(a), m.source_hash(b))

    def test_coverage_cold_start(self):
        gt = {}
        for fid in (0, 5, 10, 15):
            f = blank()
            f["C1"] = [box(7)]
            f["C2"] = [box(7)]
            gt[fid] = f
        c = m.pair_coverage(gt)
        self.assertEqual(c["true_gt_cross_camera_pairs"], 4)
        self.assertEqual(c["mature_eligible_gt_pairs"], 2)
        self.assertEqual(c["crop_eligible_gt_pairs"], 4)
        self.assertAlmostEqual(c["mature_eligible_fraction"], 0.5)

    def test_coverage_ineligible_does_not_warmup(self):
        gt = {}
        for fid in (0, 5, 10, 15):
            f = blank()
            f["C1"] = [box(7, 0, 0, 15, 100) if fid == 5 else box(7)]
            f["C2"] = [box(7)]
            gt[fid] = f
        c = m.pair_coverage(gt)
        self.assertEqual(c["true_gt_cross_camera_pairs"], 4)
        self.assertEqual(c["crop_eligible_gt_pairs"], 3)
        self.assertEqual(c["mature_eligible_gt_pairs"], 1)

    def test_pair_diagnostics_correct_and_false(self):
        gt = blank()
        gt["C1"] = [box(7), box(10, 100, 0, 150, 100)]
        gt["C2"] = [box(7), box(11, 100, 0, 150, 100)]
        pr = blank()
        pr["C1"] = [pred(1, 20), pred(2, 30, 100, 0, 150, 100)]
        pr["C2"] = [pred(1, 20), pred(3, 30, 100, 0, 150, 100)]
        result = m.pair_diagnostics({0: gt}, {0: pr})["C1-C2"]
        self.assertEqual(result["true_gt_pairs"], 1)
        self.assertEqual(result["predicted_same_identity_pairs"], 2)
        self.assertEqual(result["correct_same_identity_pairs"], 1)
        self.assertEqual(result["false_same_identity_pairs"], 1)
        self.assertAlmostEqual(result["pair_precision"], 0.5)
        self.assertAlmostEqual(result["pair_recall"], 1.0)

    def test_pair_diagnostics_none_precision_no_prediction(self):
        g, p = blank(), blank()
        g["C1"], g["C2"] = [box(7)], [box(7)]
        p["C1"], p["C2"] = [pred(1, None)], [pred(2, None)]
        r = m.pair_diagnostics({0: g}, {0: p})["C1-C2"]
        self.assertIsNone(r["pair_precision"])
        self.assertEqual(r["pair_recall"], 0)

    def test_predictions_strict_alignment_and_extras(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pred.jsonl"
            with path.open("w") as f:
                for fid in (0, 5):
                    for cam in m.CAMERAS:
                        f.write(json.dumps({"frame_id": fid, "camera": cam,
                                            "tracks": [pred(1, 9)]}) + "\n")
            data, extras = m.load_predictions(path, [0])
            self.assertEqual(extras, 4)
            self.assertEqual(list(data.keys()), [0])
            self.assertEqual(set(data[0]), set(m.CAMERAS))

    def test_predictions_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pred.jsonl"
            row = json.dumps({"frame_id": 0, "camera": "C1", "tracks": []})
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                m.load_predictions(path, [0])

    def test_predictions_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pred.jsonl"
            path.write_text(json.dumps({"frame_id": 0, "camera": "C1", "tracks": []}))
            with self.assertRaisesRegex(ValueError, "Missing"):
                m.load_predictions(path, [0])

    def test_predictions_nan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pred.jsonl"
            path.write_text(json.dumps({"frame_id": 0, "camera": "C1", "tracks": [
                {"xyxy": [0, 0, float("nan"), 1]}]}))
            with self.assertRaisesRegex(ValueError, "invalid track coordinates"):
                m.load_predictions(path, [0])

    def test_measure_matches_earlier_scorer(self):
        g, p = blank(), blank()
        g["C1"] = [box(7)]
        g["C2"] = [box(7)]
        p["C1"] = [pred(1, 10)]
        p["C2"] = [pred(2, 10)]
        actual = m.measure({0: g}, {0: p}, threshold=.5)
        self.assertEqual(actual["metrics"]["cross_camera"]["correct_same_identity_pairs"], 1)
        self.assertEqual(actual["camera_pair_breakdown"]["C1-C2"]["correct_same_identity_pairs"], 1)

    def test_compare_same_subset_and_extra_original_rows(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            oracle_dir = d / "oracle"
            oracle_dir.mkdir()
            g = blank()
            g["C1"] = [box(7)]
            g["C2"] = [box(7)]
            truth = {0: g}
            m.dump_json(oracle_dir / "oracle_manifest.json",
                        {"frame_ids": [0], "gt_sha256": m.source_hash(truth)})
            for filename in (oracle_dir / "oracle_predictions.jsonl", d / "pipeline.jsonl"):
                with filename.open("w") as f:
                    for cam in m.CAMERAS:
                        f.write(json.dumps({"frame_id": 0, "camera": cam,
                                            "tracks": [pred(1, 1)] if cam in ("C1", "C2") else []}) + "\n")
                    if filename.name == "pipeline.jsonl":
                        f.write(json.dumps({"frame_id": 5, "camera": "C1", "tracks": []}) + "\n")
            with patch.object(m, "available_frames", return_value=[0, 5]), \
                    patch.object(m, "ground_truth", return_value=truth):
                result = m.compare(d, oracle_dir, d / "pipeline.jsonl", d / "out")
            self.assertEqual(result["headline"]["oracle_correct_pairs"], 1)
            self.assertEqual(result["headline"]["pipeline_correct_pairs"], 1)
            self.assertEqual(result["pipeline_extraneous_file_rows_ignored"], 1)
            self.assertTrue((d / "out" / "camera_pair_comparison.csv").is_file())

    def test_compare_rejects_modified_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            od = d / "oracle"
            od.mkdir()
            m.dump_json(od / "oracle_manifest.json", {"frame_ids": [0], "gt_sha256": "incorrect"})
            with patch.object(m, "available_frames", return_value=[0]), \
                    patch.object(m, "ground_truth", return_value={0: blank()}):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    m.compare(d, od, d / "fake.jsonl", d / "out")

    def test_configure_threads_bad(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            m._configure_execution(0, "all")


if __name__ == "__main__":
    unittest.main()
