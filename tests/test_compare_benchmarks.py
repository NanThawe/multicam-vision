"""Tests for summarising three independent benchmark runs."""

import json
from pathlib import Path
import tempfile
import unittest

from src.compare_benchmarks import summarise


class CompareBenchmarkTests(unittest.TestCase):
    def test_medians_are_computed_from_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode, fps_values in (("per_camera", (5, 6, 7)),
                                      ("batched", (9, 12, 15))):
                for i, fps in enumerate(fps_values, 1):
                    path = root / f"{mode}_{i}" / "report.json"
                    path.parent.mkdir()
                    path.write_text(json.dumps({
                        "configuration": {"reid_batch_across_cameras": mode == "batched"},
                        "aggregate_fps": fps,
                        "latency_ms_per_four_camera_batch": {
                            "total": {"p95": 200}, "reid": {"p50": 120}},
                        "reid_workload": {"encoder_calls": 20, "encoded_crops": 100}
                    }))
            summary = summarise(root, 3)
            self.assertEqual(summary["per_camera"]["aggregate_fps_median"], 6)
            self.assertEqual(summary["batched"]["aggregate_fps_median"], 12)
            self.assertEqual(summary["throughput_ratio_batched_vs_per_camera"], 2.0)


if __name__ == "__main__":
    unittest.main()

class CompareAssociationTests(unittest.TestCase):
    def test_identity_comparison_invariant_to_global_id_names(self):
        from src.compare_benchmarks import compare_identity_partitions

        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "ref.jsonl", Path(temp) / "candidate.jsonl"
            def write(path, ids):
                with path.open("w", encoding="utf-8") as out:
                    for camera, track_id, global_id in ids:
                        out.write(json.dumps({
                            "camera": camera,
                            "tracks": [{"track_id": track_id, "global_id": global_id}]
                        }) + "\n")
            write(left, [("C1", 1, 1), ("C2", 1, 1), ("C3", 2, 2)])
            write(right, [("C1", 1, 42), ("C2", 1, 42), ("C3", 2, 55)])
            result = compare_identity_partitions(left, right)
            self.assertEqual(result["disagreement_pairs"], 0)
            self.assertEqual(result["reference_same_identity_pairs"], 1)
            write(right, [("C1", 1, 42), ("C2", 1, 55), ("C3", 2, 55)])
            result = compare_identity_partitions(left, right)
            self.assertEqual(result["reference_matches_missing_in_candidate"], 1)
            self.assertEqual(result["new_candidate_matches_vs_reference"], 1)
