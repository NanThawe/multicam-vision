"""Step 14.8 synthetic tests: do not need an RTSP server or model weights."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.live_rtsp import save_run_results
from src.rtsp_stability import (compare, inspect_run, mqtt_smoke, run_trials,
                                save_analysis, ten_second_windows)
import src.rtsp_stability as stability


class StabilityTests(unittest.TestCase):
    def test_complete_ten_second_windows_are_not_estimated_from_frame_id(self):
        rows = [{"elapsed_s": "2", "processing_ms": "100"},
                {"elapsed_s": "7", "processing_ms": "200"},
                {"elapsed_s": "10", "processing_ms": "300"},
                {"elapsed_s": "12", "processing_ms": "100"},
                {"elapsed_s": "20.1", "processing_ms": "400"}]
        windows = ten_second_windows(rows, 10)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["aggregate_fps"], 1.2)
        self.assertEqual(windows[1]["aggregate_fps"], .4)
        self.assertEqual(windows[0]["batch_p95_ms"], 290)

    def test_no_full_window_yet(self):
        self.assertEqual(ten_second_windows([{"elapsed_s": "9.8", "processing_ms": "100"}]), [])

    def test_empty_window_means_zero_rate(self):
        rows = [{"elapsed_s": "1", "processing_ms": "150"},
                {"elapsed_s": "30", "processing_ms": "150"}]
        windows = ten_second_windows(rows)
        self.assertEqual(windows[1]["aggregate_fps"], 0.0)

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            ten_second_windows([], 0)

    def test_profiled_result_has_stage_report_and_unmodified_base_fields(self):
        samples = []
        names = ("wait_for_batch", "detector", "tracker", "provisional", "reid",
                 "events", "event_io_mqtt", "preview", "thread_check")
        for frame_id, elapsed in [(5, 0.2), (10, 0.45)]:
            samples.append({"frame_id": frame_id, "elapsed_s": elapsed,
                            "frame_age_ms": {c: 30 for c in ("C1", "C2", "C3", "C4")},
                            "processing_ms": 110, "event_count": 1,
                            "stage_ms": {name: (5 if name == "wait_for_batch" else 10)
                                         for name in names},
                            "counters": {c: (10, 1) for c in ("C1", "C2", "C3", "C4")}})
        counters = {c: (0, 0) for c in ("C1", "C2", "C3", "C4")}
        counters_end = {c: (20, 2) for c in counters}
        with tempfile.TemporaryDirectory() as directory:
            save_run_results(samples, counters, counters_end, 0.5, event_count=2,
                             mqtt_acknowledged=0, warmup_batches=1, threads=4,
                             status="completed", output_dir=directory, profile_stages=True)
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["aggregate_processed_fps"], 16)
            self.assertEqual(report["stage_timings_ms"]["detector"]["p95"], 10)
            with (Path(directory) / "per_batch.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["overwrites_C4"], "1")
            self.assertEqual(rows[0]["elapsed_s"], "0.2")
            self.assertEqual(rows[1]["preview_ms"], "10")

    def test_missing_report_is_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            rec, windows = inspect_run(directory)
            self.assertFalse(rec["complete"])
            self.assertEqual(rec["run_status"], "missing")
            self.assertFalse(rec["meets_run_target"])
            self.assertEqual(windows, [])

    def test_report_parsing_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = {
                "complete": True, "aggregate_processed_fps": 20.68,
                "per_camera": {c: {"decoded_frames_received": 100,
                                   "application_buffer_overwrites": 2}
                               for c in ("C1", "C2", "C3", "C4")},
                "processing_ms_per_four_camera_batch": {"p50": 160, "p95": 210},
            }
            (Path(directory) / "report.json").write_text(json.dumps(source))
            (Path(directory) / "run_status.json").write_text('{"status":"completed","complete":true}')
            record, _ = inspect_run(directory)
            self.assertTrue(record["complete"])
            self.assertTrue(record["meets_run_target"])
            self.assertGreater(record["decoded_received"], 0)

    def test_comparison_excludes_failed_runs(self):
        values = [
            {"mode": "baseline", "complete": True, "aggregate_fps": 20.5,
             "p95_processing_ms": 200, "total_overwrites": 1, "meets_run_target": True,
             "windows_below_target": 0},
            {"mode": "baseline", "complete": False, "aggregate_fps": 100,
             "p95_processing_ms": 100, "total_overwrites": 0, "meets_run_target": False,
             "windows_below_target": 0},
            {"mode": "overlay", "complete": True, "aggregate_fps": 19,
             "p95_processing_ms": 230, "total_overwrites": 12, "meets_run_target": False,
             "windows_below_target": 2},
        ]
        result = compare(values)
        self.assertEqual(result["baseline"]["completed_runs"], 1)
        self.assertEqual(result["baseline"]["median_aggregate_fps"], 20.5)
        self.assertEqual(result["overlay_minus_baseline_median_fps"], -1.5)
        self.assertEqual(result["overlay"]["runs_with_window_below_20fps"], 1)

    def test_analysis_export_includes_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = [{"run": "baseline_01", "mode": "baseline"},
                        {"run": "overlay_01", "mode": "overlay"}]
            result = save_analysis(directory, schedule)
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(result["comparison"]["overlay"]["completed_runs"], 0)
            self.assertTrue((Path(directory) / "per_run.csv").exists())
            self.assertTrue((Path(directory) / "time_windows.csv").exists())

    def test_run_alternates_modes_and_records_execution_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(root=Path(directory), seconds=1,
                                   repeats=3, threads=4, warmup_batches=0,
                                   rtsp_base="rtsp://127.0.0.1:8554", minimum_fps=20,
                                   window_s=10)
            with patch.object(stability.subprocess, "run") as execute:
                execute.return_value = SimpleNamespace(returncode=3)
                run_trials(args)
            data = json.loads((Path(directory) / "manifest.json").read_text())
            order = [r["mode"] for r in data["schedule"]]
            self.assertEqual(order, ["baseline", "overlay", "overlay", "baseline", "baseline", "overlay"])
            self.assertEqual(execute.call_count, 6)
            self.assertTrue(all(r["returncode"] == 3 for r in data["schedule"]))
            with self.assertRaises(FileExistsError):
                run_trials(args)

    def test_cli_report_reads_existing_run_without_models(self):
        with tempfile.TemporaryDirectory() as directory:
            code = stability.main(["report", "--root", directory])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
