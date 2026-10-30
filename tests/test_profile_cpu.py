"""CPU profiler tests. No camera recordings or ML runtime required."""
import unittest

from src.profile_cpu import benchmark_command, parse_threads, summarize_runs
from src.benchmark import configure_cpu_threads, prepare_inference, check_intraop_threads


def report(fps, *, crops=120, threads=2):
    return {
        "cameras": ["C1", "C2", "C3", "C4"],
        "measured_batches": 100,
        "sampling": {"first_measured_frame": 60, "last_measured_frame": 1248},
        "reid_workload": {"eligible_crops": crops, "encoded_crops": 50,
                          "encoder_calls": 75},
        "configuration": {"reid_interval": 5, "reid_batch_across_cameras": True,
                          "reid_backend": "pytorch", "device": "cpu",
                          "sampling_stride": 12, "torch_intraop_threads": threads,
                          "torch_intraop_threads_verified": True},
        "event_count": 10,
        "aggregate_fps": fps,
        "latency_ms_per_four_camera_batch": {
            "reid": {"p50": 200.0}, "total": {"p95": 400.0}
        },
        "reid_encoder_profile": {"mean_ms_per_batch": {
            "model_encode": 150.0, "input_prep": 30.0}},
    }


class ProfileCpuTests(unittest.TestCase):
    def test_thread_parse(self):
        self.assertEqual(parse_threads("auto, 2,4,2"), ["auto", "2", "4"])
        for raw in ["", "0", "auto,", "foo", "-1"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_threads(raw)

    def test_build_command(self):
        cmd = benchmark_command("4", "output", batches=20, warmup=3)
        self.assertEqual(cmd[-2:], ["--torch-threads", "4"])
        self.assertIn("--batch-across-cameras", cmd)
        self.assertIn("--reid-interval", cmd)
        self.assertNotIn("--torch-threads", benchmark_command(
            "auto", "output", batches=20, warmup=3))

    def test_valid_comparison(self):
        result = summarize_runs({"auto": [report(7), report(8)],
                                 "4": [report(10, threads=4), report(11, threads=4)]})
        self.assertTrue(result["workloads_comparable"])
        self.assertEqual(result["fastest_setting_if_comparable"], "4")
        self.assertEqual(result["thread_settings"]["4"]["aggregate_fps_median"], 10.5)

    def test_different_crops_blocks_ranking(self):
        result = summarize_runs({"auto": [report(7)],
                                 "4": [report(10, crops=100, threads=4)]})
        self.assertFalse(result["workloads_comparable"])
        self.assertIsNone(result["fastest_setting_if_comparable"])

    def test_thread_mismatch_blocks_ranking(self):
        bad = report(10, threads=8)
        result = summarize_runs({"2": [bad], "4": [report(9, threads=4)]})
        self.assertTrue(result["workloads_comparable"])
        self.assertFalse(result["thread_settings_verified"])
        self.assertIsNone(result["fastest_setting_if_comparable"])

    def test_initialisation_replays_first_batch_and_resets_threads(self):
        class FakeTorch:
            num_threads = 8
            def get_num_threads(self): return self.num_threads
            def set_num_threads(self, count): self.num_threads = count
        class FakeUltralyticsUtils:
            NUM_THREADS = 8
        class FakeDetector:
            def __init__(self, library, utils):
                self.library = library
                self.utils = utils
                self.calls = 0
            def detect(self, frames):
                self.calls += 1
                self.library.set_num_threads(self.utils.NUM_THREADS)
        torch = FakeTorch()
        utils = FakeUltralyticsUtils()
        configure_cpu_threads(torch, utils, requested_threads=2)
        self.assertEqual(utils.NUM_THREADS, 2)
        detector = FakeDetector(torch, utils)
        source = iter([(0, {"C1": "frame0"}), (12, {"C1": "frame12"})])
        frames, active = prepare_inference(source, detector, torch, requested_threads=2)
        self.assertEqual(active, 2)
        self.assertEqual(torch.get_num_threads(), 2)
        self.assertEqual([fid for fid, _ in frames], [0, 12])
        self.assertEqual(detector.calls, 1)
        check_intraop_threads(torch, 2)
        torch.set_num_threads(8)
        with self.assertRaisesRegex(RuntimeError, "thread count changed"):
            check_intraop_threads(torch, 2)

    def test_empty_reports(self):
        with self.assertRaises(ValueError):
            summarize_runs({})
        with self.assertRaises(ValueError):
            summarize_runs({"auto": []})


if __name__ == "__main__":
    unittest.main()
