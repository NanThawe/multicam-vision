"""No RTSP server, FFmpeg, or ML model is needed for these unit tests."""

import unittest

from src.live_rtsp import LatestSlot, percentile, summarize, timeline_frame_id, wait_for_batch
from src.publish_rtsp import ffmpeg_command


class FakeReader:
    def __init__(self, camera):
        self.camera = camera
        self.slot = LatestSlot()
        self.error = None


class RTSPTests(unittest.TestCase):
    def test_latest_slot_replaces_old_unconsumed_frame(self):
        slot = LatestSlot()
        slot.put("first", 1.0)
        slot.put("second", 1.1)
        self.assertEqual(slot.counters(), (2, 1))
        self.assertEqual(slot.take().image, "second")
        self.assertIsNone(slot.take())
        slot.put("third", 1.2)
        self.assertEqual(slot.counters(), (3, 1))

    def test_receive_four_cameras(self):
        readers = {f"C{i}": FakeReader(f"C{i}") for i in range(1, 5)}
        for reader in readers.values():
            reader.slot.put(reader.camera, 10.0)
        batch = wait_for_batch(readers, timeout=0.1)
        self.assertEqual(set(batch), set(readers))
        self.assertEqual(batch["C3"].image, "C3")
        self.assertTrue(all(not reader.slot.has_frame() for reader in readers.values()))

    def test_wait_times_out_without_fresh_frames(self):
        readers = {"C1": FakeReader("C1")}
        with self.assertRaises(TimeoutError):
            wait_for_batch(readers, timeout=0.01, sleep=lambda _: None)

    def test_reader_errors_are_reported(self):
        reader = FakeReader("C1")
        reader.error = OSError("feed ended")
        with self.assertRaisesRegex(OSError, "feed ended"):
            wait_for_batch({"C1": reader}, timeout=.1)

    def test_timeline_is_monotonic(self):
        self.assertEqual(timeline_frame_id(10.0, 10.0, -1), 0)
        self.assertEqual(timeline_frame_id(10.0, 10.0, 0), 1)
        self.assertEqual(timeline_frame_id(10.2, 10.0, 1), 12)

    def test_report_separates_overwrites_from_processing(self):
        before = {c: (10, 2) for c in ("C1", "C2", "C3", "C4")}
        after = {c: (30, 6) for c in before}
        samples = [
            {"frame_age_ms": {c: 20 for c in before}, "processing_ms": 90},
            {"frame_age_ms": {c: 40 for c in before}, "processing_ms": 110},
        ]
        report = summarize(samples, before, after, 2, event_count=3,
                           mqtt_acknowledged=2, warmup_batches=5)
        self.assertEqual(report["aggregate_processed_fps"], 4)
        self.assertEqual(report["per_camera"]["C1"]["application_buffer_overwrites"], 4)
        self.assertEqual(report["per_camera"]["C1"]["decoded_frames_received"], 20)
        self.assertEqual(report["per_camera"]["C2"]["processed_frames"], 2)
        self.assertEqual(report["processing_ms_per_four_camera_batch"]["p50"], 100)
        self.assertEqual(report["events_generated"], 3)

    def test_ffmpeg_command_has_loop_pacing_and_tcp(self):
        cmd = ffmpeg_command("input with spaces.mp4", "C2", fps=5, width=960)
        self.assertEqual(cmd[cmd.index("-vf") + 1], "fps=5,scale=960:-2")
        self.assertEqual(cmd[cmd.index("-rtsp_transport") + 1], "tcp")
        self.assertTrue(cmd[-1].endswith("/C2"))
        self.assertEqual(cmd[cmd.index("-stream_loop") + 1], "-1")

    def test_bad_ffmpeg_scale_rejected(self):
        with self.assertRaises(ValueError):
            ffmpeg_command("test.mp4", "C1", width=961)


if __name__ == "__main__":
    unittest.main()


class ReportPersistenceTests(unittest.TestCase):
    """Saved reports must distinguish completed, failed and empty runs."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.cameras = ('C1', 'C2', 'C3', 'C4')
        self.start = {c: (0, 0) for c in self.cameras}
        self.end = {c: (10, 2) for c in self.cameras}
        self.samples = [{
            'frame_id': 12,
            'frame_age_ms': {c: 12 for c in self.cameras},
            'processing_ms': 100,
            'event_count': 1,
        }]

    def tearDown(self):
        self.temp.cleanup()

    def _save(self, samples, status, error=None):
        from src.live_rtsp import save_run_results
        return save_run_results(
            samples, self.start, self.end, 1.0, event_count=1,
            mqtt_acknowledged=0, warmup_batches=1, threads=4,
            status=status, error=error, output_dir=self.out,
        )

    def test_completed_run_writes_both_result_files(self):
        import json
        report = self._save(self.samples, 'completed')
        self.assertTrue((self.out / 'report.json').is_file())
        self.assertTrue((self.out / 'per_batch.csv').is_file())
        self.assertEqual(report['run_status'], 'completed')
        self.assertTrue(json.loads((self.out / 'run_status.json').read_text())['complete'])

    def test_failed_run_with_samples_saves_partial_report(self):
        report = self._save(self.samples, 'failed', 'TimeoutError: C2 stalled')
        self.assertEqual(report['measured_batches'], 1)
        self.assertFalse(report['complete'])
        self.assertIn('C2 stalled', report['error'])
        self.assertTrue((self.out / 'per_batch.csv').is_file())

    def test_zero_samples_writes_status_but_no_false_fps(self):
        import json
        report = self._save([], 'failed', 'TimeoutError: stream not ready')
        self.assertIsNone(report)
        self.assertFalse((self.out / 'report.json').exists())
        status = json.loads((self.out / 'run_status.json').read_text())
        self.assertEqual(status['measured_batches'], 0)
        self.assertEqual(status['status'], 'failed')
