"""Small regression tests for the timeout diagnostic."""
import unittest

from src.live_rtsp import LatestSlot, wait_for_batch


class FakeReader:
    def __init__(self, camera):
        self.camera = camera
        self.slot = LatestSlot()
        self.error = None
        self.thread = None


class RTSPDiagnosticTests(unittest.TestCase):
    def test_timeout_identifies_missing_camera(self):
        readers = {c: FakeReader(c) for c in ("C1", "C2", "C3", "C4")}
        for c in ("C1", "C2", "C4"):
            readers[c].slot.put(object(), 1.0)
        with self.assertRaisesRegex(TimeoutError, r"C3: fresh=False, decoded=0"):
            wait_for_batch(readers, timeout=0.001, sleep=lambda _: None)

    def test_ready_batch(self):
        readers = {c: FakeReader(c) for c in ("C1", "C2", "C3", "C4")}
        for reader in readers.values():
            reader.slot.put(reader.camera, 1.0)
        self.assertEqual(set(wait_for_batch(readers, timeout=0.1)), set(readers))


if __name__ == "__main__":
    unittest.main()
