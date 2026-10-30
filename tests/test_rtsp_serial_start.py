"""Unit tests for sequential RTSP reader startup, without a stream server."""
import time
import unittest

from src.live_rtsp import RTSPReader


class FakeCapture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable
        self.closed = False

    def isOpened(self):
        return self.opened

    def set(self, _key, _value):
        return False

    def read(self):
        time.sleep(0.005)
        return (True, object()) if self.readable else (False, None)

    def release(self):
        self.closed = True


class RTSPStartTests(unittest.TestCase):
    def test_waits_for_first_decoded_frame(self):
        capture = FakeCapture()
        reader = RTSPReader('C1', 'fake', capture_factory=lambda _: capture)
        try:
            reader.start(timeout=1.0)
            self.assertGreaterEqual(reader.slot.counters()[0], 1)
            self.assertIsNone(reader.error)
        finally:
            reader.stop()
        self.assertTrue(capture.closed)

    def test_unavailable_stream_reports_camera_error(self):
        capture = FakeCapture(opened=False)
        reader = RTSPReader('C2', 'fake', capture_factory=lambda _: capture)
        with self.assertRaisesRegex(OSError, 'C2'):
            reader.start(timeout=1.0)
        reader.stop()
        self.assertTrue(capture.closed)

    def test_read_failure_is_reported(self):
        capture = FakeCapture(readable=False)
        reader = RTSPReader('C3', 'fake', capture_factory=lambda _: capture)
        with self.assertRaisesRegex(OSError, 'C3'):
            reader.start(timeout=1.0)
        reader.stop()
        self.assertTrue(capture.closed)


if __name__ == '__main__':
    unittest.main()
