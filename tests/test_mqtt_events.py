"""Offline tests for MQTT event generation, ordering and routing."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from src.mqtt_events import publish_file, read_events, replay_events


SAMPLE = {
    "event_id": "C1:4:12:zone_entry",
    "event_type": "zone_entry",
    "camera": "C1",
    "track_id": 4,
    "global_id": 8,
    "frame_id": 12,
    "elapsed_s": 0.2,
    "footpoint": [0.4, 0.5],
}


class MQTTEventsTest(unittest.TestCase):
    def write_jsonl(self, records):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "events.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        return path

    def test_publish_waits_for_qos1_acknowledgements(self):
        path = self.write_jsonl([SAMPLE])
        sent = []
        acknowledgements = []

        class FakeInfo:
            rc = 0

            def wait_for_publish(self, timeout):
                acknowledgements.append(timeout)

            def is_published(self):
                return True

        class FakeClient:
            def connect(self, *_args, **_kwargs):
                return 0

            def loop_start(self):
                self.on_connect(self, None, None,
                                SimpleNamespace(is_failure=False), None)

            def publish(self, *args, **kwargs):
                sent.append((args, kwargs))
                return FakeInfo()

            def disconnect(self):
                pass

            def loop_stop(self):
                pass

        mqtt = SimpleNamespace(MQTT_ERR_SUCCESS=0)
        with patch("src.mqtt_events.mqtt_client", return_value=(mqtt, FakeClient())):
            self.assertEqual(publish_file(path=path, speed=0), 1)
        self.assertEqual(sent[0][1], {"qos": 1, "retain": False})
        self.assertEqual(sent[0][0][0], "multicam/events/C1/zone_entry")
        self.assertEqual(json.loads(sent[0][0][1]), SAMPLE)
        self.assertEqual(acknowledgements, [10])

    def test_topic_and_payload_are_preserved(self):
        sent = []
        n = replay_events([SAMPLE], lambda topic, payload: sent.append((topic, payload)), speed=0)
        self.assertEqual(n, 1)
        self.assertEqual(sent[0][0], "multicam/events/C1/zone_entry")
        self.assertEqual(json.loads(sent[0][1]), SAMPLE)

    def test_replay_timing(self):
        events = [SAMPLE, {**SAMPLE, "event_id": "second", "elapsed_s": 0.8}]
        delays = []
        n = replay_events(events, lambda _topic, _payload: None,
                          speed=2, sleep=delays.append)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(delays[0], 0.3)

    def test_rejects_unknown_camera(self):
        path = self.write_jsonl([{**SAMPLE, "camera": "unknown"}])
        with self.assertRaisesRegex(ValueError, "Unknown camera"):
            list(read_events(path))

    def test_rejects_non_chronological_events(self):
        path = self.write_jsonl([SAMPLE, {**SAMPLE, "elapsed_s": 0.1}])
        with self.assertRaisesRegex(ValueError, "ordered"):
            list(read_events(path))

    def test_missing_file_has_explanation(self):
        with self.assertRaisesRegex(FileNotFoundError, "src.events"):
            list(read_events("missing-events.jsonl"))

    def test_speed_must_not_be_negative(self):
        with self.assertRaisesRegex(ValueError, "speed"):
            replay_events([], lambda *_: None, speed=-1)


if __name__ == "__main__":
    unittest.main()
