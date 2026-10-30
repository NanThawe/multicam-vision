"""Small deterministic event tests; no dataset or ML libraries required."""

import unittest

from src.events import EventEngine


RULES = {
    camera: {"zone": (0.3, 0.3, 0.7, 0.7),
             "line": ((0.5, 0.1), (0.5, 0.9))}
    for camera in ("C1", "C2")
}


def person(x, y=0.5, track_id=1, global_id=7):
    return {"track_id": track_id, "global_id": global_id,
            "xyxy": [(x - 0.02) * 100, (y - 0.15) * 100,
                     (x + 0.02) * 100, y * 100]}


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.engine = EventEngine(rules=RULES, width=100, height=100,
                                  source_fps=60, max_gap_frames=120)

    def test_zone_entry_crossing_and_exit(self):
        self.assertEqual(self.engine.update(0, "C1", [person(0.2)]), [])
        entry = self.engine.update(12, "C1", [person(0.4)])
        self.assertEqual([e["event_type"] for e in entry], ["zone_entry"])
        crossing = self.engine.update(24, "C1", [person(0.6)])
        self.assertEqual([e["event_type"] for e in crossing], ["line_crossing"])
        self.assertEqual(crossing[0]["direction"], "positive_to_negative")
        exit_events = self.engine.update(36, "C1", [person(0.8)])
        self.assertEqual([e["event_type"] for e in exit_events], ["zone_exit"])

    def test_initial_inside_does_not_trigger_entry(self):
        self.assertEqual(self.engine.update(0, "C1", [person(0.4)]), [])

    def test_gaps_reset_history(self):
        self.engine.update(0, "C1", [person(0.2)])
        self.assertEqual(self.engine.update(180, "C1", [person(0.8)]), [])

    def test_cameras_do_not_share_local_identity_state(self):
        self.engine.update(0, "C1", [person(0.2)])
        self.assertEqual(self.engine.update(12, "C2", [person(0.6)]), [])

    def test_infinite_line_extension_does_not_trigger(self):
        self.engine.update(0, "C1", [person(0.4, y=0.95)])
        result = self.engine.update(12, "C1", [person(0.6, y=0.95)])
        self.assertFalse(any(e["event_type"] == "line_crossing" for e in result))

    def test_hysteresis_avoids_jitter_event(self):
        self.engine.update(0, "C1", [person(0.49)])
        self.engine.update(12, "C1", [person(0.499)])
        self.assertEqual(self.engine.update(24, "C1", [person(0.501)]), [])


if __name__ == "__main__":
    unittest.main()
