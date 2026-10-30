import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import end_to_end_validation as v


def record(eid='C1:4:12:zone_entry'):
    return {'event_id': eid, 'event_type': 'zone_entry', 'camera': 'C1',
            'track_id': 4, 'global_id': 88, 'frame_id': 12,
            'elapsed_s': 2.0, 'footpoint': [0.4, 0.5]}


def received(event=None, topic='multicam/events/C1/zone_entry', retain=False):
    return {'topic': topic, 'event': event or record(), 'retain': retain}


def report(count=1, *, complete=True, mqtt=True):
    return {'run_status': 'completed' if complete else 'failed', 'complete': complete,
            'mqtt_enabled': mqtt, 'events_generated': count, 'mqtt_qos1_acknowledged': count}


class DeliveryTests(unittest.TestCase):
    def test_positive_exact_match(self):
        self.assertEqual(v.compare_delivery([record()], [received()], report())['verdict'], 'PASS')

    def test_missing_is_fail(self):
        result = v.compare_delivery([record()], [], report())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['missing_ids_count'], 1)

    def test_broker_ack_without_subscriber_is_not_pass(self):
        self.assertEqual(v.compare_delivery([record()], [], report())['unique_matching_received_ids'], 0)

    def test_duplicate_qos1_is_not_falsely_exactly_once(self):
        result = v.compare_delivery([record()], [received(), received()], report())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['duplicate_delivery_ids_count'], 1)

    def test_unexpected_event_fails(self):
        wrong = record('C1:5:13:zone_entry')
        self.assertEqual(v.compare_delivery([record()], [received(), received(wrong)], report())['verdict'], 'FAIL')

    def test_payload_mismatch_fails(self):
        wrong = {**record(), 'global_id': 99}
        self.assertEqual(v.compare_delivery([record()], [received(wrong)], report())['verdict'], 'FAIL')

    def test_wrong_topic_fails(self):
        self.assertEqual(v.compare_delivery([record()], [received(topic='multicam/events/C2/zone_entry')],
                                            report())['verdict'], 'FAIL')

    def test_no_events_is_inconclusive(self):
        self.assertEqual(v.compare_delivery([], [], report(0))['verdict'], 'INCONCLUSIVE')

    def test_no_events_failed_run_is_fail(self):
        self.assertEqual(v.compare_delivery([], [], report(0, complete=False))['verdict'], 'FAIL')

    def test_ack_count_mismatch_fails(self):
        self.assertEqual(v.compare_delivery([record()], [received()], {**report(), 'mqtt_qos1_acknowledged':0})['verdict'], 'FAIL')

    def test_duplicate_source_ids_fail(self):
        self.assertEqual(v.compare_delivery([record(), record()], [received()], report(2))['verdict'], 'FAIL')

    def test_broker_disconnect_fails(self):
        self.assertEqual(v.compare_delivery([record()], [received()], report(),
                                            subscriber_errors=['broker disconnected'])['verdict'], 'FAIL')

    def test_retain_is_not_accepted(self):
        self.assertEqual(v.compare_delivery([record()], [received(retain=True)], report())['verdict'], 'FAIL')

    def test_report_not_mqtt_is_fail(self):
        self.assertEqual(v.compare_delivery([record()], [received()], report(mqtt=False))['verdict'], 'FAIL')

    def test_invalid_source_track_is_fail(self):
        ev = {**record(), 'track_id': None}
        self.assertEqual(v.compare_delivery([ev], [received(ev)], report())['verdict'], 'FAIL')


class WindowsTests(unittest.TestCase):
    def test_count_windows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'per_batch.csv'
            path.write_text('elapsed_s,frame_id\n0.1,1\n9.9,2\n10.0,3\n20.2,4\n', encoding='utf-8')
            windows = v.windows_for_run(path, seconds=25, interval=10)
            self.assertEqual([x['batches'] for x in windows], [2,1])
            self.assertEqual(windows[0]['aggregate_fps'], 0.8)

    def test_compare_fails_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v.write_json(root/'manifest.json', {'configuration': {'repeats':1},
                                                 'runs':[{'name':'overlay_mqtt_01','complete':False,'verdict':'FAIL'}]})
            value=v.compare(root)
            self.assertFalse(value['all_runs_passed'])
            self.assertFalse(value['all_runs_at_least_20fps'])

    def test_missing_report_is_fail(self):
        with tempfile.TemporaryDirectory() as d:
            result=v.evaluate_run(Path(d), [], [], 0)
            self.assertEqual(result['verdict'], 'FAIL')

    def test_trial_orchestration_writes_all_artifacts(self):
        class FakeCapture:
            errors = []
            def start(self): return self
            def snapshot(self): return [received()]
            def stop(self): pass

        def fake_runner(cmd, path):
            out = Path(cmd[cmd.index('--output-dir') + 1])
            out.mkdir(exist_ok=True)
            v.write_json(out/'report.json', {**report(),
                'aggregate_processed_fps': 20.5,
                'wall_time_s': 12.0,
                'per_camera': {x: {'application_buffer_overwrites':0} for x in v.CAMERAS},
                'processing_ms_per_four_camera_batch': {'p95':180}})
            (out/'events.jsonl').write_text(json.dumps(record())+'\n', encoding='utf-8')
            (out/'per_batch.csv').write_text('elapsed_s\n1\n2\n3\n', encoding='utf-8')
            Path(path).write_text('mock benchmark', encoding='utf-8')
            return 0

        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(v.time,'sleep', return_value=None):
                result=v.run_trials(root,seconds=12,repeats=2,preflight=False,
                    runner=fake_runner,capture_factory=FakeCapture)
            self.assertEqual(len(result['runs']),2)
            self.assertTrue(all(x['complete'] for x in result['runs']))
            self.assertEqual(v.compare(root)['passed_runs'],2)
            self.assertEqual(v.load_json(root/'overlay_mqtt_01'/'validation.json')['mqtt_delivery']['broker_qos1_acks'],1)

    def test_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v.write_json(root/'manifest.json', {'saved':'previous'})
            with self.assertRaises(FileExistsError):
                v.run_trials(root, seconds=12, preflight=False)
            self.assertEqual(v.load_json(root/'manifest.json')['saved'], 'previous')

    def test_preflight_no_streams(self):
        with patch.object(v,'probe_streams',return_value={'C1':{'ready':False}}):
            with patch.object(v.MQTTCapture, 'start', side_effect=ConnectionError('offline')):
                value=v.validate_setup(rtsp_base='x', mqtt_host='x', mqtt_port=1883)
                self.assertFalse(value['ready'])


if __name__=='__main__':
    unittest.main()
