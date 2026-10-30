"""Step14.29 tests; synthetic source artifacts, never a claim about WILDTRACK."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import demo_release as d


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'step14_28'
        self.trial = self.root / 'overlay_mqtt_01'
        self.trial.mkdir(parents=True)
        self.out = Path(self.tmp.name) / 'demo'
        d.write_json(self.root / 'manifest.json', {'runs': [{'name': 'overlay_mqtt_01', 'returncode': 0}]})
        self.report = {
            'complete': True, 'run_status': 'completed', 'cameras': list(d.CAMERAS),
            'measured_batches': 3, 'warmup_batches': 2, 'wall_time_s': 0.7,
            'aggregate_processed_fps': 17.14, 'processed_fps_per_camera': 4.285,
            'events_generated': 2, 'processing_ms_per_four_camera_batch': {'p95': 300},
            'provisional_preview': {'enabled': True, 'video_playback_fps': 5,
                                    'rendered_batches_including_warmup': 5},
            'per_camera': {cam: {'application_buffer_overwrites': 2,
                                'processed_frames': 3,'decoded_frames_received': 5} for cam in d.CAMERAS},
        }
        self.delivery = {'verdict':'PASS', 'mqtt_delivery': {
            'verdict': 'PASS', 'expected_jsonl_events': 2,
            'unique_matching_received_ids': 2, 'broker_qos1_acks': 2,
            'duplicate_delivery_ids_count': 0}}
        d.write_json(self.trial / 'report.json', self.report)
        d.write_json(self.trial / 'validation.json', self.delivery)
        d.write_json(self.trial / 'reliability_metrics.json', {'performance_target_met': False})
        d.write_json(self.root / 'reliability_comparison.json', {'trial_count': 3,
            'fps_min': 15.196,'fps_median':16.683,'fps_max':16.839,
            'mqtt_status':'PASS','strict_realtime_target_met':False})
        with (self.trial / 'per_batch.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['frame_id','processing_ms','event_count'])
            w.writeheader()
            w.writerows([{'frame_id': i, 'processing_ms': 300, 'event_count': 1 if i>9 else 0}
                         for i in (10,20,30)])
        self.events = [
            {'event_id':'C1:3:20:zone_entry','camera':'C1','event_type':'zone_entry',
             'track_id':3,'global_id':3, 'frame_id':20, 'elapsed_s':0.35, 'rule_id':'zone_1'},
            {'event_id':'C2:6:30:line_crossing','camera':'C2','event_type':'line_crossing',
             'track_id':6,'global_id':None,'frame_id':30,'elapsed_s':0.6, 'rule_id':'line_1'},
        ]
        (self.trial / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events), encoding='utf-8')
        (self.trial / 'provisional_preview.mp4').write_bytes(b'synthetic video bytes')

    @staticmethod
    def converter(source, dest, executable):
        # Synthetic video fixture; do not make a claim about video playback.
        Path(dest).write_bytes(b'synthetic exported bytes')

    def test_serialization_maps_event_frames_including_warmup(self):
        data = d.serialize_events(self.events, {10:0,20:1,30:2}, 2, 5)
        self.assertEqual([e['video_offset_s'] for e in data], [0.6,0.8])

    def test_delivery_verification(self):
        folder, report, val = d.verified_trial(self.root, 'overlay_mqtt_01')
        self.assertEqual(folder, self.trial)
        self.assertEqual(val['verdict'], 'PASS')

    def test_full_build_writes_readonly_evidence(self):
        result = d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)
        self.assertEqual(result['events'], 2)
        self.assertEqual(result['mqtt'], 'PASS')
        payload = d.read_json(self.out / 'data.json')
        self.assertEqual(payload['metrics']['buffer_overwrites'], 8)
        self.assertFalse(payload['metrics']['target_met'])
        self.assertEqual(payload['events'][0]['video_offset_s'], 0.6)
        self.assertIn('C1', payload['rules'])
        self.assertIn('source_artifact_hashes_sha256', d.read_json(self.out / 'evidence_manifest.json'))
        self.assertEqual((self.trial / 'provisional_preview.mp4').read_bytes(), b'synthetic video bytes')
        self.assertIn('RECORDED REPLAY', (self.out / 'index.html').read_text(encoding='utf-8'))

    def test_build_refuses_overwrite(self):
        d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)
        with self.assertRaises(FileExistsError):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_missing_previews_fail_closed(self):
        (self.trial / 'provisional_preview.mp4').unlink()
        with self.assertRaises(FileNotFoundError):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_mqtt_not_pass_refused(self):
        self.delivery['verdict'] = 'FAIL'
        d.write_json(self.trial / 'validation.json', self.delivery)
        with self.assertRaises(ValueError):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_bad_event_frame_refused(self):
        self.events[0]['frame_id'] = 999
        (self.trial / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
        with self.assertRaisesRegex(ValueError, 'absent'):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_duplicate_event_id_refused(self):
        self.events[1]['event_id'] = self.events[0]['event_id']
        (self.trial / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_report_count_mismatch_refused(self):
        self.report['events_generated'] = 5
        d.write_json(self.trial / 'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'Event counts'):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_missing_full_run_refused(self):
        self.report['complete'] = False
        d.write_json(self.trial / 'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'not completed'):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_preview_misalignment_refused(self):
        self.report['provisional_preview']['rendered_batches_including_warmup'] = 12
        d.write_json(self.trial / 'report.json', self.report)
        with self.assertRaisesRegex(ValueError, 'frame count'):
            d.build(self.root, 'overlay_mqtt_01', self.out, converter=self.converter)

    def test_trial_path_validation(self):
        with self.assertRaisesRegex(ValueError, 'trial'):
            d.verified_trial(self.root, '../overlay_mqtt_01')

    def test_bad_camera_and_event_type(self):
        self.events[0]['camera'] = 'C5'
        with self.assertRaisesRegex(ValueError, 'Invalid event'):
            d.serialize_events(self.events, {10:0,20:1,30:2}, 2, 5)

    def test_convert_missing_ffmpeg_errors(self):
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg was not found'):
            d.ffmpeg_convert(self.trial / 'provisional_preview.mp4', self.out / 'out.mp4', '__no_such_ffmpeg__')

    def test_data_contains_unverified_identity_caveat(self):
        from src.events import RULES
        data = d.make_data(self.report, self.delivery, d.serialize_events(self.events, {10:0,20:1,30:2}, 2, 5), 'overlay_mqtt_01', self.root)
        self.assertTrue(any('NOT verified' in s for s in data['limitations']))

    def test_all_camera_rules_serialized(self):
        self.assertEqual(set(d.rules_for_dashboard()), set(d.CAMERAS))

    def test_jsonl_errors_include_line(self):
        f = self.trial / 'bad.jsonl'
        f.write_text('{}\nnot json\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            d.read_jsonl(f)


if __name__ == '__main__':
    unittest.main()
