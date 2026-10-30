"""Step 14.10 CPU-free unit tests for OSNet cache/latency observability."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.live_rtsp import (REID_FIELDS, reid_snapshot, reid_delta,
                           summarize_reid_details, save_run_results)


CAMERAS = ('C1', 'C2', 'C3', 'C4')
STAGES = ('wait_for_batch','detector','tracker','provisional','reid',
          'events','event_io_mqtt','preview','thread_check')


def mock_matcher():
    return SimpleNamespace(
        stats={'eligible_crops': 10, 'encoded_crops': 6,
               'reused_crops': 4, 'encoder_calls': 2},
        local={'a': 1, 'b': 2}, global_ids={1: 1},
        encoder=SimpleNamespace(timings_ms={
            'input_prep': 15.0, 'model_encode': 120.0, 'output_convert': 3.0}))


def sample(encoded, reused, time_ms, idx=0):
    profile = {
        'eligible_crops': encoded + reused, 'encoded_crops': encoded,
        'reused_crops': reused, 'encoder_calls': int(bool(encoded)),
        'new_local_states': 1, 'new_global_ids': 1,
        'input_prep_ms': 1.0 if encoded else 0,
        'boxmot_encode_ms': 0.8*time_ms if encoded else 0,
        'output_convert_ms': 0.5 if encoded else 0,
        'other_reid_ms': time_ms * 0.1 if encoded else time_ms,
    }
    return {'frame_id': idx,'elapsed_s': idx + 1,
            'frame_age_ms': {c: 10.0 for c in CAMERAS},
            'processing_ms': 100.0 + time_ms, 'event_count': 0,
            'stage_ms': {name: (time_ms if name == 'reid' else 1.0)
                         for name in STAGES},
            'counters': {c: (10+idx, 1) for c in CAMERAS},
            'reid_detail': profile}


class ReidTelemetryTests(unittest.TestCase):
    def test_snapshot_is_immutable_copy(self):
        matcher = mock_matcher()
        before = reid_snapshot(matcher)
        matcher.stats['encoded_crops'] += 3
        self.assertEqual(before['encoded_crops'], 6)

    def test_snapshot_with_no_reid(self):
        self.assertEqual(reid_snapshot(None)['encoder_calls'], 0)
        self.assertEqual(reid_snapshot(None)['model_encode'], 0.0)

    def test_delta_counts_and_encode_timings(self):
        matcher = mock_matcher()
        before = reid_snapshot(matcher)
        matcher.stats.update(eligible_crops=15, encoded_crops=9,
                             reused_crops=6, encoder_calls=3)
        matcher.local['c'] = 3
        matcher.global_ids[2] = 2
        matcher.encoder.timings_ms.update(input_prep=17.0,model_encode=240.0,
                                           output_convert=4.0)
        output = reid_delta(before, reid_snapshot(matcher), 150.0)
        self.assertEqual((output['eligible_crops'], output['encoded_crops'],
                          output['reused_crops'], output['encoder_calls']), (5, 3, 2, 1))
        self.assertEqual(output['new_local_states'], 1)
        self.assertEqual(output['new_global_ids'], 1)
        self.assertEqual(output['boxmot_encode_ms'], 120)
        self.assertEqual(output['other_reid_ms'], 27.0)

    def test_delta_zero_for_no_reid(self):
        blank = reid_snapshot(None)
        result = reid_delta(blank, blank, 0.05)
        self.assertEqual(result['eligible_crops'], 0)
        self.assertEqual(result['other_reid_ms'], .05)

    def test_mismatched_count_is_rejected(self):
        matcher = mock_matcher()
        before = reid_snapshot(matcher)
        matcher.stats['eligible_crops'] += 5
        with self.assertRaisesRegex(ValueError, 'balance'):
            reid_delta(before, reid_snapshot(matcher), 1)

    def test_missing_encoder_timers_error(self):
        matcher = mock_matcher()
        matcher.encoder.timings_ms = None
        with self.assertRaisesRegex(RuntimeError, 'instrumented'):
            reid_snapshot(matcher)

    def test_summary_cache_reuse_and_buckets(self):
        records = [sample(0, 5, 15,0), sample(3, 3, 40,1),
                   sample(8, 2, 190,2),sample(12, 0, 300,3)]
        s = summarize_reid_details(records)
        self.assertEqual(s['totals']['eligible_crops'], 33)
        self.assertEqual(s['totals']['encoded_crops'], 23)
        self.assertEqual(s['totals']['reused_crops'], 10)
        self.assertEqual(s['cache_reuse_pct'], 30.3)
        self.assertEqual(s['reid_stage_ms_by_encoded_crop_count']['zero']['batches'], 1)
        self.assertEqual(s['reid_stage_ms_by_encoded_crop_count']['1_to_4']['batches'], 1)
        self.assertEqual(s['reid_stage_ms_by_encoded_crop_count']['5_to_9']['batches'], 1)
        self.assertEqual(s['reid_stage_ms_by_encoded_crop_count']['10_plus']['batches'], 1)

    def test_zero_eligible_reports_null_rate(self):
        s = summarize_reid_details([sample(0,0,0)])
        self.assertIsNone(s['cache_reuse_pct'])
        self.assertIsNone(s['encoded_per_active_batch_p50'])

    def test_report_csv_telemetry_without_reid_module_import(self):
        records = [sample(2,4,40,0),sample(8,2,160,1)]
        before = {c: (0,0) for c in CAMERAS}
        after = {c: (12,1) for c in CAMERAS}
        with tempfile.TemporaryDirectory() as directory:
            save_run_results(records,before,after,2,event_count=0,
                             mqtt_acknowledged=0,warmup_batches=0,threads=4,
                             status='completed',output_dir=directory,
                             profile_stages=True,profile_reid=True)
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['reid_cache_diagnostics']['totals']['encoded_crops'],10)
            self.assertEqual(report['stage_timings_ms']['detector']['p50'], 1)
            with (Path(directory)/'per_batch.csv').open(newline='') as fh:
                csv_rows=list(csv.DictReader(fh))
            self.assertEqual(csv_rows[1]['reid_encoded_crops'], '8')
            self.assertEqual(csv_rows[0]['reid_boxmot_encode_ms'], '32.0')

    def test_normal_report_has_no_extra_fields(self):
        records = [sample(2, 1, 15)]
        before = {c: (0, 0) for c in CAMERAS}
        with tempfile.TemporaryDirectory() as directory:
            save_run_results(records, before, before, 1, event_count=0,
                             mqtt_acknowledged=0,warmup_batches=0,threads=4,
                             status='completed', output_dir=directory)
            report=json.loads((Path(directory)/'report.json').read_text())
            self.assertNotIn('reid_cache_diagnostics',report)
            with (Path(directory)/'per_batch.csv').open(newline='') as fh:
                self.assertNotIn('reid_encoded_crops',next(csv.DictReader(fh)))

if __name__ == '__main__':
    unittest.main()
