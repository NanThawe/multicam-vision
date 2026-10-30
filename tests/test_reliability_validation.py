"""Offline regression tests for Step14.28; no RTSP, model or broker required."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import reliability_validation as r


def report(*, fps=20.3, overwrite=0, complete=True, wall=20.0):
    return {
        'aggregate_processed_fps': fps, 'wall_time_s': wall,
        'run_status': 'completed' if complete else 'error', 'complete': complete,
        'per_camera': {cam: {'decoded_frames_received': 110,
                             'processed_frames': 105,
                             'decoded_receipt_fps': 5.2,
                             'application_buffer_overwrites': overwrite}
                       for cam in r.CAMERAS},
        'processing_ms_per_four_camera_batch': {'p95': 183.2},
    }


def setup_trial(root, name='overlay_mqtt_01', *, fps=20.3, overwrite=0,
                verdict='PASS', elapsed=None, complete=True, resources=True):
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    r.e2e.write_json(out / 'report.json', report(fps=fps, overwrite=overwrite, complete=complete))
    r.e2e.write_json(out / 'validation.json', {
        'verdict': verdict,
        'mqtt_delivery': {'verdict': verdict, 'expected_jsonl_events': 2,
                          'unique_matching_received_ids': 2, 'missing_ids_count': 0,
                          'duplicate_delivery_ids_count': 0},
    })
    if elapsed is None:
        elapsed = [0.1 + i / 5.1 for i in range(102)]
    with (out / 'per_batch.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['elapsed_s','processing_ms','age_C1_ms',
                                          'age_C2_ms','age_C3_ms','age_C4_ms'])
        w.writeheader()
        for x in elapsed:
            w.writerow({'elapsed_s': x, 'processing_ms': 170,
                        **{f'age_{cam}_ms': 42 for cam in r.CAMERAS}})
    if resources:
        r.e2e.write_json(out / 'resources.json', r.resource_summary(sample(5), returncode=0))
    return out


def sample(n):
    return [{'utc_time':'test', 'elapsed_wall_s':40.0+i,
             'process_rss_mib':100.0+i, 'process_cpu_core_pct':250.0,
             'process_threads': 15, 'host_cpu_pct':50.0, 'host_memory_pct':60.0,
             'process_count':1} for i in range(n)]


class Helpers(unittest.TestCase):
    def test_percentile_empty(self):
        self.assertIsNone(r.percentile([], .5))

    def test_percentile_interpolates(self):
        self.assertEqual(r.percentile([10,20], .5), 15)
        self.assertEqual(r.percentile([1,2,3,4,5], .95), 4.8)

    def test_safe_float_finite(self):
        self.assertIsNone(r._safe_float('nan'))
        self.assertIsNone(r._safe_float('inf'))
        self.assertIsNone(r._safe_float(None))
        self.assertEqual(r._safe_float('2.1'), 2.1)

    def test_resource_summary_short_no_drift(self):
        result=r.resource_summary(sample(5), returncode=0)
        self.assertIsNone(result['steady_state_rss_delta_mib'])
        self.assertEqual(result['process_tree_rss_mib_peak'],104)
        self.assertEqual(result['process_tree_cpu_percent_of_one_core_p50'],250)

    def test_resource_summary_long_detects_descriptive_drift(self):
        result=r.resource_summary(sample(12), returncode=0)
        self.assertEqual(result['steady_state_rss_delta_mib'],9)
        self.assertEqual(result['sample_span_s'],11)

    def test_resource_summary_timeout(self):
        self.assertTrue(r.resource_summary([], timeout=True, returncode=124)['timed_out'])

    def test_camera_health_overwrites(self):
        c=r.camera_health(report(overwrite=11))
        self.assertEqual(c['C1']['overwrite_fraction_of_decoded'],.1)

    def test_camera_health_missing_gracefully(self):
        self.assertIsNone(r.camera_health({})['C1']['overwrite_fraction_of_decoded'])

    def test_percentile_fields_missing_columns(self):
        value=r.percentile_fields([{'other':'1'}], ['missing'])
        self.assertEqual(value['missing']['samples'],0)


class Metrics(unittest.TestCase):
    def test_perfect_run_full_windows(self):
        with tempfile.TemporaryDirectory() as d:
            out=setup_trial(d,elapsed=[.01+i*.19 for i in range(106)])
            v=r.metrics_for_run(out)
            self.assertTrue(v['runner_complete'])
            self.assertEqual(v['10s_complete_windows'],2)
            self.assertTrue(v['10s_all_windows_meet_target'])
            self.assertTrue(v['performance_target_met'])
            self.assertEqual(v['decode_receipt_to_processing_ms']['age_C1_ms']['p95'],42)
            self.assertTrue((out/'reliability_metrics.json').exists())

    def test_slow_window_is_independent_of_good_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            out=setup_trial(d,elapsed=[.1+i*.2 for i in range(20)] + [10.1+i*.15 for i in range(80)])
            v=r.metrics_for_run(out)
            self.assertEqual(v['10s_windows_below_target'],1)
            self.assertFalse(v['performance_target_met'])
            self.assertEqual(v['mqtt_delivery_verdict'],'PASS')

    def test_overwrites_reject_strict_performance(self):
        with tempfile.TemporaryDirectory() as d:
            out=setup_trial(d,overwrite=2,elapsed=[.01+i*.19 for i in range(106)])
            v=r.metrics_for_run(out)
            self.assertFalse(v['zero_application_buffer_overwrites'])
            self.assertEqual(v['total_application_buffer_overwrites'],8)
            self.assertFalse(v['performance_target_met'])

    def test_missing_report_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'overlay_mqtt_01'
            out.mkdir()
            v=r.metrics_for_run(out)
            self.assertFalse(v['runner_complete'])
            self.assertFalse(v['performance_target_met'])
            self.assertEqual(v['mqtt_delivery_verdict'],'MISSING')

    def test_missing_elapsed_field_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            out=setup_trial(d)
            (out/'per_batch.csv').write_text('processing_ms\n100\n',encoding='utf-8')
            v=r.metrics_for_run(out)
            self.assertFalse(v['10s_all_windows_meet_target'])

    def test_short_run_has_no_complete_windows(self):
        with tempfile.TemporaryDirectory() as d:
            out=setup_trial(d)
            dta=json.loads((out/'report.json').read_text())
            dta['wall_time_s']=8
            r.e2e.write_json(out/'report.json',dta)
            v=r.metrics_for_run(out)
            self.assertFalse(v['performance_target_met'])


class Aggregate(unittest.TestCase):
    def test_compare_two_all_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r.e2e.write_json(root/'manifest.json',{
                'configuration':{'repeats':2},
                'runs':[{'name':'overlay_mqtt_01','complete':True},
                        {'name':'overlay_mqtt_02','complete':True}],
            })
            setup_trial(root,'overlay_mqtt_01',elapsed=[.01+i*.19 for i in range(106)])
            setup_trial(root,'overlay_mqtt_02',elapsed=[.01+i*.19 for i in range(106)])
            v=r.compare(root)
            self.assertTrue(v['mqtt_all_pass'])
            self.assertTrue(v['strict_realtime_target_met'])
            self.assertEqual(v['fps_median'],20.3)
            self.assertTrue((root/'reliability_comparison.json').exists())
            self.assertTrue((root/'reliability_per_run.csv').exists())

    def test_compare_transport_pass_performance_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r.e2e.write_json(root/'manifest.json',{'configuration':{'repeats':2},
                'runs':[{'name':'overlay_mqtt_01'}, {'name':'overlay_mqtt_02'}]})
            setup_trial(root,'overlay_mqtt_01',fps=20.3,elapsed=[.01+i*.19 for i in range(106)])
            setup_trial(root,'overlay_mqtt_02',fps=18.8,overwrite=20,
                        elapsed=[.01+i*.22 for i in range(90)])
            v=r.compare(root)
            self.assertTrue(v['mqtt_all_pass'])
            self.assertFalse(v['strict_realtime_target_met'])
            self.assertEqual(v['fps_min'],18.8)

    def test_compare_inconclusive_transport(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r.e2e.write_json(root/'manifest.json',{'configuration':{'repeats':1},
                          'runs':[{'name':'overlay_mqtt_01'}]})
            setup_trial(root,verdict='INCONCLUSIVE',elapsed=[.01+i*.19 for i in range(106)])
            v=r.compare(root)
            self.assertEqual(v['mqtt_status'],'INCONCLUSIVE')
            self.assertFalse(v['mqtt_all_pass'])

    def test_compare_missing_trials_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r.e2e.write_json(root/'manifest.json',{'configuration':{'repeats':1},'runs':[]})
            with self.assertRaises(ValueError):
                r.compare(root)

    def test_existing_manifest_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            r.e2e.write_json(root/'manifest.json',{'sentinel':'unaltered'})
            with self.assertRaises(FileExistsError):
                r.run(root)
            self.assertEqual(r.e2e.load_json(root/'manifest.json'),{'sentinel':'unaltered'})

    def test_runner_wrapper_calls_existing_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest={'configuration':{'repeats':1},'runs':[{'name':'overlay_mqtt_01'}]}
            def mock_run(target, **kwargs):
                self.assertEqual(target,root)
                self.assertTrue(callable(kwargs['runner']))
                r.e2e.write_json(root/'manifest.json',manifest)
                setup_trial(root,elapsed=[.01+i*.19 for i in range(106)])
                return manifest
            with patch.object(r.e2e,'run_trials',side_effect=mock_run):
                with patch.object(r.e2e,'compare',return_value={}):
                    v=r.run(root,seconds=120,repeats=1)
            self.assertEqual(v['trial_count'],1)

    def test_run_rejects_bad_monitor_interval(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                r.run(Path(d),seconds=50,repeats=1,monitor_interval=0)


class Monitor(unittest.TestCase):
    def test_real_subprocess_samples_resources(self):
        # This smoke test uses a tiny Python subprocess, no model or network.
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            cmd=[sys.executable,'-c','import time; a=bytearray(2_000_000); time.sleep(.15)']
            code=r.monitored_runner(cmd,out/'console.log',seconds=1,
                                    interval_s=.05,preflight_each_run=False)
            self.assertEqual(code,0)
            result=r.e2e.load_json(out/'resources.json')
            self.assertGreaterEqual(result['samples'],2)
            self.assertGreater(result['process_tree_rss_mib_peak'],0)
            self.assertFalse(result['timed_out'])
            self.assertTrue((out/'resources.csv').exists())

    def test_per_trial_stream_preflight_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(r.e2e,'probe_streams',return_value={'C1':{'ready':False}}):
                with self.assertRaises(RuntimeError):
                    r.monitored_runner(['fake'],Path(d)/'console.log',seconds=1)
            self.assertTrue((Path(d)/'rtsp_preflight.json').exists())


if __name__=='__main__':
    unittest.main()
