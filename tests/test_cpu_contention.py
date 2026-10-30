"""Step 14.11: CPU-only synthetic tests; no RTSP/model required."""
import csv
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import src.cpu_contention as monitor
from pathlib import Path

from src.cpu_contention import (
    median, correlation, join_batches, build_windows, analyse,
    compare, save_csv, save_json, FIELDS,
)


def batch(elapsed, detector=80, encode=90, encoded=8):
    return {"elapsed_s": elapsed, "detector_ms": detector,
            "reid_ms": encode+5, "reid_boxmot_encode_ms": encode,
            "reid_encoded_crops": encoded, "processing_ms": detector + encode+8}


def host(absolute, cpu=40, python=350, ffmpeg=60):
    return {"timestamp_unix_s":absolute, "system_cpu_pct":cpu,
            "python_cpu_pct_one_core":python, "ffmpeg_cpu_pct_one_core":ffmpeg,
            "docker_cpu_pct_one_core":10, "python_threads":13,
            "python_rss_mb":500}


class HostTelemetryTests(unittest.TestCase):
    def test_median_ignores_missing(self):
        self.assertEqual(median([5, None, '', 7]), 6)
        self.assertIsNone(median([None, '']))

    def test_correlation_exact_and_flat(self):
        self.assertEqual(correlation([1,2,3,4], [4,6,8,10]), 1)
        self.assertIsNone(correlation([1,2,3], [2,4,6]))
        self.assertIsNone(correlation([1,2,3,4], [5,5,5,5]))

    def test_align_on_actual_measurement_time_not_subprocess_start(self):
        rows=[batch(1),batch(2)]
        samples=[host(500, cpu=20),host(1001, cpu=40),host(1002, cpu=80)]
        joined=join_batches(rows,samples,1000)
        self.assertEqual(joined[0]['system_cpu_pct'],40)
        self.assertEqual(joined[1]['system_cpu_pct'],80)

    def test_distant_sample_not_misattributed(self):
        joined=join_batches([batch(10)],[host(1000)],1000,max_gap_s=0.5)
        self.assertIsNone(joined[0]['system_cpu_pct'])
        self.assertEqual(joined[0]['host_sample_gap_s'],10)

    def test_empty_host_samples_still_keep_batches(self):
        joined=join_batches([batch(1)],[],1000)
        self.assertEqual(len(joined),1)
        self.assertIsNone(joined[0]['host_sample_gap_s'])

    def test_windows_exclude_incomplete_final_window(self):
        windows=build_windows([batch(1), batch(10), batch(11)],
                              [host(1001),host(1011)],1000,1021,width_s=10)
        self.assertEqual(len(windows),2)
        self.assertEqual(windows[0]['batches'],2)
        self.assertEqual(windows[1]['batches'],1)
        self.assertEqual(windows[0]['aggregate_fps'],0.8)

    def test_windows_detect_slow_batches(self):
        windows=build_windows([batch(2,detector=500),batch(7,encode=400),batch(9)],
                              [host(1002), host(1007)],1000,1011)
        self.assertEqual(windows[0]['slow_batches'],2)
        self.assertEqual(windows[0]['detector_p50_ms'],80)

    def test_missing_timing_has_no_false_join_or_correlation(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            save_json(p/'report.json', {'complete':True,'run_status':'completed'})
            save_csv(p/'host_samples.csv',[host(1000)],FIELDS)
            result=analyse(p)
            self.assertEqual(result['aligned_windows'],0)
            self.assertIn('measurement_timing',result['warnings'][0])
            self.assertFalse((p/'joined_batches.csv').exists())

    def test_analysis_writes_window_join_and_ignores_cold_start(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            save_json(p/'report.json', {
                'run_status':'completed','complete':True,'aggregate_processed_fps':1.2,
                'processing_ms_per_four_camera_batch':{'p95':800},
                'measurement_timing':{'start_unix_s':1000,'end_unix_s':1021}})
            save_csv(p/'per_batch.csv', [batch(1),batch(6),batch(11)],
                     ['elapsed_s','detector_ms','reid_ms','reid_boxmot_encode_ms',
                      'reid_encoded_crops','processing_ms'])
            save_csv(p/'host_samples.csv',[host(900,cpu=2),host(1001,cpu=20),
                      host(1006,cpu=30),host(1011,cpu=50)],FIELDS)
            result=analyse(p)
            self.assertEqual(result['aligned_windows'],2)
            self.assertEqual(result['host_during_measurement']['system_cpu_p50_pct'],37.5)
            self.assertTrue((p/'joined_batches.csv').is_file())
            self.assertTrue((p/'windows.csv').is_file())
            with (p/'windows.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_compare_retains_two_modes_without_inventing_causality(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            for mode in ('reid','no_reid'):
                (p/mode).mkdir()
                save_json(p/mode/'analysis.json',{'complete':True,'aggregate_fps':10,
                          'aligned_windows':2,'host_during_measurement':{},
                          'pearson_window_correlations':{},'warnings':[]})
            result=compare(p)
            self.assertEqual(result['runs']['reid']['aggregate_fps'],10)
            self.assertIn('Non-synchronised',result['limitations'])
            self.assertTrue((p/'comparison.json').is_file())

    def test_monitored_subprocess_smoke_without_rtsp(self):
        """Real subprocess and host polling against a tiny stand-in runner."""
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'src').mkdir()
            (p/'src'/'__init__.py').write_text('')
            (p/'src'/'live_rtsp.py').write_text('''
import argparse, csv, json, time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--output-dir',required=True)
a,_=ap.parse_known_args()
p=Path(a.output_dir);p.mkdir(parents=True,exist_ok=True)
start=time.time()
time.sleep(1.3)
end=time.time()
(p/'report.json').write_text(json.dumps({'complete':True,'run_status':'completed',
 'aggregate_processed_fps':8, 'measurement_timing':{'start_unix_s':start,'end_unix_s':end},
 'processing_ms_per_four_camera_batch':{'p95':150}}))
with (p/'per_batch.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['elapsed_s','detector_ms','reid_ms','reid_boxmot_encode_ms','reid_encoded_crops','processing_ms'])
 w.writeheader()
 for t in (0.3,0.6,0.9,1.2):
  w.writerow({'elapsed_s':t,'detector_ms':80,'reid_ms':90,
              'reid_boxmot_encode_ms':85,'reid_encoded_crops':8,'processing_ms':180})
''',encoding='utf-8')
            args = SimpleNamespace(root=p/'results',mode='reid',seconds=1.3,
                                   warmup_batches=0,threads=4,sample_interval=.25,
                                   window_s=.5,rtsp_base='rtsp://127.0.0.1:8554')
            with patch.object(monitor,'ROOT',p):
                result = monitor.run(args)
            self.assertEqual(result,0)
            analysis=json.loads((p/'results'/'reid'/'analysis.json').read_text())
            self.assertGreaterEqual(analysis['host_samples_captured'],3)
            self.assertEqual(analysis['aligned_windows'],2)
            self.assertTrue((p/'results'/'reid'/'console.log').exists())


if __name__ == '__main__':
    unittest.main()
