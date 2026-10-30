"""Step 14.13 tests. No YOLO/BoxMOT weights or RTSP server required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import fixed_inference as fi


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def detect(self, frames):
        self.calls += 1
        return {c: [dict(xyxy=[20, 15, 70, 95], confidence=0.9, class_id=0),
                    dict(xyxy=[23, 15, 29, 20], confidence=0.7, class_id=0)]
                for c in frames}


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode_batch(self, requests):
        self.calls += 1
        return [np.full((len(people), 16), 0.1, dtype=np.float32)
                for _, _, _, people in requests]


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frames = {c: np.zeros((110, 140, 3), dtype=np.uint8) for c in fi.CAMERAS}
        self.provenance = {c: {'source_path':'synthetic', 'frame_id':0} for c in fi.CAMERAS}

    def prepare(self):
        with mock.patch.object(fi, 'load_source_images', return_value=(self.frames,self.provenance)):
            return fi.prepare(self.root, detector=FakeDetector())

    def test_select_boxes_valid_only(self):
        det = FakeDetector().detect(self.frames)
        boxes = fi.select_boxes(det, self.frames)
        self.assertEqual([len(v) for v in boxes.values()], [1]*4)

    def test_select_boxes_reject_no_crops(self):
        self.assertRaises(ValueError,fi.select_boxes, {c:[] for c in fi.CAMERAS},self.frames)

    def test_select_boxes_requires_positive_cap(self):
        self.assertRaises(ValueError,fi.select_boxes, FakeDetector().detect(self.frames),self.frames,0)

    def test_preparation_roundtrip(self):
        original = self.prepare()
        frames,boxes,fixture = fi.load_fixture(self.root)
        self.assertEqual(original['total_crops'],4)
        self.assertEqual(len(boxes['C1']),1)
        self.assertTrue(np.array_equal(frames['C2'],self.frames['C2']))
        self.assertEqual(fixture['frames_sha256'],fi.sha256_file(self.root/'frames.npz'))

    def test_cannot_silently_overwrite_fixture(self):
        self.prepare()
        with mock.patch.object(fi, 'load_source_images', return_value=(self.frames,self.provenance)):
            self.assertRaises(FileExistsError,fi.prepare,self.root,detector=FakeDetector())

    def test_corrupted_boxes_detected(self):
        self.prepare()
        (self.root/'boxes.json').write_text('{}',encoding='utf-8')
        self.assertRaises(ValueError,fi.load_fixture,self.root)

    def test_corrupted_image_detected(self):
        self.prepare()
        (self.root/'frames.npz').write_bytes(b'bad')
        self.assertRaises(ValueError,fi.load_fixture,self.root)

    def test_fixed_input_measurement_fake_models(self):
        self.prepare()
        report = fi.measure(self.root, seconds=0.025, warmup=1, threads=2,
                            detector_factory=FakeDetector, encoder_factory=FakeEncoder,
                            out=self.root/'runs'/'fake')
        self.assertEqual(report['status'],'completed')
        self.assertGreater(report['iterations'],0)
        self.assertEqual(report['crops_per_four_camera_batch'],4)
        self.assertIsNotNone(report['latency_ms']['yolo']['p50'])
        self.assertIsNotNone(report['latency_ms']['osnet']['p50'])
        self.assertEqual(report['cpu']['intraop_actual'],2)
        self.assertTrue((self.root/'runs'/'fake'/'per_iteration.csv').exists())

    def test_measure_detector_only_does_not_build_encoder(self):
        self.prepare()
        report = fi.measure(self.root, seconds=0.012, warmup=0, threads=2,
                            workload='detector', detector_factory=FakeDetector,
                            encoder_factory=lambda: (_ for _ in ()).throw(AssertionError()),
                            out=self.root/'runs'/'detector')
        self.assertEqual(report['crops_per_four_camera_batch'],0)
        self.assertIsNone(report['latency_ms']['osnet']['p50'])

    def test_measure_encoder_only_does_not_build_detector(self):
        self.prepare()
        report = fi.measure(self.root, seconds=0.012, warmup=0, threads=2,
                            workload='encoder', encoder_factory=FakeEncoder,
                            detector_factory=lambda: (_ for _ in ()).throw(AssertionError()),
                            out=self.root/'runs'/'encoder')
        self.assertIsNone(report['latency_ms']['yolo']['p50'])
        self.assertGreater(report['crops_per_four_camera_batch'],0)

    def test_compare_reports_incomplete(self):
        self.prepare()
        fi.save_json(self.root/'sweep_manifest.json',{'runs':[{'name':'failed','completed':False,'returncode':1}]})
        report = fi.compare(self.root)
        self.assertFalse(report['cases'][0]['completed'])
        self.assertIsNone(report['fixture_hashes'])

    def test_compare_detects_mismatched_hashes(self):
        self.prepare()
        for name, identity in [('a','one'),('b','two')]:
            fi.save_json(self.root/'runs'/name/'summary.json',
                    {'fixture_hashes':{'frames':identity,'boxes':'common'},
                     'aggregate_equivalent_fps':3,
                     'latency_ms':{k:{'p50':1,'p95':1} for k in ('yolo','osnet','combined')},
                     'cpu':{'process_cpu_pct_one_core':100,'affinity_actual':[0,1]}})
        fi.save_json(self.root/'sweep_manifest.json',{'runs':[{'name':'a','completed':True},
                                                              {'name':'b','completed':True}]})
        self.assertRaises(RuntimeError,fi.compare,self.root)

    def test_compare_preserves_metrics(self):
        self.prepare()
        fixture=fi.load_json(self.root/'fixture.json')
        fi.save_json(self.root/'runs'/'a'/'summary.json',
                    {'fixture_hashes':{'frames':fixture['frames_sha256'],'boxes':fixture['boxes_sha256']},
                     'aggregate_equivalent_fps':22,
                     'latency_ms':{k:{'p50':v,'p95':v*1.5} for k,v in [('yolo',60),('osnet',80),('combined',140)]},
                     'cpu':{'process_cpu_pct_one_core':290,'affinity_actual':[0,1]}})
        fi.save_json(self.root/'sweep_manifest.json',{'runs':[{'name':'a','completed':True,'threads':4,'affinity':'all'}]})
        result=fi.compare(self.root)
        self.assertEqual(result['cases'][0]['osnet_p50_ms'],80)
        self.assertTrue((self.root/'comparison.csv').exists())


class PureTest(unittest.TestCase):
    def test_affinity_all(self):
        self.assertEqual(fi.affinity_selection([7,3,1,9],'all'),[1,3,7,9])

    def test_affinity_first_last(self):
        self.assertEqual(fi.affinity_selection(list(range(8)),'first4'),[0,1,2,3])
        self.assertEqual(fi.affinity_selection(list(range(8)),'last4'),[4,5,6,7])

    def test_affinity_custom_subset(self):
        self.assertEqual(fi.affinity_selection([0,2,4,6],'6,2'),[2,6])
        self.assertRaises(ValueError,fi.affinity_selection,[0,2,4],'1,3')

    def test_affinity_cannot_be_empty(self):
        self.assertRaises(ValueError,fi.affinity_selection,[], 'all')

    def test_percentile_interpolation(self):
        self.assertEqual(fi.percentile([1,2,3,4],.5),2.5)
        self.assertIsNone(fi.percentile([],.9))

    def test_correlation(self):
        self.assertEqual(fi.correlation([1,2,3,4],[10,20,30,40]),1.0)
        self.assertIsNone(fi.correlation([1,1,1],[1,2,3]))

    def test_scenario_order(self):
        cases = fi.scenarios(with_publishers='copy')
        self.assertEqual(cases[0][0],'baseline_a')
        self.assertEqual(cases[-1][0],'baseline_after_publishers')
        self.assertEqual(cases[-2][3],'copy')

    def test_scenario_bad_mode(self):
        self.assertRaises(ValueError,fi.scenarios,with_publishers='rtmp')


if __name__ == '__main__':
    unittest.main()
