import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from src import rtsp_affinity_validation as m


class TestRtspAffinityValidation(unittest.TestCase):
    def setUp(self):
        self.topology = {'allowed_affinity': list(range(20)),
                         'candidate_groups': {'first4': [0, 1, 2, 3]}}
        self.comparison = {'candidate_cpu_ids': [0, 2, 4, 6]}

    def test_groups(self):
        result = m.validated_groups(self.comparison, self.topology)
        self.assertEqual(result, {'all': 'all', 'candidate': '0,2,4,6', 'first4': '0,1,2,3'})

    def test_invalid_candidate(self):
        with self.assertRaises(ValueError):
            m.validated_groups({'candidate_cpu_ids': [0, 2, 4, 99]}, self.topology)

    def test_deduplicates_groups(self):
        result = m.validated_groups({'candidate_cpu_ids': [0, 1, 2, 3]}, self.topology)
        self.assertNotIn('first4', result)

    def test_order_reversal(self):
        seq = m.sequence({'all':'all','candidate':'0,2,4,6','first4':'0,1,2,3'}, 2)
        self.assertEqual([x['name'] for x in seq],
                         ['all', 'candidate', 'first4', 'first4', 'candidate', 'all'])

    def test_repeats_required(self):
        with self.assertRaises(ValueError):
            m.sequence({'all': 'all'}, 1)

    def test_time_windows_complete_only(self):
        with TemporaryDirectory() as d:
            path = Path(d) / 'x.csv'
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['elapsed_s','processing_ms'])
                w.writeheader()
                for t in [1, 2, 9.9, 10, 11, 19.9, 20, 21]:
                    w.writerow({'elapsed_s': t, 'processing_ms': 5})
            x = m.time_windows(path, wall_s=25, window_s=10)
            self.assertEqual([r['batches'] for r in x], [3, 3])
            self.assertEqual([r['aggregate_fps'] for r in x], [1.2,1.2])

    def test_run_and_compare_mocked(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            root = base / 'out'
            comp, topo = base / 'comparison.json', base / 'topology.json'
            m.write_json(comp, self.comparison)
            m.write_json(topo, self.topology)
            calls = []
            def simulate(cmd, log):
                calls.append(cmd)
                run_dir = Path(cmd[-1])
                m.write_json(run_dir / 'report.json', {
                    'complete': True, 'run_status':'completed','cameras':['C1','C2','C3','C4'],
                    'aggregate_processed_fps':20.8,'processed_fps_per_camera':5.2,
                    'wall_time_s':21.1,'per_camera':{k:{'application_buffer_overwrites':1}
                                                    for k in ('C1','C2','C3','C4')},
                    'processing_ms_per_four_camera_batch':{'p50':100,'p95':180},
                    'stage_timings_ms':{'detector':{'p50':75},'reid':{'p50':90}}
                })
                with (run_dir / 'per_batch.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['elapsed_s'])
                    for i in range(52):
                        w.writerow([(i+1)*0.4])
                return 0
            data = m.run(root, comparison=comp, topology=topo, seconds=21, repeats=2,
                         runner=simulate)
            self.assertEqual(len(data), 6)
            self.assertTrue(all(item['complete'] for item in data))
            self.assertTrue(all('--profile-reid' in cmd for cmd in calls))
            summary=m.aggregate(root)
            self.assertEqual(len(summary['groups']),3)
            self.assertEqual(summary['groups'][0]['completed_runs'],2)
            self.assertTrue((root/'time_windows.csv').exists())

    def test_reject_stale_report(self):
        with TemporaryDirectory() as d:
            base=Path(d)
            root=base/'out'
            comp,topo=base/'comp.json',base/'topo.json'
            m.write_json(comp,self.comparison);m.write_json(topo,self.topology)
            def failing(cmd,log):
                out=Path(cmd[-1]);m.write_json(out/'report.json',{'complete':False})
                return 1
            records=m.run(root,comparison=comp,topology=topo,seconds=20,repeats=2,
                          runner=failing)
            self.assertEqual(len(records),6)
            self.assertTrue(all(not x['complete'] for x in records))

if __name__ == '__main__':
    unittest.main()
