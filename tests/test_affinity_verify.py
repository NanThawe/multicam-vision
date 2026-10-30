"""Synthetic Step 14.14 tests; no model weights, Windows APIs or RTSP required."""
import json
import tempfile
import struct
import unittest
from pathlib import Path
from unittest import mock

from src import affinity_verify as av
from src import fixed_inference as fi


class TopologyTests(unittest.TestCase):
    def test_no_windows_api_returns_reason(self):
        with mock.patch.object(av.sys, 'platform', 'linux'):
            cpus, reason=av.windows_cpu_sets()
        self.assertEqual(cpus, [])
        self.assertIn('Windows', reason)

    def test_parse_cpu_set_records_and_efficiency(self):
        # 32-byte Win32 SYSTEM_CPU_SET_INFORMATION records, with group-relative core IDs.
        a=struct.pack('<IIIHBBBBBBIQ',32,0,42,0,0,0,0,0,2,0,0,0)
        b=struct.pack('<IIIHBBBBBBIQ',32,0,43,0,1,1,0,0,0,0,0,0)
        rows=av.parse_cpu_set_buffer(a+b)
        self.assertEqual([(r['logical_cpu'],r['efficiency_class']) for r in rows],[(0,2),(1,0)])
        self.assertRaises(ValueError,av.parse_cpu_set_buffer,b'\x00'*8)

    def test_parse_skips_unknown_cpu_set_types(self):
        unknown=struct.pack('<II',8,99)
        good=struct.pack('<IIIHBBBBBBIQ',32,0,42,0,0,0,0,0,2,0,0,0)
        self.assertEqual(len(av.parse_cpu_set_buffer(unknown+good)),1)

    def test_select_distinct_cores_before_hyperthreads(self):
        records=[{'group':0,'logical_cpu':a,'core_index':core,'efficiency_class':cl}
                 for a,core,cl in [(0,0,2),(1,0,2),(2,1,2),(3,1,2),(4,2,2),(5,2,2),(6,3,2),(7,3,2)]]
        self.assertEqual(av.select_distinct_cores(records,4), [0,2,4,6])

    def test_distinct_cores_falls_back_to_siblings(self):
        records=[{'group':0,'logical_cpu':a,'core_index':core,'efficiency_class':1}
                 for a,core in [(0,0),(1,0),(2,1),(3,1)]]
        self.assertEqual(av.select_distinct_cores(records,4),[0,1,2,3])

    def test_high_and_low_efficiency_class_groups(self):
        rows=[dict(group=0,logical_cpu=i,core_index=i,efficiency_class=(2 if i%2==0 else 0),allocated=False)
              for i in range(12)]
        groups=av.candidate_groups(range(12),rows)
        self.assertEqual(groups['higher_efficiency_class4'],[0,2,4,6])
        self.assertEqual(groups['lower_efficiency_class4'],[1,3,5,7])
        self.assertEqual(groups['last4'],[8,9,10,11])

    def test_unclassified_has_only_exploratory_groups(self):
        rows=[dict(group=0,logical_cpu=i,core_index=i,efficiency_class=0,allocated=False)
              for i in range(12)]
        self.assertNotIn('higher_efficiency_class4', av.candidate_groups(range(12),rows))

    def test_no_duplicates_or_excluded_cpus(self):
        rows=[dict(group=0,logical_cpu=i,core_index=i,efficiency_class=2,
                   allocated=False) for i in (0,1,2,3)]
        self.assertEqual(list(av.candidate_groups([0,1,2,3],rows)),['all'])

    def test_trial_order_is_reversed_on_second_round(self):
        schedule=av.trial_order(['a','b','c'],3)
        self.assertEqual(schedule[0],['a','b','c'])
        self.assertEqual(schedule[1],['a','c','b'])
        self.assertNotEqual(schedule[1],schedule[2])
        self.assertRaises(ValueError,av.trial_order,['a'],1)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        fi.save_json(self.root/'fixture.json',{'frames_sha256':'fhash','boxes_sha256':'bhash'})
        self.groups={'all':[0,1,2,3,4,5,6,7], 'first4':[0,1,2,3]}
        self.topology={'candidate_groups':self.groups,'logical_cpus':8}

    def fake_executor(self, cmd, log_path):
        out=Path(cmd[cmd.index('--out')+1])
        ids=[int(x) for x in cmd[cmd.index('--affinity')+1].split(',')]
        fast=(len(ids)==4)
        iteration=12 if fast else 10
        fi.save_json(out/'summary.json',{'fixture_hashes':{'frames':'fhash','boxes':'bhash'},
            'aggregate_equivalent_fps':16 if fast else 9,
            'iterations':iteration,'cpu':{'affinity_actual':ids,'process_cpu_pct_one_core':380},
            'latency_ms':{'combined':{'p50':160 if fast else 450,
                                      'p95':195 if fast else 900,'p99':220 if fast else 1200},
                          'yolo':{'p50':70},'osnet':{'p50':90}}})
        return 0

    def test_sweep_and_comparison_choose_reproducible_group(self):
        with mock.patch.object(av,'topology',return_value=self.topology):
            trials=av.run_sweep(self.root, seconds=.1,repeats=3,executor=self.fake_executor)
        self.assertEqual(len(trials),6)
        self.assertTrue(all(t['complete'] for t in trials))
        report=av.summarize_trials(self.root)
        self.assertEqual(report['candidate_for_rtsp_validation'],'first4')
        self.assertEqual(report['candidate_cpu_ids'],[0,1,2,3])
        self.assertTrue((self.root/'affinity_per_run.csv').exists())

    def test_failed_trial_excluded_from_candidate(self):
        def fail_one(cmd, log_path):
            if 'round02_first4' in str(log_path):
                return 1
            return self.fake_executor(cmd, log_path)
        with mock.patch.object(av,'topology',return_value=self.topology):
            trials=av.run_sweep(self.root,seconds=.1,repeats=3,executor=fail_one)
        self.assertFalse(all(x['complete'] for x in trials))
        report=av.summarize_trials(self.root)
        self.assertEqual(report['candidate_for_rtsp_validation'],'all')

    def test_input_mismatch_rejects_run(self):
        def bad(cmd, log_path):
            code=self.fake_executor(cmd,log_path)
            out=Path(cmd[cmd.index('--out')+1])
            record=fi.load_json(out/'summary.json')
            record['fixture_hashes']['boxes']='corrupted'
            fi.save_json(out/'summary.json',record)
            return code
        with mock.patch.object(av,'topology',return_value=self.topology):
            trials=av.run_sweep(self.root,seconds=.1,repeats=2,executor=bad)
        self.assertTrue(all(not x['complete'] for x in trials))
        self.assertIsNone(av.summarize_trials(self.root)['candidate_for_rtsp_validation'])

    def test_affinity_not_applied_rejects_run(self):
        def bad(cmd,log_path):
            code=self.fake_executor(cmd,log_path)
            out=Path(cmd[cmd.index('--out')+1])
            row=fi.load_json(out/'summary.json')
            row['cpu']['affinity_actual']=[9]
            fi.save_json(out/'summary.json',row)
            return code
        with mock.patch.object(av,'topology',return_value=self.topology):
            trials=av.run_sweep(self.root,seconds=.1,repeats=2,executor=bad)
        self.assertTrue(all(not x['complete'] for x in trials))

    def test_rtsp_affinity_applied_before_import(self):
        with mock.patch.object(av,'apply_cpu_affinity',return_value=[0,1,2,3]) as affinity:
            with mock.patch('psutil.Process') as process:
                with mock.patch('src.live_rtsp.main') as live:
                    av.run_rtsp(affinity='0,1,2,3',seconds=5,threads=4,
                                output_dir=self.root/'rtsp',profile_reid=True)
        affinity.assert_called_once_with(process.return_value,'0,1,2,3')
        self.assertIn('--profile-reid',live.call_args.args[0])
        self.assertIn('--output-dir',live.call_args.args[0])

    def test_invalid_arguments(self):
        with mock.patch.object(av,'topology',return_value=self.topology):
            self.assertRaises(ValueError,av.run_sweep,self.root,repeats=1,executor=self.fake_executor)
            self.assertRaises(ValueError,av.run_sweep,self.root,seconds=0,executor=self.fake_executor)


if __name__=='__main__':
    unittest.main()
