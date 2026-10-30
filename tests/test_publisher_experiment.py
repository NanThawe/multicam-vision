"""Step 14.12 tests, synthetic unless explicitly using local FFmpeg binaries."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

from src import publisher_experiment as pe
from src import cpu_contention as monitor


class PublisherCommands(unittest.TestCase):
    def test_legacy_command_preserved_without_hidden_runtime_change(self):
        commands = pe.publisher_commands("live", "/tmp/out", {"C1":"raw.mp4"},
                                         rtsp_base="rtsp://127.0.0.1:8554")
        cmd = commands["C1"]
        self.assertIn("-vf", cmd)
        self.assertIn("fps=5,scale=960:-2", cmd)
        self.assertIn("libx264", cmd)
        self.assertIn("-re", cmd)

    def test_copy_command_does_not_decode_scale_or_encode(self):
        cmd = pe.copy_command("pre.mp4", "C1")
        self.assertEqual(cmd[cmd.index("-c:v")+1], "copy")
        self.assertNotIn("-vf", cmd)
        self.assertNotIn("libx264", cmd)
        self.assertIn("-re", cmd)
        self.assertEqual(cmd[-1], "rtsp://127.0.0.1:8554/C1")

    def test_offline_encoding_matches_resolution_and_codec(self):
        cmd = pe.preparation_command("src.mp4", "pre.mp4", ffmpeg="ffmpeg")
        self.assertEqual(cmd[cmd.index("-c:v")+1], "libx264")
        self.assertIn("fps=5,scale=960:-2", cmd)
        self.assertIn("zerolatency", cmd)
        self.assertNotIn("-re", cmd)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            pe.preparation_command("x", "y", width=959)
        with self.assertRaises(ValueError):
            pe.publisher_commands("invalid", "/tmp/x", {}, rtsp_base="rtsp://x")

    def test_rate_and_format(self):
        good={"codec_name":"h264","width":960,"height":540,"avg_frame_rate":"5/1"}
        self.assertTrue(pe.check_video(good, 5, 960))
        self.assertFalse(pe.check_video(dict(good, avg_frame_rate="10/1"), 5, 960))
        self.assertFalse(pe.check_video(dict(good, codec_name="mpeg4"), 5, 960))
        self.assertIsNone(pe.rate("0/0"))

    def test_probe_uses_tcp_for_rtsp(self):
        response=SimpleNamespace(returncode=0,stdout=json.dumps({"streams":[{"codec_name":"h264","width":960,"avg_frame_rate":"5/1"}]}),stderr="")
        with patch.object(pe.subprocess,"run",return_value=response) as fake:
            pe.probe_video("rtsp://127.0.0.1:8554/C1")
        self.assertIn("-rtsp_transport",fake.call_args.args[0])
        self.assertIn("tcp",fake.call_args.args[0])

    def test_cache_manifest_invalidates_on_source_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            src=root/'src.mp4'; src.write_bytes(b'fake')
            dest=pe.prepared_video(root,'C1');dest.parent.mkdir(parents=True)
            dest.write_bytes(b'encoded')
            sig={'source':str(src.resolve()),'source_bytes':src.stat().st_size,
                 'source_mtime_ns':src.stat().st_mtime_ns,'fps':5,'width':960}
            monitor.save_json(dest.parent/'manifest.json',{'fps':5,'width':960,
              'cameras':{'C1':{'source_signature':sig}}})
            fake_probe={"codec_name":"h264","width":960,"avg_frame_rate":"5/1"}
            with patch.object(pe,'probe_video',return_value=fake_probe):
                pe.verified_prepared(root,sources={'C1':src})
                src.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):
                    pe.verified_prepared(root,sources={'C1':src})

    def test_compare_preserves_limits_and_cpu_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for mode,fps,cpu in [('live',4,420),('copy',20,70)]:
                out=p/mode/'reid';out.mkdir(parents=True)
                monitor.save_json(out/'analysis.json',{
                    'run_status':'completed', 'aggregate_fps':fps,'batch_p95_ms':150,
                    'aligned_windows':12,'windows_below_20fps':1,
                    'host_during_measurement':{'ffmpeg_cpu_p50_one_core':cpu,
                                               'python_cpu_p50_one_core':250,
                                               'python_rss_p50_mb':350},
                    'inference_process_verified_samples':100,
                    'ffmpeg_scoped_to_publishers':True,'warnings':[]})
                monitor.save_json(p/mode/'publisher_trial.json',{'completed':True})
            result=pe.compare(p)
            self.assertEqual(result['delta_copy_minus_live_fps'],16)
            self.assertEqual(result['delta_copy_minus_live_publisher_cpu_one_core'],-350)
            self.assertTrue((p/'comparison.json').is_file())


class InferencePID(unittest.TestCase):
    class Proc:
        def __init__(self,pid,rss,children=None):
            self.pid=pid;self.rss=rss;self.descendants=children or []
        def name(self): return 'python.exe'
        def cmdline(self): return ['python.exe','-m','src.live_rtsp']
        def memory_info(self): return SimpleNamespace(rss=self.rss)
        def children(self,recursive=True): return self.descendants
        def create_time(self): return 123.0 + self.pid

    def test_real_child_beats_virtualenv_launcher(self):
        actual=self.Proc(222,700*1024**2)
        launcher=self.Proc(111,6*1024**2,[actual])
        target,verified=monitor.resolve_inference_process(launcher)
        self.assertEqual(target.pid,222)
        self.assertTrue(verified)

    def test_descendant_with_rewritten_argv_is_selected(self):
        child=self.Proc(222,700*1024**2)
        child.cmdline=lambda: ['C:/Python/python.exe','internal-bootstrap.py']
        launcher=self.Proc(111,6*1024**2,[child])
        target,verified=monitor.resolve_inference_process(launcher)
        self.assertEqual(target.pid,222)
        self.assertTrue(verified)

    def test_windows_tiny_shim_not_verified(self):
        shim=self.Proc(111,5*1024**2)
        with patch.object(monitor.sys,'platform','win32'):
            _,valid=monitor.resolve_inference_process(shim)
        self.assertFalse(valid)

    def test_cpu_percent_process_object_reused(self):
        real=self.Proc(222,400*1024**2)
        launcher=self.Proc(111,7*1024**2,[real])
        cache={}
        first,_=monitor.resolve_inference_process(launcher,cache)
        second,_=monitor.resolve_inference_process(launcher,cache)
        self.assertIs(first,second)

    def test_unverified_process_has_no_inference_cpu(self):
        sample=monitor.read_host_sample(None,{},verified=False,launcher_pid=1,
                                        ffmpeg_pids=set())
        self.assertIsNone(sample['python_cpu_pct_one_core'])
        self.assertFalse(sample['inference_pid_verified'])
        self.assertTrue(sample['ffmpeg_scoped_to_publishers'])


class OfflineEncoderSmoke(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Requires FFmpeg')
    def test_prepared_mp4_is_h264_960px_5fps_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sources={}
            for cam in ('C1','C2'):
                source=root/f'{cam}.mp4'
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
                                '-f','lavfi','-i','testsrc2=size=320x180:rate=10',
                                '-t','1.5','-c:v','libx264','-pix_fmt','yuv420p',str(source)],
                               check=True,stdout=subprocess.DEVNULL)
                sources[cam]=source
            manifest=pe.prepare(root,sources=sources)
            self.assertEqual(set(manifest['cameras']),{'C1','C2'})
            for cam in sources:
                path=pe.prepared_video(root,cam)
                self.assertTrue(pe.check_video(pe.probe_video(path),5,960))
            cache_mtimes={cam:pe.prepared_video(root,cam).stat().st_mtime_ns for cam in sources}
            second=pe.prepare(root,sources=sources)
            self.assertEqual(set(second['cameras']),{'C1','C2'})
            self.assertEqual(cache_mtimes,{cam:pe.prepared_video(root,cam).stat().st_mtime_ns for cam in sources})


if __name__=='__main__':
    unittest.main()
