"""Step 14.28 -- repeated four-camera RTSP + MQTT reliability / resource benchmark.

Uses the unchanged Step 14.18 subscriber-verified runner, without modifying YOLO,
ByteTrack, OSNet, production matching, or event logic.

Examples:
  python -m src.reliability_validation preflight
  python -m src.reliability_validation run --seconds 30 --repeats 1 --root outputs/rtsp/step14_28_smoke
  python -m src.reliability_validation run --seconds 180 --repeats 3 --root outputs/rtsp/step14_28
  python -m src.reliability_validation compare --root outputs/rtsp/step14_28
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from . import end_to_end_validation as e2e

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / 'outputs' / 'rtsp' / 'step14_28'
CAMERAS = ('C1', 'C2', 'C3', 'C4')
THROUGHPUT_TARGET_FPS = 20.0
WINDOW_SECONDS = 10.0
SAMPLE_FIELDS = ('utc_time', 'elapsed_wall_s', 'process_rss_mib',
                 'process_cpu_core_pct', 'process_threads',
                 'host_cpu_pct', 'host_memory_pct', 'process_count')


def percentile(values, q):
    if not values:
        return None
    seq = sorted(float(x) for x in values)
    pos = (len(seq) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(seq[lo] + (seq[hi] - seq[lo]) * (pos - lo), 3)


def _median(values):
    return round(statistics.median(values), 3) if values else None


def _safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def _tree_snapshot(root_proc, psutil):
    """Collect process + child RSS / cumulative CPU without assuming a single PID."""
    try:
        procs = [root_proc, *root_proc.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        procs = [root_proc]
    rss = 0
    cpu = 0.0
    threads = 0
    seen = set()
    for proc in procs:
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        try:
            times = proc.cpu_times()
            rss += proc.memory_info().rss
            cpu += float(times.user + times.system)
            threads += proc.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return {'rss_mib': round(rss / 1048576.0, 3),
            'cpu_s': cpu, 'threads': threads, 'count': len(seen)}


def resource_summary(samples, *, timeout=False, returncode=None, interval_s=1.0):
    rss = [r['process_rss_mib'] for r in samples if r.get('process_rss_mib') is not None]
    cpu = [r['process_cpu_core_pct'] for r in samples if r.get('process_cpu_core_pct') is not None]
    host = [r['host_cpu_pct'] for r in samples if r.get('host_cpu_pct') is not None]
    host_mem = [r['host_memory_pct'] for r in samples if r.get('host_memory_pct') is not None]
    threads = [r['process_threads'] for r in samples if r.get('process_threads') is not None]
    # Exclude start-up and warm-up for a descriptive drift estimate. Not a leak test.
    steady = [r for r in samples if r.get('elapsed_wall_s', 0) >= 30.0
              and r.get('process_rss_mib') is not None]
    drift = None
    if len(steady) >= 8:
        k = max(2, len(steady) // 4)
        drift = round(statistics.median(r['process_rss_mib'] for r in steady[-k:]) -
                      statistics.median(r['process_rss_mib'] for r in steady[:k]), 3)
    return {
        'returncode': returncode, 'timed_out': bool(timeout),
        'sample_interval_s': interval_s, 'samples': len(samples),
        'sample_span_s': (round(samples[-1]['elapsed_wall_s'] - samples[0]['elapsed_wall_s'], 3)
                          if len(samples) >= 2 else None),
        'process_tree_rss_mib_p50': percentile(rss, .5),
        'process_tree_rss_mib_p95': percentile(rss, .95),
        'process_tree_rss_mib_peak': max(rss) if rss else None,
        'steady_state_rss_delta_mib': drift,
        'process_tree_cpu_percent_of_one_core_p50': percentile(cpu, .5),
        'process_tree_cpu_percent_of_one_core_p95': percentile(cpu, .95),
        'process_tree_threads_max': max(threads) if threads else None,
        'host_cpu_pct_p95': percentile(host, .95),
        'host_memory_pct_p95': percentile(host_mem, .95),
        'limitations': [
            'CPU percentage is relative to one logical CPU; 400% is approximately four fully busy cores.',
            'RSS combines the inference process and its descendants, not externally running FFmpeg publishers, Docker or Mosquitto.',
            'RSS change is descriptive and is not proof of a memory leak; startup and warmup are excluded only approximately.',
            'Host CPU and memory are samples, not performance counters for individual publishers.',
        ]
    }


def _terminate_tree(process, psutil):
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for p in reversed(children):
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        for p in reversed(children):
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        process.kill()
        process.wait(timeout=5)


def monitored_runner(command, log_path, *, seconds, timeout_grace=180.0, interval_s=1.0,
                     preflight_each_run=True, rtsp_base='rtsp://127.0.0.1:8554'):
    """Invoked by Step14.18 after MQTT SUBACK. Samples live child resources."""
    import psutil
    if interval_s <= 0 or timeout_grace <= 0:
        raise ValueError('interval_s and timeout_grace must be positive')
    log_path = Path(log_path)
    out = log_path.parent
    if preflight_each_run:
        result = e2e.probe_streams(rtsp_base)
        e2e.write_json(out / 'rtsp_preflight.json', result)
        if not all(result.get(cam, {}).get('ready') for cam in CAMERAS):
            raise RuntimeError('One or more camera streams missing immediately before run')
    started = time.monotonic()
    previous = None
    samples = []
    timed_out = False
    process = None
    psutil.cpu_percent(interval=None)  # prime non-blocking system sampler
    try:
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            root_proc = psutil.Process(process.pid)
            while True:
                wall = time.monotonic()
                snap = _tree_snapshot(root_proc, psutil)
                cpu_pct = None
                if previous is not None:
                    elapsed = wall - previous['wall']
                    if elapsed > 0:
                        # This is percent of one CPU, not percent of total host CPUs.
                        cpu_pct = round(max(0.0, snap['cpu_s'] - previous['cpu_s']) / elapsed * 100.0, 2)
                previous = {'wall': wall, 'cpu_s': snap['cpu_s']}
                samples.append({
                    'utc_time': datetime.now(timezone.utc).isoformat(),
                    'elapsed_wall_s': round(wall - started, 3),
                    'process_rss_mib': snap['rss_mib'],
                    'process_cpu_core_pct': cpu_pct,
                    'process_threads': snap['threads'],
                    'host_cpu_pct': round(psutil.cpu_percent(interval=None), 2),
                    'host_memory_pct': round(psutil.virtual_memory().percent, 2),
                    'process_count': snap['count'],
                })
                if process.poll() is not None:
                    break
                if wall - started > seconds + timeout_grace:
                    timed_out = True
                    _terminate_tree(process, psutil)
                    break
                time.sleep(interval_s)
        returncode = 124 if timed_out else process.wait()
        return returncode
    finally:
        if process is not None and process.poll() is None:
            _terminate_tree(process, psutil)
        with (out / 'resources.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(samples)
        code = 124 if timed_out else (process.poll() if process is not None else None)
        e2e.write_json(out / 'resources.json',
                       resource_summary(samples, timeout=timed_out, returncode=code,
                                        interval_s=interval_s))


def camera_health(report):
    per = report.get('per_camera', {})
    result = {}
    for cam in CAMERAS:
        item = per.get(cam, {})
        rec = _safe_float(item.get('decoded_frames_received'))
        proc = _safe_float(item.get('processed_frames'))
        overwritten = _safe_float(item.get('application_buffer_overwrites'))
        result[cam] = {
            'decoded_receipt_fps': _safe_float(item.get('decoded_receipt_fps')),
            'processed_frames': proc, 'decoded_frames_received': rec,
            'application_buffer_overwrites': overwritten,
            'overwrite_fraction_of_decoded': round(overwritten / rec, 5)
                   if rec is not None and rec > 0 and overwritten is not None else None,
        }
    return result


def _load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def percentile_fields(rows, names):
    output = {}
    for field in names:
        values = [n for row in rows if (n := _safe_float(row.get(field))) is not None]
        output[field] = {'samples': len(values), 'p50': percentile(values, .5),
                         'p95': percentile(values, .95), 'max': max(values) if values else None}
    return output


def metrics_for_run(out, *, target_fps=THROUGHPUT_TARGET_FPS):
    out = Path(out)
    report = e2e.load_json(out / 'report.json') if (out / 'report.json').exists() else {}
    validation = e2e.load_json(out / 'validation.json') if (out / 'validation.json').exists() else {}
    resources = e2e.load_json(out / 'resources.json') if (out / 'resources.json').exists() else {}
    batch_rows = _load_rows(out / 'per_batch.csv')
    ages = percentile_fields(batch_rows, [f'age_{cam}_ms' for cam in CAMERAS])
    stages = percentile_fields(batch_rows,
             ['processing_ms','wait_for_batch_ms','detector_ms','tracker_ms','reid_ms',
              'event_io_mqtt_ms','preview_ms'])
    fps = _safe_float(report.get('aggregate_processed_fps'))
    wall_s = _safe_float(report.get('wall_time_s'))
    window = e2e.windows_for_run(out / 'per_batch.csv', wall_s or 0, WINDOW_SECONDS)
    cameras = camera_health(report)
    overwrite_total = sum(x['application_buffer_overwrites'] for x in cameras.values()
                          if x['application_buffer_overwrites'] is not None)
    available_windows = bool(window)
    all_windows_meet = available_windows and all(w['aggregate_fps'] >= target_fps for w in window)
    no_overwrites = (all(x['application_buffer_overwrites'] == 0 for x in cameras.values()))
    report_ok = (report.get('complete') is True and report.get('run_status') == 'completed')
    transport = validation.get('mqtt_delivery', {})
    delivery_status = transport.get('verdict', validation.get('verdict', 'MISSING'))
    stage_stats = report.get('processing_ms_per_four_camera_batch', {})
    score = {
        'trial': out.name,
        'runner_complete': report_ok,
        'mqtt_delivery_verdict': delivery_status,
        'mqtt_generated_events': transport.get('expected_jsonl_events'),
        'mqtt_unique_matched_received': transport.get('unique_matching_received_ids'),
        'mqtt_missing_ids': transport.get('missing_ids_count'),
        'mqtt_duplicate_delivery_ids': transport.get('duplicate_delivery_ids_count'),
        'aggregate_processed_fps': fps,
        'target_fps': target_fps,
        'mean_fps_target_met': bool(report_ok and fps is not None and fps >= target_fps),
        '10s_complete_windows': len(window),
        '10s_windows_below_target': sum(w['aggregate_fps'] < target_fps for w in window),
        '10s_worst_fps': min((w['aggregate_fps'] for w in window), default=None),
        '10s_all_windows_meet_target': all_windows_meet,
        'processing_batch_p95_ms': stage_stats.get('p95'),
        'decode_receipt_to_processing_ms': ages,
        'stage_times_ms': stages,
        'per_camera': cameras,
        'total_application_buffer_overwrites': overwrite_total,
        'zero_application_buffer_overwrites': no_overwrites,
        'cpu_memory': resources,
        'performance_target_met': bool(report_ok and fps is not None and fps >= target_fps
                                       and all_windows_meet and no_overwrites),
        'throughput_window_definition': 'Complete 10-second intervals from per_batch elapsed_s (four processed frames per batch).',
        'limits': [
            'Frame age is Python decode receipt to processing start, not camera exposure to display or network latency.',
            'Application overwrites exclude source, network, FFmpeg and decoder frame losses.',
            'Short 10-second windows at nominal 5 FPS can fluctuate; strict target is reported separately.',
            'MQTT delivery refers to one independent local subscriber, not exactly-once or persistence.',
            'Zone/line event semantic accuracy and global identity accuracy are not measured.',
        ]
    }
    e2e.write_json(out / 'reliability_metrics.json', score)
    return score


def compare(root=DEFAULT_ROOT, *, target_fps=THROUGHPUT_TARGET_FPS):
    root = Path(root)
    manifest = e2e.load_json(root / 'manifest.json')
    entries = manifest.get('runs', [])
    if not entries:
        raise ValueError('No completed or attempted runs in manifest')
    metrics = [metrics_for_run(root / item['name'], target_fps=target_fps) for item in entries]
    rows = []
    for entry, item in zip(entries, metrics):
        r = item['cpu_memory']
        rows.append({
            'trial': item['trial'], 'runner_complete': item['runner_complete'],
            'transport_verdict': item['mqtt_delivery_verdict'],
            'fps': item['aggregate_processed_fps'],
            'worst_10s_fps': item['10s_worst_fps'],
            'windows_below_target': item['10s_windows_below_target'],
            'overwrites': item['total_application_buffer_overwrites'],
            'batch_p95_ms': item['processing_batch_p95_ms'],
            'cpu_p50_single_core_pct': r.get('process_tree_cpu_percent_of_one_core_p50'),
            'rss_peak_mib': r.get('process_tree_rss_mib_peak'),
            'steady_rss_delta_mib': r.get('steady_state_rss_delta_mib'),
            'events_generated': item['mqtt_generated_events'],
            'unique_mqtt_received': item['mqtt_unique_matched_received'],
            'mqtt_missing': item['mqtt_missing_ids'],
            'mqtt_duplicates': item['mqtt_duplicate_delivery_ids'],
            'strict_performance_met': item['performance_target_met'],
        })
    fps = [r['fps'] for r in rows if r['fps'] is not None]
    transport = [r['transport_verdict'] for r in rows]
    all_complete = all(r['runner_complete'] for r in rows) and len(rows) == manifest.get('configuration', {}).get('repeats')
    report = {
        'experiment': 'Step14.28 repeated live RTSP / event delivery / resource reliability validation',
        'trial_count': len(rows), 'target_fps': target_fps,
        'all_trials_completed': all_complete,
        'mqtt_all_pass': all(t == 'PASS' for t in transport),
        'mqtt_any_inconclusive': any(t == 'INCONCLUSIVE' for t in transport),
        'all_trial_average_fps_meet_target': all_complete and len(fps) == len(rows) and all(f >= target_fps for f in fps),
        'all_complete_10s_windows_meet_target': all_complete and all(m['10s_all_windows_meet_target'] for m in metrics),
        'all_trials_zero_application_overwrites': all_complete and all(m['zero_application_buffer_overwrites'] for m in metrics),
        'strict_realtime_target_met': all_complete and all(m['performance_target_met'] for m in metrics),
        'fps_min': min(fps) if fps else None,
        'fps_median': _median(fps),
        'fps_max': max(fps) if fps else None,
        'fps_stdev_sample': round(statistics.stdev(fps), 3) if len(fps) > 1 else None,
        'mqtt_status': ('PASS' if all(t == 'PASS' for t in transport) else
                        'INCONCLUSIVE' if all(t in ('PASS', 'INCONCLUSIVE') for t in transport) else 'FAIL'),
        'trials': rows,
        'limits': [
            'Separates broker/subscriber MQTT transport reliability from the strict realtime-throughput target.',
            'No capture-to-display timestamps or network loss counters; frame age begins at decode receipt.',
            'Host CPU sampling includes unrelated processes and four publishers; process-tree RSS excludes external publishers.',
            'Unchanged RTSP configuration and publishers across runs are an operator responsibility.',
            'No tracking IDF1/HOTA, event ground-truth correctness or global ID accuracy assessed.',
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'reliability_per_run.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    e2e.write_json(root / 'reliability_comparison.json', report)
    return report


def run(root=DEFAULT_ROOT, *, seconds=180, repeats=3, threads=4,
        affinity='0,2,4,6,8,10', rtsp_base='rtsp://127.0.0.1:8554',
        mqtt_host='127.0.0.1', mqtt_port=1883, monitor_interval=1.0,
        timeout_grace=180.0):
    root = Path(root)
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Existing experiment at {root}. Choose a new --root.')
    if seconds <= 10 or repeats < 1 or monitor_interval <= 0 or timeout_grace <= 0:
        raise ValueError('seconds > 10, repeats >= 1, monitor_interval > 0, timeout_grace > 0 required')
    def runner(command, logfile):
        return monitored_runner(command, logfile, seconds=seconds,
                                interval_s=monitor_interval, timeout_grace=timeout_grace,
                                rtsp_base=rtsp_base)
    manifest = e2e.run_trials(root, seconds=seconds, repeats=repeats, threads=threads,
                              affinity=affinity, rtsp_base=rtsp_base, mqtt_host=mqtt_host,
                              mqtt_port=mqtt_port, runner=runner)
    e2e.compare(root)  # keep original Step 14.18 output / transport parity check
    return compare(root)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('preflight', help='Probe C1-C4 RTSP and MQTT SUBACK')
    a.add_argument('--rtsp-base', default='rtsp://127.0.0.1:8554')
    a.add_argument('--mqtt-host', default='127.0.0.1')
    a.add_argument('--mqtt-port', type=int, default=1883)
    b = sub.add_parser('run', help='Run separate real-system trials and monitor CPU/RSS')
    b.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    b.add_argument('--seconds', type=float, default=180)
    b.add_argument('--repeats', type=int, default=3)
    b.add_argument('--threads', type=int, default=4)
    b.add_argument('--affinity', default='0,2,4,6,8,10')
    b.add_argument('--rtsp-base', default='rtsp://127.0.0.1:8554')
    b.add_argument('--mqtt-host', default='127.0.0.1')
    b.add_argument('--mqtt-port', type=int, default=1883)
    b.add_argument('--monitor-interval', type=float, default=1.0)
    b.add_argument('--timeout-grace', type=float, default=180.0)
    c = sub.add_parser('compare', help='Rebuild reports from an existing run directory')
    c.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    c.add_argument('--target-fps', type=float, default=THROUGHPUT_TARGET_FPS)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.command == 'preflight':
        result = e2e.validate_setup(rtsp_base=a.rtsp_base,
                                     mqtt_host=a.mqtt_host, mqtt_port=a.mqtt_port)
        print(json.dumps(result, indent=2))
        if not result['ready']:
            raise SystemExit(1)
    elif a.command == 'run':
        try:
            report = run(a.root, seconds=a.seconds, repeats=a.repeats,
                         threads=a.threads, affinity=a.affinity,
                         rtsp_base=a.rtsp_base, mqtt_host=a.mqtt_host,
                         mqtt_port=a.mqtt_port, monitor_interval=a.monitor_interval,
                         timeout_grace=a.timeout_grace)
        except (FileExistsError, RuntimeError, ValueError) as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            raise SystemExit(1) from exc
        print(json.dumps({k: report[k] for k in ('trial_count','fps_min','fps_median','mqtt_status',
                      'strict_realtime_target_met')}, indent=2))
        if not report['all_trials_completed'] or report['mqtt_status'] == 'FAIL':
            raise SystemExit(1)
    else:
        if a.target_fps <= 0:
            raise SystemExit('--target-fps must be positive')
        print(json.dumps(compare(a.root, target_fps=a.target_fps), indent=2))


if __name__ == '__main__':
    main()
