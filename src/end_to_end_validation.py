"""Step 14.18: subscriber-verified live RTSP event delivery and repeatability.

No changes to YOLO, ByteTrack, OSNet, the publisher, or EventEngine.
Run from the project root; see STEP14_18.md.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / 'outputs' / 'rtsp' / 'step14_18'
CAMERAS = ('C1', 'C2', 'C3', 'C4')
AFFINITY = '0,2,4,6,8,10'
TOPIC_ROOT = 'multicam/events'
EVENT_TYPES = {'zone_entry', 'zone_exit', 'line_crossing'}


def load_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def load_jsonl(path):
    path = Path(path)
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]



def event_identity(event):
    if not isinstance(event, dict):
        raise ValueError('Expected event JSON object')
    identifier = event.get('event_id')
    camera = event.get('camera')
    category = event.get('event_type')
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('Invalid event_id')
    if camera not in CAMERAS or category not in EVENT_TYPES:
        raise ValueError('Invalid camera or event type')
    # track_id belongs to a confirmed local tracker. It is not verified identity accuracy.
    if not isinstance(event.get('track_id'), int) or isinstance(event.get('track_id'), bool):
        raise ValueError('Invalid local track_id')
    return identifier, f'{TOPIC_ROOT}/{camera}/{category}'


def fingerprint(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def compare_delivery(events, received, report, *, child_exit=0, subscriber_errors=()):
    """Validate unique event IDs, payloads, broker acknowledgments and receipt.

    A QoS1 duplicate is counted separately, not treated as a failure of at-least-once delivery.
    An empty run is inconclusive, never a successful delivery validation.
    """
    expected = {}
    repeated_expected = 0
    invalid_expected = 0
    for event in events:
        try:
            eid, topic = event_identity(event)
        except ValueError:
            invalid_expected += 1
            continue
        if eid in expected:
            repeated_expected += 1
        else:
            expected[eid] = (topic, fingerprint(event))

    received_ids = Counter()
    received_valid = set()
    mismatched = unexpected = invalid_received = retained = 0
    for item in received:
        if item.get('retain'):
            retained += 1
        try:
            event = item['event']
            eid, computed_topic = event_identity(event)
            topic = item.get('topic')
            if computed_topic != topic:
                mismatched += 1
                continue
        except (KeyError, ValueError, TypeError):
            invalid_received += 1
            continue
        received_ids[eid] += 1
        if eid not in expected:
            unexpected += 1
        elif (topic, fingerprint(event)) != expected[eid]:
            mismatched += 1
        else:
            received_valid.add(eid)

    acked = report.get('mqtt_qos1_acknowledged', 0)
    recorded = report.get('events_generated')
    missing_ids = sorted(set(expected) - received_valid)
    duplicate_ids = sorted(eid for eid, count in received_ids.items() if count > 1)
    errors = list(subscriber_errors)
    run_ok = (child_exit == 0 and report.get('complete') is True
              and report.get('run_status') == 'completed'
              and report.get('mqtt_enabled') is True)
    counts_ok = (isinstance(acked, int) and not isinstance(acked, bool)
                 and isinstance(recorded, int) and not isinstance(recorded, bool)
                 and acked == recorded == len(events))
    data_ok = (not invalid_expected and not repeated_expected and not unexpected
               and not mismatched and not invalid_received and not retained)
    delivery_ok = (not missing_ids and not errors and data_ok and counts_ok and run_ok)
    verdict = 'PASS' if events and delivery_ok else ('INCONCLUSIVE' if not events and run_ok and not errors
                                                     and data_ok and counts_ok else 'FAIL')
    return {
        'verdict': verdict,
        'basis': 'QoS1 broker acknowledgements plus independent subscriber receipt; at-least-once only',
        'publisher_run_complete': run_ok,
        'reported_generated': recorded,
        'expected_jsonl_events': len(events),
        'broker_qos1_acks': acked,
        'subscriber_messages': len(received),
        'unique_matching_received_ids': len(received_valid),
        'missing_ids_count': len(missing_ids),
        'missing_ids_sample': missing_ids[:20],
        'duplicate_delivery_ids_count': len(duplicate_ids),
        'duplicate_delivery_ids_sample': duplicate_ids[:20],
        'unexpected_messages': unexpected,
        'mismatched_topic_or_payload': mismatched,
        'invalid_received': invalid_received,
        'retained_messages': retained,
        'duplicated_source_event_ids': repeated_expected,
        'invalid_source_events': invalid_expected,
        'subscriber_errors': errors,
        'notes': [
            'QoS1 provides at-least-once transport; duplicates are possible.',
            'A broker PUBACK is not proof that a separate subscriber received an event.',
            'Matching event IDs/payloads does not validate zone rules or cross-camera identity.',
            'The existing live_rtsp implementation keeps provisional observations out of EventEngine.',
            'Event-log auditing alone cannot prove that runtime separation; test production implementation.',
        ]
    }


class MQTTCapture:
    """Subscribe before launching the publisher and record received payloads."""

    def __init__(self, host='127.0.0.1', port=1883):
        self.host, self.port = host, port
        self.messages = []
        self.errors = []
        self.lock = Lock()
        self.connected = Event()
        self.subscribed = Event()
        self.client = None
        self.loop_started = False
        self.closing = False

    def start(self, timeout=8.0):
        from .mqtt_events import mqtt_client
        _, client = mqtt_client()
        self.client = client

        def on_connect(c, _u, _flags, reason, _props):
            if reason.is_failure:
                self.errors.append(f'MQTT CONNACK rejected: {reason}')
                self.connected.set()
                return
            result, _mid = c.subscribe(f'{TOPIC_ROOT}/#', qos=1)
            if result != 0:
                self.errors.append(f'MQTT subscribe returned {result}')
            self.connected.set()

        def on_subscribe(_c, _u, _mid, reason_codes, _props):
            if any(getattr(code, 'is_failure', False) for code in reason_codes):
                self.errors.append('MQTT SUBACK rejected')
            self.subscribed.set()

        def on_message(_c, _u, message):
            try:
                value = json.loads(message.payload.decode('utf-8'))
                record = {'topic': message.topic, 'event': value,
                          'retain': bool(message.retain), 'received_unix_s': time.time()}
            except (ValueError, UnicodeDecodeError) as exc:
                record = {'topic': message.topic, 'invalid_payload': str(exc),
                          'retain': bool(message.retain), 'received_unix_s': time.time()}
            with self.lock:
                self.messages.append(record)

        def on_disconnect(_c, _u, _flags, reason, _props):
            if not self.closing:
                self.errors.append(f'MQTT subscriber disconnected: {reason}')

        client.on_connect = on_connect
        client.on_subscribe = on_subscribe
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        try:
            client.connect(self.host, self.port, keepalive=30)
            client.loop_start()
            self.loop_started = True
            if not self.connected.wait(timeout) or not self.subscribed.wait(timeout):
                raise TimeoutError('Subscriber did not receive CONNACK and SUBACK')
            if self.errors:
                raise ConnectionError('; '.join(self.errors))
        except Exception:
            self.stop()
            raise
        return self

    def snapshot(self):
        with self.lock:
            return list(self.messages)

    def stop(self):
        self.closing = True
        if self.client is not None:
            try:
                self.client.disconnect()
            finally:
                if self.loop_started:
                    self.client.loop_stop()
                    self.loop_started = False


def probe_streams(rtsp_base='rtsp://127.0.0.1:8554', timeout=10):
    """Probe all four streams, rather than checking whether MediaMTX merely responds."""
    result = {}
    for camera in CAMERAS:
        url = f'{rtsp_base.rstrip("/")}/{camera}'
        cmd = ['ffprobe', '-v', 'error', '-rtsp_transport', 'tcp',
               '-i', url, '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_name,width,height', '-of', 'json']
        try:
            p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                               text=True, timeout=timeout, check=False)
            data = json.loads(p.stdout) if p.returncode == 0 else {}
            streams = data.get('streams', [])
            ok = p.returncode == 0 and len(streams) == 1 and streams[0].get('width', 0) > 0
            result[camera] = {'ready': ok, 'codec': streams[0].get('codec_name') if streams else None,
                              'error': p.stderr.strip()[-400:] if not ok else None}
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
            result[camera] = {'ready': False, 'error': str(exc)}
    return result


def live_child(*, seconds, threads, affinity, output_dir, host, port,
               rtsp_base, preview=True):
    """Apply affinity BEFORE importing torch/ultralytics. Keep production runner unchanged."""
    import psutil
    from .affinity_verify import apply_cpu_affinity
    actual = apply_cpu_affinity(psutil.Process(), affinity)
    from . import live_rtsp
    args = ['--seconds', str(seconds), '--threads', str(threads),
            '--warmup-batches', '5', '--profile-reid',
            '--output-dir', str(output_dir), '--rtsp-base', rtsp_base,
            '--mqtt', '--mqtt-host', host, '--mqtt-port', str(port)]
    if preview:
        args.append('--provisional-preview')
    print(f'Step14.18 verified inference CPU affinity: {actual}', flush=True)
    live_rtsp.main(args)


def windows_for_run(per_batch_path, seconds, interval=10.0):
    """Per-batch timing from the production CSV. Ignore incomplete final windows."""
    path = Path(per_batch_path)
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'elapsed_s' not in rows[0]:
        return []
    stamps = []
    for row in rows:
        try:
            stamp = float(row['elapsed_s'])
        except (TypeError, ValueError):
            continue
        if 0 <= stamp <= seconds + 1:
            stamps.append(stamp)
    result = []
    for n in range(int(seconds // interval)):
        lo, hi = n * interval, (n + 1) * interval
        batches = sum(lo <= stamp < hi for stamp in stamps)
        result.append({'window_start_s': lo, 'window_end_s': hi,
                       'batches': batches, 'aggregate_fps': round(4 * batches / interval, 3)})
    return result


def evaluate_run(out, received, errors, returncode):
    out = Path(out)
    report_path, events_path = out / 'report.json', out / 'events.jsonl'
    write_json(out / 'subscriber.json', {'messages': received, 'errors': errors})
    if not report_path.is_file():
        result = {'verdict': 'FAIL', 'error': 'Missing new report.json',
                  'returncode': returncode, 'subscriber_errors': errors}
        write_json(out / 'validation.json', result)
        return result
    report = load_json(report_path)
    if not events_path.is_file():
        result = {'verdict': 'FAIL', 'error': 'Missing new events.jsonl',
                  'returncode': returncode, 'subscriber_errors': errors}
        write_json(out / 'validation.json', result)
        return result
    events = load_jsonl(events_path)
    delivery = compare_delivery(events, received, report,
                                child_exit=returncode, subscriber_errors=errors)
    per_camera = report.get('per_camera', {})
    total_overwrites = sum(int(p.get('application_buffer_overwrites', 0)) for p in per_camera.values())
    interval = 10
    windows = windows_for_run(out / 'per_batch.csv', float(report.get('wall_time_s', 0)), interval)
    fps = report.get('aggregate_processed_fps')
    result = {
        'verdict': delivery['verdict'], 'returncode': returncode,
        'fps': fps, 'batch_p95_ms': report.get('processing_ms_per_four_camera_batch', {}).get('p95'),
        'buffer_overwrites': total_overwrites,
        'window_s': interval, 'windows': windows,
        'worst_10s_fps': min((r['aggregate_fps'] for r in windows), default=None),
        'windows_below_20fps': sum(r['aggregate_fps'] < 20 for r in windows),
        'mqtt_delivery': delivery,
        'provisional_preview': report.get('provisional_preview', {}),
        'limits': 'RTSP replay does not assess identity accuracy, event rule correctness, or capture-to-display latency.'
    }
    write_json(out / 'validation.json', result)
    return result


def validate_setup(*, rtsp_base, mqtt_host, mqtt_port):
    cameras = probe_streams(rtsp_base)
    mqtt = {'ready': False, 'error': None}
    try:
        sink = MQTTCapture(mqtt_host, mqtt_port).start()
        mqtt['ready'] = True
        sink.stop()
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
        mqtt['error'] = str(exc)
    return {'rtsp': cameras, 'mqtt': mqtt,
            'ready': all(d['ready'] for d in cameras.values()) and mqtt['ready']}


def run_trials(root=DEFAULT_ROOT, *, seconds=180, repeats=2, threads=4,
               affinity=AFFINITY, rtsp_base='rtsp://127.0.0.1:8554',
               mqtt_host='127.0.0.1', mqtt_port=1883, preflight=True,
               grace=2.0, runner=None, capture_factory=None):
    if seconds <= 10 or repeats < 1 or threads < 1 or grace < 0:
        raise ValueError('seconds > 10, repeats >= 1, threads >= 1, grace >= 0 required')
    root = Path(root)
    if preflight:
        check = validate_setup(rtsp_base=rtsp_base, mqtt_host=mqtt_host, mqtt_port=mqtt_port)
        write_json(root / 'preflight.json', check)
        if not check['ready']:
            raise RuntimeError(f'Preflight failed. Check {root / "preflight.json"}')
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Experiment already exists at {root}; use a new --root to avoid stale runs')
    manifest = {'configuration': {'seconds': seconds, 'repeats': repeats, 'threads': threads,
                                  'affinity': affinity, 'preview': True, 'reid': True,
                                  'mqtt': True, 'rtsp_base': rtsp_base}, 'runs': []}
    write_json(root / 'manifest.json', manifest)
    capture_factory = capture_factory or (lambda: MQTTCapture(mqtt_host, mqtt_port))
    for rep in range(1, repeats + 1):
        name = f'overlay_mqtt_{rep:02d}'
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        # Stale outputs must never pass validation.
        for filename in ('report.json', 'events.jsonl', 'per_batch.csv', 'validation.json', 'subscriber.json'):
            (out / filename).unlink(missing_ok=True)
        command = [sys.executable, '-m', 'src.end_to_end_validation', 'live-child',
                   '--seconds', str(seconds), '--threads', str(threads),
                   '--affinity', affinity, '--output-dir', str(out),
                   '--mqtt-host', mqtt_host, '--mqtt-port', str(mqtt_port),
                   '--rtsp-base', rtsp_base]
        result = {'name': name, 'complete': False, 'error': None, 'returncode': None}
        capture = None
        try:
            capture = capture_factory().start()
            if runner:
                code = runner(command, out / 'console.log')
            else:
                with (out / 'console.log').open('w', encoding='utf-8') as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=PROJECT_ROOT, check=False).returncode
            time.sleep(grace)
            received = capture.snapshot()
            errors = list(capture.errors)
            result['returncode'] = code
            validation = evaluate_run(out, received, errors, code)
            result['verdict'] = validation['verdict']
            result['complete'] = validation['verdict'] == 'PASS'
            if not result['complete']:
                result['error'] = validation.get('error') or validation.get('mqtt_delivery', {}).get('verdict')
        except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if capture:
                capture.stop()
        manifest['runs'].append(result)
        write_json(root / 'manifest.json', manifest)
        print(f'{name}: {result.get("verdict", "FAIL")} | {result.get("error") or ""}', flush=True)
    return manifest


def compare(root=DEFAULT_ROOT):
    root = Path(root)
    manifest = load_json(root / 'manifest.json')
    rows = []
    for item in manifest['runs']:
        run_dir = root / item['name']
        row = dict(item)
        if (run_dir / 'validation.json').exists():
            v = load_json(run_dir / 'validation.json')
            d = v.get('mqtt_delivery', {})
            row.update({k: v.get(k) for k in ('fps', 'batch_p95_ms', 'buffer_overwrites',
                                              'worst_10s_fps', 'windows_below_20fps')})
            row.update({'generated_events': d.get('expected_jsonl_events'),
                        'broker_acks': d.get('broker_qos1_acks'),
                        'unique_received': d.get('unique_matching_received_ids'),
                        'missing_events': d.get('missing_ids_count'),
                        'duplicate_delivery_ids': d.get('duplicate_delivery_ids_count')})
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'per_run.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)
    passes = [r for r in rows if r.get('verdict') == 'PASS']
    fps = [float(r['fps']) for r in passes if r.get('fps') is not None]
    outcome = {
        'passed_runs': len(passes), 'expected_runs': manifest['configuration']['repeats'],
        'all_runs_passed': len(passes) == manifest['configuration']['repeats'],
        'min_fps_passed_runs': min(fps) if fps else None,
        'all_runs_at_least_20fps': (len(passes) == manifest['configuration']['repeats']
                                     and bool(fps) and all(n >= 20 for n in fps)),
        'each_complete_10s_window_at_least_20fps': (len(passes) == manifest['configuration']['repeats']
                                   and all(r.get('windows_below_20fps') == 0
                                           and r.get('worst_10s_fps') is not None for r in passes)),
        'runs': rows,
        'conclusion_limits': [
            'MQTT delivery here tests only the local broker and one independently subscribed client.',
            'QoS1 does not guarantee no duplicates or durable end-to-end storage.',
            '20 aggregate FPS requires comparison with actual receipt FPS and overwrite counts.',
            'No claim about cross-camera identity accuracy or rule correctness is supported.',
            'Recorded sources may replay different pedestrian content across sequential trials.',
        ]}
    write_json(root / 'comparison.json', outcome)
    return outcome


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('preflight')
    a.add_argument('--rtsp-base', default='rtsp://127.0.0.1:8554')
    a.add_argument('--mqtt-host', default='127.0.0.1')
    a.add_argument('--mqtt-port', type=int, default=1883)
    b = sub.add_parser('run')
    b.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    b.add_argument('--seconds', type=float, default=180)
    b.add_argument('--repeats', type=int, default=2)
    b.add_argument('--threads', type=int, default=4)
    b.add_argument('--affinity', default=AFFINITY)
    b.add_argument('--rtsp-base', default='rtsp://127.0.0.1:8554')
    b.add_argument('--mqtt-host', default='127.0.0.1')
    b.add_argument('--mqtt-port', type=int, default=1883)
    b.add_argument('--grace', type=float, default=2.0)
    c = sub.add_parser('compare')
    c.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    d = sub.add_parser('live-child')
    d.add_argument('--seconds', type=float, required=True)
    d.add_argument('--threads', type=int, required=True)
    d.add_argument('--affinity', required=True)
    d.add_argument('--output-dir', type=Path, required=True)
    d.add_argument('--mqtt-host', default='127.0.0.1')
    d.add_argument('--mqtt-port', type=int, default=1883)
    d.add_argument('--rtsp-base', default='rtsp://127.0.0.1:8554')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == 'preflight':
        value = validate_setup(rtsp_base=args.rtsp_base,
                               mqtt_host=args.mqtt_host, mqtt_port=args.mqtt_port)
        print(json.dumps(value, indent=2))
        if not value['ready']:
            raise SystemExit(1)
    elif args.command == 'run':
        try:
            manifest = run_trials(root=args.root, seconds=args.seconds, repeats=args.repeats,
                                  threads=args.threads, affinity=args.affinity,
                                  rtsp_base=args.rtsp_base, mqtt_host=args.mqtt_host,
                                  mqtt_port=args.mqtt_port, grace=args.grace)
        except (RuntimeError, FileExistsError, ValueError) as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            raise SystemExit(1) from exc
        if any(not item['complete'] for item in manifest['runs']):
            raise SystemExit(1)
    elif args.command == 'compare':
        print(json.dumps(compare(args.root), indent=2))
    else:
        live_child(seconds=args.seconds, threads=args.threads, affinity=args.affinity,
                   output_dir=args.output_dir, host=args.mqtt_host,
                   port=args.mqtt_port, rtsp_base=args.rtsp_base)


if __name__ == '__main__':
    main()
