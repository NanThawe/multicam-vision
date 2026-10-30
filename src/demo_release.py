"""Step 14.29 — package a *recorded* Step14.28 four-camera demo.

Builds a localhost dashboard from existing, verified RTSP-run artifacts. No
inference, RTSP decoding, OSNet matching, or event generation occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs' / 'rtsp' / 'step14_28'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'demo' / 'step14_29'
ASSET_HTML = Path(__file__).resolve().parent / 'demo_assets' / 'index.html'
CAMERAS = ('C1', 'C2', 'C3', 'C4')
EVENT_TYPES = ('zone_entry', 'zone_exit', 'line_crossing')
TRIAL_PATTERN = re.compile(r'^overlay_mqtt_\d{2}$')


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    data = []
    with Path(path).open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Malformed JSONL at {path} line {n}') from exc
    return data


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_trial(source, trial):
    """Do not silently choose a different successful run or accept stale evidence."""
    if not TRIAL_PATTERN.fullmatch(trial):
        raise ValueError('trial must look like overlay_mqtt_01')
    source = Path(source)
    manifest = read_json(source / 'manifest.json')
    known = {item['name']: item for item in manifest.get('runs', [])}
    if trial not in known:
        raise ValueError(f'Trial {trial} is not present in {source / "manifest.json"}')
    folder = source / trial
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    report = read_json(folder / 'report.json')
    delivery = read_json(folder / 'validation.json')
    if report.get('complete') is not True or report.get('run_status') != 'completed':
        raise ValueError(f'{trial}: source run not completed')
    if delivery.get('verdict') != 'PASS' or delivery.get('mqtt_delivery', {}).get('verdict') != 'PASS':
        raise ValueError(f'{trial}: independent MQTT delivery was not verified PASS')
    if known[trial].get('returncode') not in (0,):
        raise ValueError(f'{trial}: nonzero/missing child process return code')
    if not report.get('provisional_preview', {}).get('enabled'):
        raise ValueError('Source run must contain the four-camera provisional_preview.mp4')
    if report.get('cameras') != list(CAMERAS):
        raise ValueError('Source camera order does not match C1–C4')
    for p in ('per_batch.csv', 'events.jsonl', 'provisional_preview.mp4'):
        if not (folder / p).is_file():
            raise FileNotFoundError(folder / p)
    return folder, report, delivery


def load_batch_lookup(path, report):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != report['measured_batches']:
        raise ValueError('per_batch.csv and report.measured_batches differ')
    mapping = {}
    for i, row in enumerate(rows):
        frame = int(row['frame_id'])
        if frame in mapping:
            raise ValueError(f'Duplicate frame ID {frame} in per_batch.csv')
        mapping[frame] = i
    return mapping


def serialize_events(events, batch_index, warmup, preview_fps):
    seen = set()
    rendered = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('An event is not a JSON object')
        eid = event.get('event_id')
        cam = event.get('camera')
        typ = event.get('event_type')
        if not isinstance(eid, str) or not eid or eid in seen:
            raise ValueError(f'Invalid/duplicate event ID {eid!r}')
        if cam not in CAMERAS or typ not in EVENT_TYPES:
            raise ValueError(f'Invalid event camera/type {cam}/{typ}')
        if type(event.get('track_id')) is not int:
            raise ValueError(f'{eid}: invalid local track ID')
        frame = int(event['frame_id'])
        if frame not in batch_index:
            raise ValueError(f'{eid}: frame {frame} absent from measured per_batch rows')
        seen.add(eid)
        out = {k: event.get(k) for k in ('event_id', 'camera', 'event_type', 'rule_id',
               'track_id', 'global_id', 'frame_id', 'elapsed_s', 'direction') if k in event}
        # Frames written to video: warmup frames followed by measured frame batches.
        # The output preview is constant-FPS playback, not actual real-time latency.
        out['video_offset_s'] = round((warmup + batch_index[frame]) / preview_fps, 3)
        rendered.append(out)
    return rendered


def ffmpeg_convert(source, destination, ffmpeg='ffmpeg'):
    """Browser-compatible H.264; no synthetic video or replacement frames."""
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-i', str(source), '-map', '0:v:0', '-an', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(destination)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError('FFmpeg was not found. Install FFmpeg with libx264 on PATH.') from exc
    if result.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f'FFmpeg H.264 conversion failed: {result.stderr[-1200:]}')


def rules_for_dashboard():
    from .events import RULES
    rules = {}
    for cam in CAMERAS:
        value = RULES[cam]
        zone = [float(n) for n in value['zone']]
        line = [[float(x), float(y)] for x, y in value['line']]
        if not (len(zone) == 4 and 0 <= zone[0] < zone[2] <= 1
                and 0 <= zone[1] < zone[3] <= 1
                and len(line) == 2 and all(len(pt) == 2 and all(0 <= v <= 1 for v in pt) for pt in line)):
            raise ValueError(f'Invalid illustration rules for {cam}')
        rules[cam] = {'zone': zone, 'line': line}
    return rules


def make_data(report, delivery, events, trial, source, reliability=None, trial_metrics=None):
    counts = {cam: {typ: 0 for typ in EVENT_TYPES} for cam in CAMERAS}
    for event in events:
        counts[event['camera']][event['event_type']] += 1
    overwrites = sum(int(row.get('application_buffer_overwrites', 0))
                     for row in report['per_camera'].values())
    mqtt = delivery['mqtt_delivery']
    if len(events) != report.get('events_generated') or len(events) != mqtt.get('expected_jsonl_events'):
        raise ValueError('Event counts do not match report and MQTT delivery validation')
    if mqtt.get('unique_matching_received_ids') != len(events):
        raise ValueError('Independent MQTT receipts do not match generated event count')
    result = {
        'title': 'Four-Camera Analytics | Recorded Demo',
        'trial': trial,
        'mode': 'Recorded RTSP replay, not live monitoring',
        'video': 'preview_h264.mp4',
        'metrics': {
            'aggregate_fps': report.get('aggregate_processed_fps'),
            'fps_target': 20.0,
            'processed_per_camera_fps': report.get('processed_fps_per_camera'),
            'batch_p95_ms': report.get('processing_ms_per_four_camera_batch', {}).get('p95'),
            'buffer_overwrites': overwrites,
            'events_generated': len(events),
            'mqtt_acks': mqtt.get('broker_qos1_acks'),
            'mqtt_unique_received': mqtt.get('unique_matching_received_ids'),
            'mqtt_duplicate_ids': mqtt.get('duplicate_delivery_ids_count'),
            'mqtt_status': delivery.get('verdict'),
            'runner_complete': report.get('complete'),
            'measured_seconds': report.get('wall_time_s'),
            'target_met': bool(trial_metrics and trial_metrics.get('performance_target_met') is True),
        },
        'per_camera': report['per_camera'],
        'event_counts': counts,
        'rules': rules_for_dashboard(),
        'events': events,
        'limitations': [
            'Four tiled views replay previously recorded RTSP input; publisher streams are not synchronized.',
            'Preview is rendered at constant 5 FPS; video position is not capture-to-display latency.',
            'Zone and counting-line boundaries are illustrative unless separately calibrated.',
            'Green L-labels are confirmed local tracks. Orange P boxes are provisional and do not produce events.',
            'G-labels are experimental appearance associations, NOT verified same-person identities.',
            'Event presence and MQTT delivery do not validate event semantics or identity accuracy.',
            'Broker acknowledgements and one independent subscriber test do not imply exactly-once MQTT.',
            'Frame overwrites describe the application LatestSlot, not all network/source frame losses.',
        ],
    }
    if reliability is not None:
        result['repeated_validation'] = {
            k: reliability.get(k) for k in (
                'trial_count','fps_min','fps_median','fps_max','mqtt_status','strict_realtime_target_met')
        }
    return result


def build(source=DEFAULT_SOURCE, trial='overlay_mqtt_02', output=DEFAULT_OUTPUT,
          *, ffmpeg='ffmpeg', converter=None):
    source, output = Path(source), Path(output)
    run, report, delivery = verified_trial(source, trial)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Demo output is not empty: {output}. Use a new --output directory.')
    batches = load_batch_lookup(run / 'per_batch.csv', report)
    fps = float(report['provisional_preview'].get('video_playback_fps', 5))
    if fps <= 0:
        raise ValueError('Invalid preview playback FPS')
    warmup = int(report.get('warmup_batches', 0))
    preview_count = report['provisional_preview'].get('rendered_batches_including_warmup')
    if preview_count != len(batches) + warmup:
        raise ValueError('Preview frame count does not agree with warmup + measured batches')
    events = serialize_events(read_jsonl(run / 'events.jsonl'), batches, warmup, fps)
    reliability_path = source / 'reliability_comparison.json'
    reliability = read_json(reliability_path) if reliability_path.exists() else None
    trial_metrics_path = run / 'reliability_metrics.json'
    trial_metrics = read_json(trial_metrics_path) if trial_metrics_path.exists() else None
    data = make_data(report, delivery, events, trial, source, reliability, trial_metrics)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if not ASSET_HTML.is_file():
            raise FileNotFoundError(f'Missing dashboard HTML asset: {ASSET_HTML}')
        shutil.copy2(ASSET_HTML, output / 'index.html')
        (converter or ffmpeg_convert)(run / 'provisional_preview.mp4',
                                     output / 'preview_h264.mp4', ffmpeg)
        if not (output / 'preview_h264.mp4').exists():
            raise FileNotFoundError('Video converter did not create preview_h264.mp4')
        write_json(output / 'data.json', data)
        source_hashes = {name: sha(run / name) for name in (
            'report.json','validation.json','per_batch.csv','events.jsonl','provisional_preview.mp4')}
        output_hashes = {name: sha(output / name) for name in (
            'index.html','data.json','preview_h264.mp4')}
        write_json(output / 'evidence_manifest.json', {
            'experiment': 'Step 14.29 recorded demo package (no new model evaluation)',
            'selected_trial': trial,
            'source_artifact_hashes_sha256': source_hashes,
            'output_hashes_sha256': output_hashes,
            'matched_event_count': len(events),
            'production_logic_changed': False,
            'browser_video_codec': 'H.264 yuv420p via FFmpeg',
            'limits': data['limitations'],
        })
    except Exception:
        # Don't leave behind a partly built folder that looks like a complete demo.
        for name in ('index.html','preview_h264.mp4','data.json','evidence_manifest.json'):
            (output / name).unlink(missing_ok=True)
        raise
    return {'output': str(output), 'trial': trial, 'events': len(events),
            'fps': report['aggregate_processed_fps'],
            'mqtt': delivery['verdict'], 'video': str(output / 'preview_h264.mp4')}


def serve(output=DEFAULT_OUTPUT, port=8765):
    output = Path(output).resolve()
    if not all((output / name).is_file() for name in ('index.html','data.json','preview_h264.mp4','evidence_manifest.json')):
        raise FileNotFoundError(f'Demo incomplete under {output}; run build first')
    # Browser video/images and raw MQTT event history are hosted on LOOPBACK ONLY.
    handler = partial(SimpleHTTPRequestHandler, directory=str(output))
    with ThreadingHTTPServer(('127.0.0.1', port), handler) as server:
        print(f'Demo: http://127.0.0.1:{server.server_port}/ (Ctrl+C to stop)', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    build_cmd = sub.add_parser('build', help='Assemble a recorded four-camera demo from verified Step14.28 trial')
    build_cmd.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    build_cmd.add_argument('--trial', default='overlay_mqtt_02')
    build_cmd.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    build_cmd.add_argument('--ffmpeg', default='ffmpeg')
    serve_cmd = sub.add_parser('serve', help='Serve local dashboard on 127.0.0.1 only')
    serve_cmd.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    serve_cmd.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if args.cmd == 'build':
        print(json.dumps(build(args.source, args.trial, args.output, ffmpeg=args.ffmpeg), indent=2))
    else:
        serve(args.output, args.port)


if __name__ == '__main__':
    main()
