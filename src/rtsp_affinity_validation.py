"""Step 14.15: Repeat matched RTSP trials for the Step 14.14 CPU candidate.

Run from project root with MediaMTX and all four publishers already running:
  python -m src.rtsp_affinity_validation run --seconds 120 --repeats 2
  python -m src.rtsp_affinity_validation compare

No production source is modified. Reuses `src.affinity_verify rtsp`.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = PROJECT_ROOT / 'outputs' / 'evaluation' / 'fixed_inference' / 'affinity_comparison.json'
DEFAULT_TOPOLOGY = PROJECT_ROOT / 'outputs' / 'evaluation' / 'fixed_inference' / 'cpu_topology.json'
DEFAULT_ROOT = PROJECT_ROOT / 'outputs' / 'rtsp' / 'step14_15'
CAMERAS = 4


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def validated_groups(comparison, topology):
    candidate = comparison.get('candidate_cpu_ids')
    allowed = set(topology.get('allowed_affinity', []))
    first4 = topology.get('candidate_groups', {}).get('first4')
    if not candidate or len(candidate) != len(set(candidate)) or not set(candidate) <= allowed:
        raise ValueError('Missing or invalid verified Step 14.14 candidate CPU IDs')
    if not first4 or not set(first4) <= allowed:
        raise ValueError('Missing first4 reference group in CPU topology')
    groups = {'all': 'all', 'candidate': ','.join(map(str, candidate)),
              'first4': ','.join(map(str, first4))}
    # If a topology produces the same 4-core group as first4, skip duplicate.
    if groups['candidate'] == groups['first4']:
        del groups['first4']
    return groups


def sequence(groups, repeats):
    if repeats < 2:
        raise ValueError('At least two repetitions are required to balance run order')
    names = list(groups)
    order = []
    for repeat in range(1, repeats + 1):
        scheduled = names if repeat % 2 else list(reversed(names))
        order.extend([{'name': name, 'repetition': repeat,
                       'run': f'{name}_{repeat:02d}'} for name in scheduled])
    return order


def invoke(cmd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT,
                              check=False)
    return proc.returncode


def run(root=DEFAULT_ROOT, comparison=DEFAULT_COMPARISON, topology=DEFAULT_TOPOLOGY,
        seconds=120, repeats=2, threads=4, runner=None):
    if seconds <= 10 or threads < 1:
        raise ValueError('Use seconds > 10 and threads >= 1')
    root = Path(root)
    groups = validated_groups(read_json(comparison), read_json(topology))
    plan = sequence(groups, repeats)
    execute = runner or invoke
    records = []
    manifest = {'configuration': {'seconds': seconds, 'repeats': repeats, 'threads': threads,
                                  'groups': groups, 'reid_enabled': True,
                                  'preview_enabled': False, 'published_streams': CAMERAS},
                'schedule': plan, 'runs': records}
    write_json(root / 'manifest.json', manifest)
    for item in plan:
        name = item['run']
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        # Avoid mistaking results from an earlier run for a successful trial.
        (out / 'report.json').unlink(missing_ok=True)
        (out / 'per_batch.csv').unlink(missing_ok=True)
        cmd = [sys.executable, '-m', 'src.affinity_verify', 'rtsp',
               '--affinity', groups[item['name']], '--seconds', str(seconds),
               '--threads', str(threads), '--profile-reid', '--output-dir', str(out)]
        record = dict(item, affinity=groups[item['name']], returncode=None,
                      complete=False, error=None)
        try:
            record['returncode'] = execute(cmd, out / 'console.log')
            if record['returncode'] != 0:
                record['error'] = f'Process exit code {record["returncode"]}'
            elif not (out / 'report.json').exists() or not (out / 'per_batch.csv').exists():
                record['error'] = 'Required new report.json or per_batch.csv missing'
            else:
                report = read_json(out / 'report.json')
                record['complete'] = bool(report.get('complete') and
                                          report.get('run_status') == 'completed' and
                                          report.get('cameras') == ['C1', 'C2', 'C3', 'C4'])
                if not record['complete']:
                    record['error'] = 'Reported run incomplete or unexpected camera set'
        except (OSError, ValueError, TypeError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        records.append(record)
        write_json(root / 'manifest.json', manifest)
        print(f'{name}: {"PASS" if record["complete"] else "FAIL"} {record["error"] or ""}',
              flush=True)
    return records


def _numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def time_windows(csv_path, *, wall_s, window_s=10.0):
    """Count completed batches by absolute elapsed time; omit partial final windows."""
    if wall_s <= 0 or window_s <= 0:
        return []
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'elapsed_s' not in rows[0]:
        return []
    elapsed = [_numeric(r.get('elapsed_s')) for r in rows]
    elapsed = [v for v in elapsed if v is not None and 0 <= v <= wall_s + 1]
    result = []
    for i in range(int(wall_s // window_s)):
        start, end = i * window_s, (i + 1) * window_s
        count = sum(start <= t < end for t in elapsed)
        result.append({'start_s': start, 'end_s': end, 'batches': count,
                       'aggregate_fps': round(CAMERAS * count / window_s, 4)})
    return result


def aggregate(root=DEFAULT_ROOT, *, min_fps=20.0, window_s=10.0):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    rows = []
    windows = []
    for record in manifest['runs']:
        row = {key: record.get(key) for key in ['run', 'name', 'repetition', 'affinity',
                                               'complete', 'error', 'returncode']}
        if record['complete']:
            out = root / record['run']
            report = read_json(out / 'report.json')
            if not report.get('complete') or report.get('run_status') != 'completed':
                row['complete'] = False
                row['error'] = 'Report no longer confirms completion'
            else:
                rate = float(report['aggregate_processed_fps'])
                batch = report.get('processing_ms_per_four_camera_batch', {})
                stage = report.get('stage_timings_ms', {})
                overwrites = sum(int(v.get('application_buffer_overwrites', 0))
                                 for v in report.get('per_camera', {}).values())
                w = time_windows(out / 'per_batch.csv',
                                 wall_s=float(report.get('wall_time_s', 0)), window_s=window_s)
                windows.extend([{**x, 'run': record['run'], 'name': record['name']}
                                for x in w])
                row.update({'aggregate_fps': rate,
                            'per_camera_fps': float(report.get('processed_fps_per_camera', 0)),
                            'processing_p50_ms': _numeric(batch.get('p50')),
                            'processing_p95_ms': _numeric(batch.get('p95')),
                            'detector_p50_ms': _numeric(stage.get('detector', {}).get('p50')),
                            'reid_p50_ms': _numeric(stage.get('reid', {}).get('p50')),
                            'buffer_overwrites': overwrites,
                            'worst_window_fps': min((x['aggregate_fps'] for x in w), default=None),
                            'windows_below_target': sum(x['aggregate_fps'] < min_fps for x in w),
                            'window_count': len(w),
                            'meets_run_target': rate >= min_fps and
                                                 bool(w) and all(x['aggregate_fps'] >= min_fps for x in w)})
        rows.append(row)
    group_summary = []
    for name in manifest['configuration']['groups']:
        valid = [row for row in rows if row['name'] == name and row['complete']]
        fps = [r['aggregate_fps'] for r in valid]
        p95 = [r['processing_p95_ms'] for r in valid if r.get('processing_p95_ms') is not None]
        group_summary.append({'name': name, 'completed_runs': len(valid),
                              'expected_runs': manifest['configuration']['repeats'],
                              'median_aggregate_fps': round(statistics.median(fps), 3) if fps else None,
                              'worst_run_aggregate_fps': min(fps) if fps else None,
                              'median_processing_p95_ms': round(statistics.median(p95), 3) if p95 else None,
                              'runs_meeting_target': sum(r.get('meets_run_target', False) for r in valid),
                              'total_buffer_overwrites': sum(r.get('buffer_overwrites', 0) for r in valid)})
    summary = {'target_aggregate_fps': min_fps, 'window_s': window_s,
               'configuration': manifest['configuration'], 'groups': group_summary,
               'caveats': ['Different runs may see different pedestrians and RTSP timestamps.',
                           'Affinity also restricts Python decoder and other native threads.',
                           'A high average FPS does not establish every 10-second interval is stable.',
                           'Actual sender FFmpeg processes remain independent of Python affinity.',
                           'No identity accuracy or official benchmark metrics are measured.']}
    write_json(root / 'comparison.json', summary)
    for path, data in [(root / 'per_run.csv', rows), (root / 'time_windows.csv', windows)]:
        columns = list(dict.fromkeys(k for row in data for k in row))
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if columns:
                writer.writeheader()
                writer.writerows(data)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_subparsers(dest='command', required=True)
    a = p.add_parser('run')
    a.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    a.add_argument('--comparison', type=Path, default=DEFAULT_COMPARISON)
    a.add_argument('--topology', type=Path, default=DEFAULT_TOPOLOGY)
    a.add_argument('--seconds', type=float, default=120)
    a.add_argument('--repeats', type=int, default=2)
    a.add_argument('--threads', type=int, default=4)
    b = p.add_parser('compare')
    b.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    b.add_argument('--min-fps', type=float, default=20)
    b.add_argument('--window-s', type=float, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'run':
        results = run(root=args.root, comparison=args.comparison, topology=args.topology,
                      seconds=args.seconds, repeats=args.repeats, threads=args.threads)
        if any(not r['complete'] for r in results):
            print('Some RTSP trials failed. Inspect manifest.json and console.log', file=sys.stderr)
            raise SystemExit(1)
    else:
        print(json.dumps(aggregate(args.root, min_fps=args.min_fps,
                                   window_s=args.window_s), indent=2))


if __name__ == '__main__':
    main()
