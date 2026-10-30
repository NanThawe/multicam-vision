"""Step 14.14: Windows CPU topology and repeated fixed-input affinity verification.

Commands (from project root):
  python -m src.affinity_verify topology
  python -m src.affinity_verify sweep --seconds 30 --repeats 3
  python -m src.affinity_verify compare
  python -m src.affinity_verify rtsp --affinity 0,1,2,3 --seconds 120

The fixed-input sweep depends on the Step 14.13 frozen fixture and runner.
It makes no changes to the production tracker, models, or RTSP publisher.
"""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'evaluation' / 'fixed_inference'


def parse_cpu_set_buffer(data):
    """Parse documented Win32 variable-size SYSTEM_CPU_SET_INFORMATION records."""
    records, pos = [], 0
    while pos + 8 <= len(data):
        size, kind = struct.unpack_from('<II', data, pos)
        if size < 8 or pos + size > len(data):
            raise ValueError('CPU Set response contained a malformed record size')
        if kind == 0 and size >= 24:
            cpu_set_id, group = struct.unpack_from('<IH', data, pos + 8)
            logical_id, core_index, cache_index, numa_index, efficiency, flags = struct.unpack_from(
                '<BBBBBB', data, pos + 14)
            if group == 0:
                records.append({'cpu_set_id':cpu_set_id, 'group':group,
                                'logical_cpu':logical_id, 'core_index':core_index,
                                'last_level_cache_index':cache_index,
                                'numa_index':numa_index, 'efficiency_class':efficiency,
                                'parked':bool(flags & 1), 'allocated':bool(flags & 2)})
        pos += size
    if pos != len(data):
        raise ValueError('CPU Set response had unparsed trailing data')
    unique = {x['logical_cpu'] for x in records}
    if len(unique) != len(records):
        raise ValueError('CPU Set response contained duplicate logical CPUs')
    return sorted(records, key=lambda r:r['logical_cpu'])


def windows_cpu_sets():
    """Enumerate Win32 CPU Sets. EfficiencyClass is relative, not a clock speed.

    Return only processor group 0. We do not attempt multi-group process affinity
    using psutil's group-relative logical CPU notation.
    """
    if sys.platform != 'win32':
        return [], 'Windows CPU Set API is unavailable on this operating system'
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        api = kernel32.GetSystemCpuSetInformation
        api.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                        ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_ulong]
        api.restype = ctypes.c_int
        length = ctypes.c_ulong(0)
        api(None, 0, ctypes.byref(length), None, 0)
        if not length.value or length.value > 2**20:
            return [], 'CPU Set API returned an invalid buffer size'
        buffer = ctypes.create_string_buffer(length.value)
        if not api(buffer, length.value, ctypes.byref(length), None, 0):
            return [], f'GetSystemCpuSetInformation failed: {ctypes.get_last_error()}'
        return parse_cpu_set_buffer(buffer.raw[:length.value]), None
    except (AttributeError, OSError, ValueError, struct.error) as exc:
        return [], f'CPU Set API unavailable: {exc}'


def select_distinct_cores(cpu_sets, count=4, *, highest=True):
    """Prefer one logical processor per physical core before adding SMT siblings."""
    if count < 1:
        raise ValueError('count must be >= 1')
    values = sorted(cpu_sets, key=lambda r:
                    ((-r['efficiency_class'] if highest else r['efficiency_class']),
                     r['core_index'], r['logical_cpu']))
    selected, seen_cores = [], set()
    for rec in values:
        key = (rec['group'], rec['core_index'])
        if key in seen_cores:
            continue
        selected.append(rec['logical_cpu'])
        seen_cores.add(key)
        if len(selected) == count:
            return sorted(selected)
    for rec in values:
        if rec['logical_cpu'] not in selected:
            selected.append(rec['logical_cpu'])
            if len(selected) == count:
                return sorted(selected)
    return sorted(selected)


def candidate_groups(allowed, cpu_sets):
    allowed = sorted(set(int(x) for x in allowed))
    if not allowed:
        raise ValueError('No available CPUs for this process')
    groups = {'all':allowed, 'first4':allowed[:4], 'last4':allowed[-4:]}
    eligible = [r for r in cpu_sets if r['logical_cpu'] in allowed and not r['allocated']]
    observed = {r['efficiency_class'] for r in eligible}
    # Topology-guided experiments only when the Windows API actually distinguishes classes.
    if len(observed) > 1:
        groups['higher_efficiency_class4'] = select_distinct_cores(eligible, 4, highest=True)
        groups['lower_efficiency_class4'] = select_distinct_cores(eligible, 4, highest=False)
    # Avoid running identical affinity masks under multiple names.
    dedup, seen = {}, set()
    for name, ids in groups.items():
        key = tuple(ids)
        if key not in seen:
            dedup[name] = ids
            seen.add(key)
    return dedup


def topology(root=DEFAULT_ROOT):
    import psutil
    from . import fixed_inference as fi
    proc = psutil.Process()
    allowed = sorted(proc.cpu_affinity())
    cpu_sets, warning = windows_cpu_sets()
    groups = candidate_groups(allowed, cpu_sets)
    result = {'platform':sys.platform, 'logical_cpus':psutil.cpu_count(logical=True),
              'physical_cpus_psutil':psutil.cpu_count(logical=False),
              'allowed_affinity':allowed, 'cpu_sets':cpu_sets,
              'cpu_set_warning':warning, 'candidate_groups':groups,
              'interpretation':('Efficiency classes are comparable only within this machine. '
                   'Higher values indicate relatively more performant and less energy-efficient CPUs. '
                   'Classes may not distinguish cores on every processor. '
                   'The API groups logical siblings by core_index. '
                   'This does not establish clocks, thermals or Windows scheduling decisions.'),
              'constraint':'Only processor group 0 is supported by this affinity experiment.'}
    fi.save_json(Path(root)/'cpu_topology.json', result)
    return result


def trial_order(group_names, repeats):
    """Alternate/reverse case order to reduce systematic temperature/order bias."""
    if repeats < 2:
        raise ValueError('Use at least two repetitions for an affinity comparison')
    if not group_names:
        raise ValueError('No groups')
    orders = []
    names = list(group_names)
    for iteration in range(repeats):
        order = names[iteration % len(names):] + names[:iteration % len(names)]
        if iteration % 2:
            order.reverse()
        orders.append(order)
    return orders


def run_sweep(root=DEFAULT_ROOT, *, seconds=30., repeats=3, warmup=8, threads=4,
              executor=None):
    from . import fixed_inference as fi
    root = Path(root)
    if not (root/'fixture.json').is_file():
        raise FileNotFoundError('Missing Step 14.13 fixture. Run python -m src.fixed_inference prepare')
    if seconds <= 0 or warmup < 0 or threads < 1:
        raise ValueError('Invalid seconds, warmup or threads')
    info = topology(root)
    if info['logical_cpus'] and info['logical_cpus'] > 64:
        raise RuntimeError('Multiple Windows processor groups are not supported by this patch')
    groups = info['candidate_groups']
    run_plan = trial_order(groups, repeats)
    execute = executor or fi.execute_trial
    results = []
    fixture = fi.load_json(root/'fixture.json')
    manifest_path = root/'affinity_sweep_manifest.json'
    for rep, names in enumerate(run_plan, 1):
        for name in names:
            ids = groups[name]
            trial_name = f'round{rep:02d}_{name}'
            out = root/'affinity_runs'/trial_name
            out.mkdir(parents=True, exist_ok=True)
            rec = {'trial':trial_name, 'repetition':rep,'group':name,'cpu_ids':ids,
                   'complete':False, 'returncode':None, 'error':None}
            cmd = [sys.executable, '-m', 'src.fixed_inference', 'run', '--root',str(root),
                   '--out',str(out),'--seconds',str(seconds),'--warmup',str(warmup),
                   '--threads',str(threads),'--blas-threads',str(threads),
                   '--affinity',','.join(str(x) for x in ids),'--workload','both']
            try:
                # Do not mistake an old summary for a successfully completed trial.
                (out/'summary.json').unlink(missing_ok=True)
                rec['returncode'] = execute(cmd, out/'console.log')
                rec['complete'] = (rec['returncode']==0 and (out/'summary.json').is_file())
                if rec['complete']:
                    summary = fi.load_json(out/'summary.json')
                    if summary.get('fixture_hashes') != {
                        'frames':fixture['frames_sha256'], 'boxes':fixture['boxes_sha256']
                    }:
                        raise RuntimeError('Fixture hashes disagree with frozen input')
                    measured = summary.get('cpu',{}).get('affinity_actual')
                    if sorted(measured or []) != ids:
                        raise RuntimeError(f'Affinity not verified: expected {ids}, measured {measured}')
                if not rec['complete']:
                    rec['error'] = f'Benchmark exited {rec["returncode"]} or summary absent'
            except Exception as exc:
                rec['complete'] = False
                rec['error'] = f'{type(exc).__name__}: {exc}'
            results.append(rec)
            fi.save_json(manifest_path, {'fixture_hashes':{
                'frames':fixture['frames_sha256'], 'boxes':fixture['boxes_sha256']},
                'threads':threads,'seconds':seconds,'warmup':warmup,
                'repeats':repeats,'groups':groups,'schedule':run_plan,'runs':results})
            print(f'{trial_name}: {"PASS" if rec["complete"] else "FAIL"} {rec["error"] or ""}',flush=True)
    return results


def summarize_trials(root=DEFAULT_ROOT):
    from . import fixed_inference as fi
    root = Path(root)
    manifest = fi.load_json(root/'affinity_sweep_manifest.json')
    fixture_hashes = manifest['fixture_hashes']
    expected_repeats = int(manifest['repeats'])
    observed = []
    for run in manifest['runs']:
        row = {k:run.get(k) for k in ('trial','repetition','group','cpu_ids','complete','error')}
        if run['complete']:
            report = fi.load_json(root/'affinity_runs'/run['trial']/'summary.json')
            if report.get('fixture_hashes') != fixture_hashes:
                raise RuntimeError(f'Fixture mismatch in {run["trial"]}')
            lat = report['latency_ms']
            row.update({'fps':report['aggregate_equivalent_fps'],
                        'iterations':report['iterations'],
                        'combined_p50_ms':lat['combined']['p50'],
                        'combined_p95_ms':lat['combined']['p95'],
                        'combined_p99_ms':lat['combined']['p99'],
                        'yolo_p50_ms':lat['yolo']['p50'],
                        'osnet_p50_ms':lat['osnet']['p50'],
                        'process_cpu_pct_one_core':report['cpu']['process_cpu_pct_one_core']})
        observed.append(row)
    summary = []
    for name, cpus in manifest['groups'].items():
        successful = [r for r in observed if r['group']==name and r['complete']]
        fps = [r['fps'] for r in successful]
        p95 = [r['combined_p95_ms'] for r in successful]
        p99 = [r['combined_p99_ms'] for r in successful]
        summary.append({'group':name,'cpu_ids':cpus,
                        'successful_runs':len(successful),'expected_runs':expected_repeats,
                        'median_equivalent_fps':round(statistics.median(fps),3) if fps else None,
                        'min_equivalent_fps':round(min(fps),3) if fps else None,
                        'max_equivalent_fps':round(max(fps),3) if fps else None,
                        'median_combined_p95_ms':round(statistics.median(p95),3) if p95 else None,
                        'worst_combined_p95_ms':round(max(p95),3) if p95 else None,
                        'median_combined_p99_ms':round(statistics.median(p99),3) if p99 else None,
                        'relative_fps_range_pct':(round((max(fps)-min(fps))/statistics.median(fps)*100,2)
                                                  if fps and statistics.median(fps)>0 else None)})
    eligible = [x for x in summary if x['successful_runs']==expected_repeats]
    # Prefer reliable worst-run throughput; break ties by worst-run P95.
    candidate = (sorted(eligible, key=lambda x:(-x['min_equivalent_fps'],
                                      x['worst_combined_p95_ms']))[0]['group'] if eligible else None)
    result = {'basis':'Identical Step 14.13 fixture, independent fresh processes, repeated alternating order',
              'fixture_hashes':fixture_hashes,'threads':manifest['threads'],
              'repeats':expected_repeats,'groups':summary,
              'candidate_for_rtsp_validation':candidate,
              'candidate_cpu_ids':manifest['groups'].get(candidate) if candidate else None,
              'candidate_selection':'Highest worst-run fixed-input FPS among complete groups; tie-break lowest worst-run P95. Not automatic production selection.',
              'warnings':['Fixed-input equivalent FPS is NOT RTSP FPS.',
                          'Changing affinity also changes decoder/other thread placement during live RTSP.',
                          'Do not infer processor core class from numeric logical CPU IDs.',
                          'Repeat under stable AC power and host load before deployment.',
                          'If any repeats failed, those groups cannot be selected.']}
    fi.save_json(root/'affinity_comparison.json',result)
    columns = ['trial','repetition','group','cpu_ids','complete','error','fps','iterations',
               'combined_p50_ms','combined_p95_ms','combined_p99_ms','yolo_p50_ms',
               'osnet_p50_ms','process_cpu_pct_one_core']
    with (root/'affinity_per_run.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in observed:
            row=dict(row)
            row['cpu_ids']=','.join(map(str,row.get('cpu_ids') or []))
            writer.writerow(row)
    return result


def apply_cpu_affinity(proc, requested):
    """Verify process affinity before model or RTSP imports execute."""
    available = sorted(proc.cpu_affinity())
    if requested == 'all':
        target = available
    else:
        try:
            target = sorted(set(int(x.strip()) for x in requested.split(',')))
        except ValueError as exc:
            raise ValueError(f'Invalid affinity IDs: {requested}') from exc
        if not target or any(cpu not in available for cpu in target):
            raise ValueError(f'Affinity {target} is not within allowed CPUs {available}')
    if target != available:
        proc.cpu_affinity(target)
    actual = sorted(proc.cpu_affinity())
    if actual != target:
        raise RuntimeError(f'Affinity not applied: expected {target}, got {actual}')
    return actual


def run_rtsp(*, affinity, seconds=120., threads=4, output_dir=None,
             provisional_preview=False, profile_reid=False, no_reid=False):
    """Apply affinity before importing model runtimes, then run existing RTSP CLI.

    This affects only the child Python process. Four publisher processes are independent.
    """
    import psutil
    proc=psutil.Process()
    applied=apply_cpu_affinity(proc,affinity)
    # Import after affinity, so model runtimes initialise in the chosen group.
    from . import live_rtsp
    args=['--seconds',str(seconds),'--threads',str(threads),
          '--warmup-batches','5']
    if output_dir is not None:
        args += ['--output-dir',str(output_dir)]
    if provisional_preview:
        args.append('--provisional-preview')
    if profile_reid:
        args.append('--profile-reid')
    if no_reid:
        args.append('--no-reid')
    print(f'RTSP Python PID {os.getpid()}, verified affinity {applied}',flush=True)
    live_rtsp.main(args)


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('topology')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p=sub.add_parser('sweep')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p=sub.add_parser('compare')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p=sub.add_parser('rtsp')
    p.add_argument('--affinity',required=True,help='all or comma-separated verified logical CPU IDs')
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--provisional-preview',action='store_true')
    p.add_argument('--profile-reid',action='store_true')
    p.add_argument('--no-reid',action='store_true')
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        if args.command=='topology':
            result=topology(args.root)
        elif args.command=='sweep':
            result=run_sweep(args.root,seconds=args.seconds,repeats=args.repeats,
                             warmup=args.warmup,threads=args.threads)
            if any(not x['complete'] for x in result):
                raise RuntimeError('At least one trial failed: inspect affinity_sweep_manifest.json')
        elif args.command=='compare':
            result=summarize_trials(args.root)
        else:
            result=run_rtsp(affinity=args.affinity,seconds=args.seconds,threads=args.threads,
                            output_dir=args.output_dir,provisional_preview=args.provisional_preview,
                            profile_reid=args.profile_reid,no_reid=args.no_reid)
        if args.command in ('topology','compare'):
            print(json.dumps(result,indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2) from exc

if __name__=='__main__':
    main()
