"""Step 14.13: frozen-input CPU benchmark for YOLO + OSNet.

Run from the multicam-analytics project root:
  python -m src.fixed_inference prepare
  python -m src.fixed_inference sweep --seconds 30
  python -m src.fixed_inference compare

This is a CPU execution diagnostic, not a tracking-accuracy or live FPS benchmark.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .framesource import ROOT, VIDEOS

DEFAULT_ROOT = ROOT / 'outputs' / 'evaluation' / 'fixed_inference'
CAMERAS = ('C1', 'C2', 'C3', 'C4')
MEASURE_FIELDS = ('iteration', 'elapsed_s', 'yolo_ms', 'osnet_ms', 'combined_ms',
                  'process_cpu_pct_one_core', 'system_cpu_pct', 'rss_mb')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def percentile(numbers, fraction):
    values = sorted(float(x) for x in numbers)
    if not values:
        return None
    pos = (len(values) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 3)


def latency(numbers):
    return {'p50': percentile(numbers, .50), 'p95': percentile(numbers, .95),
            'p99': percentile(numbers, .99)}


def correlation(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    covariance = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    sd = (sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys))**.5
    return round(covariance / sd, 4) if sd else None


def resize_width(image, width):
    import cv2
    if width <= 0:
        raise ValueError('width must be positive')
    h, w = image.shape[:2]
    target_h = round(h * width / w)
    return cv2.resize(image, (width, target_h), interpolation=cv2.INTER_AREA)


def load_source_images(source='annotated', frame_id=0, width=960):
    import cv2
    result, provenance = {}, {}
    if source not in ('annotated', 'video'):
        raise ValueError('source must be annotated or video')
    for camera in CAMERAS:
        if source == 'annotated':
            path = ROOT / 'data' / 'wildtrack_eval' / 'Image_subsets' / camera / f'{frame_id:08d}.png'
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f'Missing annotated image: {path}. Run Step 14 download, or use --source video.')
        else:
            path = VIDEOS[camera]
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise FileNotFoundError(f'Cannot open {path}')
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot read frame {frame_id} from {path}')
            finally:
                cap.release()
        result[camera] = resize_width(image, width)
        provenance[camera] = {'source_path': str(path), 'frame_id': frame_id}
    return result, provenance


def select_boxes(detections, frames, max_per_camera=4):
    """Use the SAME 16-ish person crops in every encoder iteration."""
    if max_per_camera < 1:
        raise ValueError('max_per_camera must be >= 1')
    boxes = {}
    for camera in CAMERAS:
        height, width = frames[camera].shape[:2]
        available = []
        for person in detections[camera]:
            x1,y1,x2,y2 = (float(v) for v in person['xyxy'])
            coords = [max(0.,min(x1,width)), max(0.,min(y1,height)),
                      max(0.,min(x2,width)), max(0.,min(y2,height))]
            if coords[2]-coords[0] < 16 or coords[3]-coords[1] < 32:
                continue
            available.append({'xyxy': coords,
                              'confidence': float(person['confidence']),
                              'class_id': 0})
        available.sort(key=lambda p: (-p['confidence'], p['xyxy'][0], p['xyxy'][1]))
        boxes[camera] = available[:max_per_camera]
    if not any(boxes.values()):
        raise ValueError('YOLO returned no valid pedestrian crops. Try --frame-id 20 or --source video.')
    return boxes


def prepare(root=DEFAULT_ROOT, *, source='annotated', frame_id=0,
            width=960, max_crops=4, force=False, detector=None):
    """Freeze image pixels and bounding boxes to disk. YOLO called once here."""
    import numpy as np
    root = Path(root)
    if (root / 'fixture.json').exists() and not force:
        raise FileExistsError('Fixture already exists. Use --force to intentionally replace it.')
    frames, provenance = load_source_images(source, frame_id, width)
    if detector is None:
        from .detector import PersonDetector
        detector = PersonDetector(device='cpu')
    detections = detector.detect(frames)
    boxes = select_boxes(detections, frames, max_crops)
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / 'frames.npz'
    np.savez_compressed(arrays_path, **frames)
    boxes_path = root / 'boxes.json'
    save_json(boxes_path, boxes)
    fixture = {'source': source, 'source_frame_id': frame_id, 'width': width,
               'image_shapes': {c:list(frames[c].shape) for c in CAMERAS},
               'source_details': provenance, 'cameras': list(CAMERAS),
               'crop_counts': {c:len(boxes[c]) for c in CAMERAS},
               'total_crops': sum(map(len, boxes.values())),
               'frames_sha256': sha256_file(arrays_path),
               'boxes_sha256': sha256_file(boxes_path),
               'person_model': 'models/yolo26n.pt',
               'reid_model': 'models/osnet_x0_25_msmt17.pt',
               'warning': 'Images and boxes are frozen for CPU benchmarking, not ground-truth accuracy.'}
    save_json(root/'fixture.json', fixture)
    return fixture


def load_fixture(root=DEFAULT_ROOT):
    import numpy as np
    root = Path(root)
    fixture = load_json(root / 'fixture.json')
    if fixture['cameras'] != list(CAMERAS):
        raise ValueError('Fixture camera list does not match the expected cameras')
    for name, field in [('frames.npz','frames_sha256'), ('boxes.json','boxes_sha256')]:
        if sha256_file(root / name) != fixture[field]:
            raise ValueError(f'{name} has changed. Prepare a new frozen fixture.')
    with np.load(root / 'frames.npz', allow_pickle=False) as data:
        frames = {c: data[c].copy() for c in CAMERAS}
    boxes = load_json(root / 'boxes.json')
    for c in CAMERAS:
        if list(frames[c].shape) != fixture['image_shapes'][c]:
            raise ValueError(f'Shape changed for {c}')
        if len(boxes[c]) != fixture['crop_counts'][c]:
            raise ValueError(f'Crop count changed for {c}')
    return frames, boxes, fixture


def affinity_selection(available, choice):
    """CPU indices are logical IDs; the first/last groups are NOT P/E classifications."""
    available = sorted(set(int(i) for i in available))
    if not available:
        raise ValueError('No available CPUs')
    if choice == 'all':
        return available
    if choice == 'first4':
        return available[:min(4,len(available))]
    if choice == 'last4':
        return available[-min(4,len(available)):]
    try:
        selected = sorted(set(int(x.strip()) for x in choice.split(',')))
    except ValueError as exc:
        raise ValueError(f'Invalid CPU affinity: {choice}') from exc
    if not selected or any(cpu not in available for cpu in selected):
        raise ValueError(f'Affinity CPUs {selected} must be a nonempty subset of {available}')
    return selected


def apply_affinity(proc, choice):
    try:
        allowed = proc.cpu_affinity()
    except (AttributeError, OSError) as exc:
        if choice == 'all':
            return None
        raise RuntimeError('CPU affinity unavailable on this operating system') from exc
    target = affinity_selection(allowed, choice)
    if target != sorted(allowed):
        proc.cpu_affinity(target)
    actual = sorted(proc.cpu_affinity())
    if actual != target:
        raise RuntimeError(f'CPU affinity not applied: wanted {target}, got {actual}')
    return actual


def configure_threads(threads, blas_threads):
    """Call prior to torch eager operations; verify after Ultralytics initialization."""
    if threads < 1 or blas_threads < 1:
        raise ValueError('Thread limits must be positive')
    import cv2
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    limiter = threadpool_limits(limits=blas_threads)
    return limiter, torch, {'torch_initial': int(torch.get_num_threads()),
                            'opencv': int(cv2.getNumThreads()),
                            'threadpools_initial': threadpool_info()}


def snapshot_host(proc):
    import psutil
    freq = psutil.cpu_freq()
    try:
        battery = psutil.sensors_battery()
    except (OSError, AttributeError, NotImplementedError):
        battery = None
    try:
        aff = proc.cpu_affinity()
    except (AttributeError, psutil.Error):
        aff = None
    return {'pid':proc.pid, 'affinity':aff,
            'thread_count':proc.num_threads(),
            'rss_mb': round(proc.memory_info().rss / 2**20, 2),
            'system_cpu_pct': psutil.cpu_percent(interval=None),
            'cpu_freq_reported_mhz':round(freq.current,1) if freq else None,
            'ac_power_connected':battery.power_plugged if battery else None,
            'note':'CPU frequency on Windows may be nominal, not live core clocks.'}


def measure(root=DEFAULT_ROOT, *, seconds=30, warmup=8, threads=4,
            blas_threads=None, affinity='all', workload='both', out=None,
            detector_factory=None, encoder_factory=None):
    if workload not in ('both','detector','encoder'):
        raise ValueError('workload must be both, detector, or encoder')
    if seconds <= 0 or warmup < 0:
        raise ValueError('seconds must be >0 and warmup >=0')
    import psutil
    from threadpoolctl import threadpool_info
    proc = psutil.Process()
    applied = apply_affinity(proc, affinity)
    blas_threads = threads if blas_threads is None else blas_threads
    limiter, torch, init = configure_threads(threads, blas_threads)
    frames, boxes, fixture = load_fixture(root)
    try:
        if workload in ('both','detector'):
            if detector_factory is None:
                from .detector import PersonDetector
                detector_factory = PersonDetector
            detector = detector_factory()
        else:
            detector = None
        if workload in ('both','encoder'):
            if encoder_factory is None:
                from .reid import AppearanceEncoder
                encoder_factory = AppearanceEncoder
            encoder = encoder_factory()
            # Only include cameras with nonempty fixed crop lists.
            requests = [(c, fixture['source_frame_id'], frames[c], boxes[c])
                        for c in CAMERAS if boxes[c]]
            if not requests:
                raise ValueError('No frozen person crops')
        else:
            encoder, requests = None, None
        # Ultralytics can reset torch threads during its first model.predict().
        # Initializing the predictor before enforcing threads is deliberate.
        if detector is not None:
            detector.detect(frames)
        torch.set_num_threads(threads)
        init['torch_after_model_setup'] = int(torch.get_num_threads())
        if init['torch_after_model_setup'] != threads:
            raise RuntimeError('Torch thread setting not applied after model setup')
        for _ in range(warmup):
            if detector is not None:
                detector.detect(frames)
            if encoder is not None:
                encoder.encode_batch(requests)
            if torch.get_num_threads() != threads:
                raise RuntimeError('Torch changed its thread limit during warmup')
        proc.cpu_percent(interval=None)
        before = time.perf_counter()
        rows = []
        last_sys = snapshot_host(proc)
        next_sample_at = 1.0
        while time.perf_counter() - before < seconds or not rows:
            start = time.perf_counter()
            if detector is not None:
                detector.detect(frames)
            detected = time.perf_counter()
            if encoder is not None:
                encoder.encode_batch(requests)
            end = time.perf_counter()
            if torch.get_num_threads() != threads:
                raise RuntimeError('Torch changed its thread limit during inference')
            # Instantaneous metrics are sampled at 1Hz to avoid per-batch overhead.
            if end - before >= next_sample_at:
                last_sys = snapshot_host(proc)
                next_sample_at = end - before + 1.0
            rows.append({'iteration':len(rows)+1, 'elapsed_s':round(end-before,3),
                         'yolo_ms':round((detected-start)*1000,3) if detector else None,
                         'osnet_ms':round((end-detected)*1000,3) if encoder else None,
                         'combined_ms':round((end-start)*1000,3),
                         'process_cpu_pct_one_core':None,
                         'system_cpu_pct':last_sys['system_cpu_pct'],
                         'rss_mb':last_sys['rss_mb']})
        elapsed = time.perf_counter() - before
        cpu_pct = proc.cpu_percent(interval=None)
        out = Path(out) if out is not None else Path(root) / 'runs' / f'{workload}_t{threads}_{affinity.replace(",","_")}'
        out.mkdir(parents=True, exist_ok=True)
        with (out/'per_iteration.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=MEASURE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        y = [r['yolo_ms'] for r in rows if r['yolo_ms'] is not None]
        e = [r['osnet_ms'] for r in rows if r['osnet_ms'] is not None]
        # Same immutable fixture across all trials; compare hashes in compare().
        summary = {'status':'completed', 'mode':'fixed-input', 'workload':workload,
                   'fixture_hashes':{'frames':fixture['frames_sha256'],
                                     'boxes':fixture['boxes_sha256']},
                   'image_shapes':fixture['image_shapes'],
                   'crops_per_four_camera_batch':fixture['total_crops'] if encoder else 0,
                   'iterations':len(rows), 'warmup_iterations':warmup,
                   'elapsed_s':round(elapsed,3),
                   'aggregate_equivalent_fps':round(4*len(rows)/elapsed,3),
                   'warning_equivalent_fps':'The same four frames are repeated; this is NOT RTSP throughput.',
                   'latency_ms':{'yolo':latency(y), 'osnet':latency(e),
                                 'combined':latency([r['combined_ms'] for r in rows])},
                   'yolo_osnet_latency_correlation':correlation(y,e) if workload=='both' else None,
                   'cpu':{'intraop_requested':threads,'intraop_actual':torch.get_num_threads(),
                          'blas_requested':blas_threads,'opencv_threads':init['opencv'],
                          'threadpool_info':threadpool_info(),
                          'initial':init, 'affinity_requested':affinity,
                          'affinity_actual':applied,
                          'process_cpu_pct_one_core':round(cpu_pct,2),
                          'process':snapshot_host(proc)},
                   'limitations':['Synthetic repeated frames are not live video or tracking.',
                                  'Logical CPU indexes do not identify P/E cores.',
                                  'Publisher load, power state and scheduling can vary across trials.',
                                  'BoxMOT encode time includes preprocessing, not just neural inference.']}
        save_json(out/'summary.json',summary)
        print(f"{out}: {len(rows)} iterations, combined p50={summary['latency_ms']['combined']['p50']} ms",flush=True)
        return summary
    finally:
        limiter.restore_original_limits()
        # Affinity is confined to a standalone fresh benchmark process.


def scenarios(*, threads=4, with_publishers=None):
    """Conservative first pass. Each case gets an independent subprocess."""
    cases = [('baseline_a',threads,'all',None),
             ('threads_2',2,'all',None),
             ('threads_8',8,'all',None),
             ('affinity_first4',threads,'first4',None),
             ('affinity_last4',threads,'last4',None),
             ('baseline_b',threads,'all',None)]
    if with_publishers is not None:
        if with_publishers not in ('live','copy'):
            raise ValueError('Publisher mode must be live or copy')
        cases.append((f'publishers_{with_publishers}',threads,'all',with_publishers))
        cases.append(('baseline_after_publishers',threads,'all',None))
    return cases


def execute_trial(cmd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ)
    # Fresh interpreter: native runtimes receive matching limits before import.
    with log_path.open('w',encoding='utf-8') as log:
        return subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env).returncode


def sweep(root=DEFAULT_ROOT, *, seconds=30, warmup=8, threads=4,
          publishers=None, prepared_root=None, rtsp_base='rtsp://127.0.0.1:8554',
          workload='both'):
    """Run one fresh Python process per combination; avoid native thread-pool residue."""
    root = Path(root)
    if not (root/'fixture.json').is_file():
        raise FileNotFoundError('Run: python -m src.fixed_inference prepare')
    completed = []
    for name, count, affinity, pub_mode in scenarios(threads=threads, with_publishers=publishers):
        out = root/'runs'/name
        out.mkdir(parents=True,exist_ok=True)
        run = {'name':name,'threads':count,'affinity':affinity,'publishers':pub_mode,
               'returncode':None,'error':None}
        procs, handles = {}, []
        try:
            if pub_mode:
                from . import publisher_experiment as p
                p.require_binaries('ffmpeg','ffprobe')
                p_root = Path(prepared_root) if prepared_root else p.DEFAULT_ROOT
                if pub_mode == 'copy':
                    p.verified_prepared(p_root)
                commands = p.publisher_commands(pub_mode,p_root,rtsp_base=rtsp_base)
                procs, handles = p.start_publishers(commands,out)
                p.verify_live_endpoints(procs,rtsp_base,ffprobe='ffprobe')
            cmd = [sys.executable,'-m','src.fixed_inference','run',
                   '--root',str(root),'--out',str(out),
                   '--seconds',str(seconds),'--warmup',str(warmup),
                   '--threads',str(count),'--affinity',affinity,'--workload',workload]
            run['returncode'] = execute_trial(cmd,out/'console.log')
            if run['returncode']:
                run['error'] = f'Benchmark exited with {run["returncode"]}; check {out / "console.log"}'
        except Exception as exc:
            run['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if procs:
                from . import publisher_experiment as p
                p.stop_publishers(procs,handles)
            run['completed'] = run['returncode']==0 and (out/'summary.json').exists()
            completed.append(run)
            save_json(root/'sweep_manifest.json', {'runs':completed})
            print(f'{name}: {"PASS" if run["completed"] else "FAIL"} '+str(run['error'] or ''),flush=True)
    return completed


def compare(root=DEFAULT_ROOT):
    root = Path(root)
    manifest = load_json(root/'sweep_manifest.json')
    results = []
    fixture_hash = None
    for run in manifest['runs']:
        outcome = dict(run)
        report_path = root/'runs'/run['name']/'summary.json'
        if run.get('completed') and report_path.is_file():
            summary = load_json(report_path)
            hashes = summary['fixture_hashes']
            if fixture_hash is None:
                fixture_hash = hashes
            if hashes != fixture_hash:
                raise RuntimeError('Fixture hashes differ across runs; cannot compare.')
            outcome.update({'equivalent_fps':summary['aggregate_equivalent_fps'],
                            'yolo_p50_ms':summary['latency_ms']['yolo']['p50'],
                            'yolo_p95_ms':summary['latency_ms']['yolo']['p95'],
                            'osnet_p50_ms':summary['latency_ms']['osnet']['p50'],
                            'osnet_p95_ms':summary['latency_ms']['osnet']['p95'],
                            'combined_p50_ms':summary['latency_ms']['combined']['p50'],
                            'combined_p95_ms':summary['latency_ms']['combined']['p95'],
                            'cpu_pct_one_core':summary['cpu']['process_cpu_pct_one_core'],
                            'affinity_actual':summary['cpu']['affinity_actual']})
        results.append(outcome)
    summary = {'basis':'Identical frozen images and crop coordinates; each case uses a fresh interpreter.',
               'fixture_hashes':fixture_hash,'cases':results,
               'caveats':['Equivalent FPS is repeated four-camera inference, not RTSP FPS.',
                          'First/last logical IDs do not establish P-core/E-core identity.',
                          'A single trial is not causal proof; repeat in reverse order.',
                          'Publishing uses host CPU but the benchmark has no RTSP decoding.',
                          'Any failed runs must be reviewed in console.log.']}
    save_json(root/'comparison.json',summary)
    with (root/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        fields = ['name','completed','threads','affinity','publishers','equivalent_fps',
                  'yolo_p50_ms','osnet_p50_ms','combined_p95_ms','cpu_pct_one_core']
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--source',choices=('annotated','video'),default='annotated')
    p.add_argument('--frame-id',type=int,default=0)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--max-crops',type=int,default=4)
    p.add_argument('--force',action='store_true')
    p = subs.add_parser('run')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--blas-threads',type=int)
    p.add_argument('--affinity',default='all',help='all, first4, last4, or comma-separated logical CPU IDs')
    p.add_argument('--workload',choices=('both','detector','encoder'),default='both')
    p = subs.add_parser('sweep')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--publishers',choices=('live','copy'))
    p.add_argument('--prepared-root',type=Path)
    p.add_argument('--rtsp-base',default='rtsp://127.0.0.1:8554')
    p.add_argument('--workload',choices=('both','detector','encoder'),default='both')
    p = subs.add_parser('compare')
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command=='prepare':
            result = prepare(args.root,source=args.source,frame_id=args.frame_id,
                             width=args.width,max_crops=args.max_crops,force=args.force)
            print(json.dumps(result,indent=2))
        elif args.command=='run':
            measure(args.root,seconds=args.seconds,warmup=args.warmup,threads=args.threads,
                    blas_threads=args.blas_threads,affinity=args.affinity,
                    workload=args.workload,out=args.out)
        elif args.command=='sweep':
            runs = sweep(args.root,seconds=args.seconds,warmup=args.warmup,
                         threads=args.threads,publishers=args.publishers,
                         prepared_root=args.prepared_root,rtsp_base=args.rtsp_base,
                         workload=args.workload)
            if not all(r['completed'] for r in runs):
                raise RuntimeError('Some trials failed; check sweep_manifest.json and console logs')
        else:
            print(json.dumps(compare(args.root),indent=2))
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2) from exc

if __name__=='__main__':
    main()
