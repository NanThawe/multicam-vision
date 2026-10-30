"""Offline throughput and latency benchmark for the four-camera pipeline.

Run from the project root:
    python -m src.benchmark
    python -m src.benchmark --with-reid --batches 30

This is an offline benchmark, NOT a live RTSP frame-drop measurement.
"""

import argparse
import csv
import json
import itertools
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "benchmarks"
STAGES = ("decode_ms", "detection_ms", "tracking_ms", "reid_ms", "events_ms", "total_ms")


def latency_stats(values):
    """Mean, median (P50) and interpolated P95 in milliseconds."""
    values = sorted(values)
    if not values:
        raise ValueError("No latency samples")

    def percentile(q):
        pos = (len(values) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {"mean": round(statistics.mean(values), 2),
            "p50": round(percentile(0.5), 2),
            "p95": round(percentile(0.95), 2)}


def run_benchmark(reader, detector, tracker, event_engine, *, matcher=None,
                  batches=30, warmup=3, clock=time.perf_counter,
                  synchronize=lambda: None, rss_mb=lambda: None,
                  check_threads=None):
    """Time full four-camera batches without writing videos or repeating model setup.

    All stage timings include their relevant CUDA work when `synchronize` is
    configured. `rss_mb` is sampled, not a guaranteed process peak.
    """
    if batches <= 0 or warmup < 0:
        raise ValueError("batches must be > 0 and warmup must be >= 0")

    samples = []
    cameras = None
    timed_start = None
    warmup_done = 0
    event_count = 0
    reid_start_stats = None
    encoder_start_times = None

    for index in range(warmup + batches):
        if check_threads is not None:
            check_threads()
        start = clock()
        try:
            frame_id, frames = next(reader)
        except StopIteration:
            break
        decoded = clock()

        if cameras is None:
            cameras = tuple(frames)
            if not cameras:
                raise ValueError("Frame reader returned no camera images")
        elif tuple(frames) != cameras:
            raise ValueError("Camera list changed during the benchmark")

        synchronize()
        detections = detector.detect(frames)
        synchronize()
        detected = clock()

        tracks = tracker.update(frames, detections)
        synchronize()
        tracked = clock()

        if matcher is not None:
            if index == warmup and hasattr(matcher, "stats"):
                reid_start_stats = dict(matcher.stats)
                if hasattr(matcher.encoder, "timings_ms"):
                    encoder_start_times = dict(matcher.encoder.timings_ms)
            tracks, _ = matcher.update(frame_id, frames, tracks)
        synchronize()
        identified = clock()

        events = sum(len(event_engine.update(frame_id, camera, tracks[camera]))
                     for camera in cameras)
        synchronize()
        finished = clock()
        if check_threads is not None:
            check_threads()

        if index < warmup:
            warmup_done += 1
            continue
        if timed_start is None:
            timed_start = start
        event_count += events
        samples.append({
            "frame_id": frame_id,
            "decode_ms": (decoded - start) * 1000,
            "detection_ms": (detected - decoded) * 1000,
            "tracking_ms": (tracked - detected) * 1000,
            "reid_ms": (identified - tracked) * 1000 if matcher else 0.0,
            "events_ms": (finished - identified) * 1000,
            "total_ms": (finished - start) * 1000,
            "rss_mb": rss_mb(),
        })

    if not samples:
        raise ValueError("No measured frames. Reduce --warmup or check video duration")

    wall_s = max(clock() - timed_start, 1e-9)
    count = len(samples)
    indices = [s["frame_id"] for s in samples]
    intentional_skips = sum(max(0, b - a - 1)
                            for a, b in zip(indices, indices[1:]))
    rss_values = [s["rss_mb"] for s in samples if s["rss_mb"] is not None]
    report = {
        "measured_batches": count,
        "warmup_batches": warmup_done,
        "cameras": list(cameras),
        "wall_time_s": round(wall_s, 3),
        "fps_per_camera": {camera: round(count / wall_s, 3) for camera in cameras},
        "aggregate_fps": round(count * len(cameras) / wall_s, 3),
        "latency_ms_per_four_camera_batch": {
            stage.removesuffix("_ms"): latency_stats([s[stage] for s in samples])
            for stage in STAGES
        },
        "memory": {
            "max_sampled_process_rss_mb": round(max(rss_values), 1) if rss_values else None,
        },
        "event_count": event_count,
        "sampling": {
            "first_measured_frame": indices[0],
            "last_measured_frame": indices[-1],
            "intentionally_skipped_source_frames_per_camera": intentional_skips,
            "live_dropped_frames": None,
        },
    }
    if matcher is not None and reid_start_stats is not None:
        reid_delta = {name: matcher.stats[name] - reid_start_stats[name]
                      for name in reid_start_stats}
        eligible = reid_delta.get("eligible_crops", 0)
        report["reid_workload"] = {
            **reid_delta,
            "embedding_reuse_fraction": round(
                reid_delta.get("reused_crops", 0) / eligible, 3
            ) if eligible else None,
        }
    if encoder_start_times is not None:
        measured = {name: round(matcher.encoder.timings_ms[name] - before, 2)
                    for name, before in encoder_start_times.items()}
        report["reid_encoder_profile"] = {
            "total_ms": measured,
            "mean_ms_per_batch": {name: round(value / count, 2)
                                  for name, value in measured.items()},
        }
    return report, samples


def save_results(report, samples, output_dir=OUTPUT_DIR):
    """Save human-readable aggregate metrics and per-batch timings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    csv_path = output_dir / "per_batch.csv"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["frame_id", *STAGES, "rss_mb"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v
                             for k, v in sample.items()})
    return report_path, csv_path


def read_video_fps(videos):
    """Read source metadata; framesource.py performs the actual decoding."""
    import cv2

    rates = {}
    for camera, path in videos.items():
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise FileNotFoundError(f"Cannot open {camera} video: {path}")
            rates[camera] = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
    return rates


def configure_cpu_threads(torch_module, ultralytics_torch_utils, requested_threads):
    """Set threads before inference and prevent Ultralytics resetting the value.

    Ultralytics select_device() calls set_num_threads(NUM_THREADS) on CPU.
    NUM_THREADS is an internal library constant, so this compatibility adapter
    is strictly for benchmarking and must be checked after library upgrades.
    """
    if requested_threads is None:
        return
    if not hasattr(ultralytics_torch_utils, "NUM_THREADS"):
        raise RuntimeError("This Ultralytics version has no NUM_THREADS constant")
    ultralytics_torch_utils.NUM_THREADS = requested_threads
    torch_module.set_num_threads(requested_threads)


def prepare_inference(reader, detector, torch_module, requested_threads=None):
    """Initialise the lazy YOLO predictor and observe the effective threads.

    Return the first camera batch to the benchmark, preserving frame coverage.
    Thread settings must already be configured before any inference starts.
    """
    try:
        first_batch = next(reader)
    except StopIteration as exc:
        raise ValueError("No frames available for detector initialisation") from exc
    detector.detect(first_batch[1])
    active_threads = torch_module.get_num_threads()
    if requested_threads is not None and active_threads != requested_threads:
        raise RuntimeError(
            f"Requested {requested_threads} PyTorch threads, but {active_threads} are active"
        )
    return itertools.chain((first_batch,), reader), active_threads


def check_intraop_threads(torch_module, expected):
    """Reject benchmarks silently altered by a library changing threads."""
    observed = torch_module.get_num_threads()
    if observed != expected:
        raise RuntimeError(
            f"PyTorch thread count changed during benchmark: expected {expected}, "
            f"observed {observed}. This run is not valid for comparison."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=30,
                        help="Measured frame batches after warm-up (default: 30)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Untimed frame batches (default: 3)")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--device", default="cpu", help="YOLO device: cpu or 0 (CUDA GPU)")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="CPU intra-op threads for PyTorch; omit to keep default")
    parser.add_argument("--with-reid", action="store_true",
                        help="Include OSNet/ReID (currently CPU, regardless of YOLO device)")
    parser.add_argument("--reid-interval", type=int, default=1,
                        help="Refresh established track embeddings every N sampled batches")
    parser.add_argument("--reid-backend", choices=("pytorch", "onnx"), default="pytorch",
                        help="ReID inference backend (default: pytorch)")
    parser.add_argument("--batch-across-cameras", action="store_true",
                        help="Combine ReID crops across cameras in one BoxMOT encode call")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Results folder (use different folders for comparisons)")
    args = parser.parse_args(argv)
    if args.batches <= 0 or args.warmup < 0 or not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error("batches and sample-fps must be positive; warmup must be >= 0")
    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("--torch-threads must be a positive integer")
    if args.reid_interval < 1:
        parser.error("--reid-interval must be >= 1")
    if args.reid_interval != 1 and not args.with_reid:
        parser.error("--reid-interval requires --with-reid")
    if args.batch_across_cameras and not args.with_reid:
        parser.error("--batch-across-cameras requires --with-reid")
    if args.reid_backend == "onnx" and not args.with_reid:
        parser.error("--reid-backend onnx requires --with-reid")

    # Heavy ML libraries are imported only for real benchmark runs, not unit tests.
    import torch
    # PyTorch recommends configuring threads before any eager inference.
    # Ultralytics resets this count when its CPU predictor starts, so align
    # that internal default for this controlled benchmark only.
    if args.torch_threads is not None:
        from ultralytics.utils import torch_utils as ultralytics_torch_utils
        configure_cpu_threads(torch, ultralytics_torch_utils, args.torch_threads)
    from .detector import PersonDetector
    from .events import EventEngine
    from .framesource import VIDEOS, iter_frames
    from .tracker import MultiCameraTracker

    use_cuda = args.device.lower() in ("0", "cuda", "cuda:0")
    if use_cuda and not torch.cuda.is_available():
        parser.error("CUDA device requested, but torch.cuda.is_available() is False")
    if not use_cuda and args.device.lower() != "cpu":
        parser.error("Supported devices: cpu, 0, cuda, cuda:0")

    try:
        import psutil
        process = psutil.Process()
        rss_mb = lambda: process.memory_info().rss / (1024 ** 2)
    except ImportError:
        print("Optional: pip install psutil  (RAM measurement unavailable)")
        rss_mb = lambda: None

    source_rates = read_video_fps(VIDEOS)
    stride = max(1, round(next(iter(source_rates.values())) / args.sample_fps))
    print("Loading models; startup time is excluded from benchmark throughput.")
    setup_start = time.perf_counter()
    detector = PersonDetector(device=args.device)
    tracker = MultiCameraTracker()
    matcher = None
    if args.with_reid:
        from .reid import AppearanceEncoder, CrossCameraMatcher
        matcher = CrossCameraMatcher(
            AppearanceEncoder(backend=args.reid_backend), refresh_every=args.reid_interval,
            batch_across_cameras=args.batch_across_cameras)
    event_engine = EventEngine()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    synchronize = torch.cuda.synchronize if use_cuda else lambda: None
    reader = iter_frames(sample_fps=args.sample_fps)
    try:
        # Bootstrapping an actual input batch forces lazy YOLO device setup.
        # Replay that batch in the benchmark to preserve identical workloads.
        prepared_reader, active_threads = prepare_inference(
            reader, detector, torch, args.torch_threads)
        model_setup_s = time.perf_counter() - setup_start
        print(f"PyTorch intra-op threads: requested={args.torch_threads or 'auto'}, "
              f"active={active_threads}", flush=True)
        def verify_threads():
            check_intraop_threads(torch, active_threads)
        report, samples = run_benchmark(
            prepared_reader, detector, tracker, event_engine, matcher=matcher,
            batches=args.batches, warmup=args.warmup,
            synchronize=synchronize, rss_mb=rss_mb,
            check_threads=verify_threads,
        )
        verify_threads()
    finally:
        reader.close()

    report["configuration"] = {
        "device": args.device,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_intraop_threads_verified": True,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_threads_requested": args.torch_threads,
        "osnet_device": "cpu" if args.with_reid else None,
        "reid_enabled": bool(args.with_reid),
        "reid_backend": args.reid_backend if args.with_reid else None,
        "reid_interval": args.reid_interval if args.with_reid else None,
        "reid_batch_across_cameras": args.batch_across_cameras if args.with_reid else None,
        "requested_sample_fps": args.sample_fps,
        "source_video_fps": source_rates,
        "sampling_stride": stride,
        "effective_input_sample_fps": {
            camera: round(fps / stride, 3) for camera, fps in source_rates.items()
        },
        "model_setup_s_excluded": round(model_setup_s, 3),
        "preview_encoding_included": False,
        "mqtt_included": False,
    }
    report["memory"]["pytorch_peak_allocated_gpu_mb"] = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if use_cuda else None
    )
    target = max(source_rates.values()) / stride
    report["throughput_vs_input_sampling_ratio"] = round(
        next(iter(report["fps_per_camera"].values())) / target, 3
    )
    report["notes"] = (
        "Offline throughput, not live RTSP latency. Each stage is timed per four-camera "
        "batch, including batched YOLO inference. Sampling skips are intentional; "
        "live dropped frames are not measurable. RAM is sampled process RSS, not "
        "a guaranteed peak. GPU measurement is PyTorch allocated memory only."
    )

    report_path, csv_path = save_results(report, samples, args.output_dir)
    print(f"Measured {report['measured_batches']} batches across {len(report['cameras'])} cameras")
    print(f"Per-camera FPS: {next(iter(report['fps_per_camera'].values())):.2f} | "
          f"Aggregate FPS: {report['aggregate_fps']:.2f}")
    for stage, stats in report["latency_ms_per_four_camera_batch"].items():
        if stage == "reid" and not args.with_reid:
            continue
        print(f"{stage:>9} P50={stats['p50']:.1f}ms P95={stats['p95']:.1f}ms")
    if "reid_workload" in report:
        print(f"ReID workload: {report['reid_workload']}")
    if "reid_encoder_profile" in report:
        print(f"ReID encoder profile (measured totals ms): "
              f"{report['reid_encoder_profile']['total_ms']}")
    print(f"Saved {report_path} and {csv_path}")


if __name__ == "__main__":
    main()
