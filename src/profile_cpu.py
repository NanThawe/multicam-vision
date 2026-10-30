"""Repeat CPU-thread benchmarks in separate processes, then compare workloads.

Run from project root:
    python -m src.profile_cpu --threads auto,2,4,8 --repeats 3 --batches 100

No model weights or videos are read by this coordinator. It starts the existing
src.benchmark pipeline with a fresh process and the same settings each run.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmarks" / "step12_verified"


def parse_threads(raw):
    """Interpret e.g. 'auto,2,4,8', preserving order without duplicates."""
    tokens = [part.strip().lower() for part in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("Expected comma-separated values, e.g. auto,2,4,8")
    labels = []
    for token in tokens:
        if token == "auto":
            label = "auto"
        else:
            try:
                n = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid thread count: {token}") from exc
            if n < 1:
                raise ValueError("Thread counts must be positive")
            label = str(n)
        if label not in labels:
            labels.append(label)
    return labels


def benchmark_command(label, directory, *, batches, warmup, sample_fps=5.0):
    """Build a fresh-interpreter benchmark using the production pipeline."""
    command = [
        sys.executable, "-m", "src.benchmark", "--with-reid",
        "--reid-backend", "pytorch", "--reid-interval", "5",
        "--batch-across-cameras", "--device", "cpu",
        "--batches", str(batches), "--warmup", str(warmup),
        "--sample-fps", str(sample_fps), "--output-dir", str(directory),
    ]
    if label != "auto":
        command.extend(("--torch-threads", label))
    return command


def workload_signature(report):
    """Work in different runs must match before ranking speed."""
    sampling = report["sampling"]
    work = report["reid_workload"]
    config = report["configuration"]
    return (
        tuple(report["cameras"]), report["measured_batches"],
        sampling["first_measured_frame"], sampling["last_measured_frame"],
        work["eligible_crops"], work["encoded_crops"],
        work.get("encoder_calls"), report["event_count"],
        config.get("reid_interval"), config.get("reid_batch_across_cameras"),
        config.get("reid_backend"), config.get("device"),
        config.get("sampling_stride"),
    )


def summarize_runs(runs):
    """Compare grouped reports; never pick a winner on differing workloads."""
    if not runs:
        raise ValueError("No benchmark reports supplied")
    signatures = [workload_signature(report) for reports in runs.values()
                  for report in reports]
    comparable = len(set(signatures)) == 1
    summary = {}
    for label, reports in runs.items():
        if not reports:
            raise ValueError(f"No reports for {label}")
        fps = [r["aggregate_fps"] for r in reports]
        summary[label] = {
            "runs": len(reports),
            "aggregate_fps_median": round(statistics.median(fps), 3),
            "aggregate_fps_range": [round(min(fps), 3), round(max(fps), 3)],
            "reid_p50_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["reid"]["p50"] for r in reports), 2),
            "total_p95_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["total"]["p95"] for r in reports), 2),
            "model_encode_ms_per_batch_median": round(statistics.median(
                r["reid_encoder_profile"]["mean_ms_per_batch"]["model_encode"]
                for r in reports), 2),
            "input_prep_ms_per_batch_median": round(statistics.median(
                r["reid_encoder_profile"]["mean_ms_per_batch"]["input_prep"]
                for r in reports), 2),
            "reported_torch_threads": sorted({r["configuration"]["torch_intraop_threads"]
                                              for r in reports}),
            "eligible_crops_per_run": [r["reid_workload"]["eligible_crops"] for r in reports],
            "encoded_crops_per_run": [r["reid_workload"]["encoded_crops"] for r in reports],
        }
    threads_verified = all(
        r["configuration"].get("torch_intraop_threads_verified") is True
        and (label == "auto" or r["configuration"].get("torch_intraop_threads") == int(label))
        for label, reports in runs.items() for r in reports
    )
    best = (max(summary, key=lambda name: summary[name]["aggregate_fps_median"])
            if comparable and threads_verified else None)
    return {
        "workloads_comparable": comparable,
        "thread_settings_verified": threads_verified,
        "fastest_setting_if_comparable": best,
        "thread_settings": summary,
        "caution": (
            "Offline whole-pipeline CPU throughput; model weights and decision logic unchanged. "
            "PyTorch threading affects both YOLO and OSNet. Fastest means fastest observed, "
            "not necessarily statistically significant. This is not ReID accuracy or live RTSP FPS."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="auto,2,4,8",
                        help="PyTorch CPU thread counts, e.g. auto,2,4,8")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, no models/videos required")
    args = parser.parse_args(argv)
    if args.batches < 1 or args.warmup < 0 or args.repeats < 1 or args.sample_fps <= 0:
        parser.error("batches, repeats and sample-fps must be positive; warmup cannot be negative")
    try:
        labels = parse_threads(args.threads)
    except ValueError as exc:
        parser.error(str(exc))

    runs = {label: [] for label in labels}
    total = len(labels) * args.repeats
    step = 0
    for repeat in range(1, args.repeats + 1):
        # Reverse alternate rounds to reduce consistent ordering/thermal bias.
        for label in (labels if repeat % 2 else list(reversed(labels))):
            step += 1
            run_dir = args.output_dir / f"threads_{label}_run{repeat}"
            command = benchmark_command(label, run_dir, batches=args.batches,
                                        warmup=args.warmup, sample_fps=args.sample_fps)
            print(f"[{step}/{total}] PyTorch threads={label}, repeat={repeat}", flush=True)
            if args.dry_run:
                print(" ".join(str(part) for part in command), flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "run.log").open("w", encoding="utf-8") as logfile:
                outcome = subprocess.run(command, cwd=ROOT, stdout=logfile,
                                         stderr=subprocess.STDOUT, check=False)
            if outcome.returncode:
                tail = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
                print("\n".join(tail.splitlines()[-35:]), file=sys.stderr)
                raise SystemExit(f"Benchmark failed (exit {outcome.returncode}); see {run_dir / 'run.log'}")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if report["configuration"]["reid_backend"] != "pytorch":
                raise ValueError("Expected PyTorch report")
            if not report["configuration"].get("torch_intraop_threads_verified"):
                raise ValueError(f"Thread count not verified: {run_dir}")
            active = report["configuration"]["torch_intraop_threads"]
            if label != "auto" and active != int(label):
                raise ValueError(f"Requested {label} threads, got {active}: {run_dir}")
            runs[label].append(report)
            print(f"  {report['aggregate_fps']:.2f} aggregate FPS", flush=True)

    if args.dry_run:
        return
    result = summarize_runs(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {path}")
    if not result["workloads_comparable"] or not result["thread_settings_verified"]:
        print("WARNING: workloads or thread counts differ; do not rank these settings by FPS.")


if __name__ == "__main__":
    main()
