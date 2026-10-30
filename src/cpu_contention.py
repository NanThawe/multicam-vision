"""Step 14.11: independent Windows host CPU monitor for the four-camera RTSP runner.

Run, from the project root, with the four RTSP publishers already running:
    python -m src.cpu_contention run --mode reid --seconds 120
    python -m src.cpu_contention run --mode no_reid --seconds 120
    python -m src.cpu_contention compare

Uses the Step 14.11 src/live_rtsp.py measurement_timing timestamps to align
host samples to per-batch timing. No model parameters or RTSP logic change.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError as exc:
    raise SystemExit("psutil is required for CPU monitoring. Install it in the project venv: python -m pip install psutil") from exc

from .framesource import ROOT

DEFAULT_ROOT = ROOT / "outputs" / "rtsp" / "step14_11"
FIELDS = (
    "timestamp_unix_s", "system_cpu_pct", "max_core_cpu_pct",
    "python_cpu_pct_one_core", "ffmpeg_cpu_pct_one_core", "docker_cpu_pct_one_core",
    "python_threads", "python_rss_mb", "python_affinity_count",
    "ffmpeg_processes", "ffmpeg_threads", "docker_processes",
    "inference_pid", "launcher_pid", "inference_pid_verified", "ffmpeg_scoped_to_publishers",
    "memory_used_pct", "cpu_freq_reported_mhz", "battery_percent", "ac_power_connected",
)
JOIN_FIELDS = (
    "elapsed_s", "detector_ms", "reid_ms", "reid_boxmot_encode_ms",
    "reid_encoded_crops", "processing_ms", "system_cpu_pct",
    "python_cpu_pct_one_core", "ffmpeg_cpu_pct_one_core", "docker_cpu_pct_one_core",
    "python_threads", "python_rss_mb", "host_sample_gap_s",
)
WINDOW_FIELDS = (
    "start_s", "end_s", "batches", "aggregate_fps", "detector_p50_ms",
    "reid_p50_ms", "boxmot_p50_ms", "encoded_crops_p50", "system_cpu_p50_pct",
    "python_cpu_p50_one_core", "ffmpeg_cpu_p50_one_core",
    "docker_cpu_p50_one_core", "python_threads_p50", "python_rss_p50_mb",
    "host_samples", "slow_batches",
)


def median(values):
    values = [float(x) for x in values if x is not None and x != ""]
    return round(statistics.median(values), 3) if values else None


def correlation(x, y):
    """Pearson r for matching finite pairs, descriptive only (not causality)."""
    pairs = [(float(a), float(b)) for a, b in zip(x, y)
             if a not in (None, "") and b not in (None, "")
             and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 4:
        return None
    xx, yy = zip(*pairs)
    mx, my = statistics.mean(xx), statistics.mean(yy)
    denom = math.sqrt(sum((a - mx)**2 for a in xx) * sum((b - my)**2 for b in yy))
    return round(sum((a - mx)*(b-my) for a, b in pairs)/denom, 4) if denom > 0 else None


def load_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _safe_call(callable_, default=None):
    try:
        return callable_()
    except (psutil.Error, OSError, PermissionError, AttributeError):
        return default


def _is_ffmpeg(name):
    return name.casefold() in {"ffmpeg", "ffmpeg.exe"}


def _is_docker_cpu(name):
    return name.casefold() in {"com.docker.backend.exe", "com.docker.backend", "vmmem", "vmmemwsl"}


def resolve_inference_process(launcher, target_cache=None):
    """Return (process, verified) for the Python interpreter doing inference.

    On Windows a virtualenv python.exe can be a small forwarding launcher
    with a child python.exe that loads the actual model. Prefer the deepest
    descendant with the -m src.live_rtsp command line, not launcher.pid.
    The boolean indicates whether attribution has independent evidence.
    """
    try:
        descendants = launcher.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    candidates = []
    for proc in [launcher, *descendants]:
        try:
            name = proc.name().lower()
            argv = proc.cmdline()
            marker_found = any("src.live_rtsp" in arg for arg in argv)
            if "python" not in name or (proc.pid == launcher.pid and not marker_found):
                continue
            # A descendant Python interpreter may have a rewritten argv on
            # Windows. Its ancestry from the src.live_rtsp launcher is evidence.
            rss = proc.memory_info().rss
            candidates.append((int(proc.pid != launcher.pid), rss, proc))
        except (psutil.Error, OSError):
            continue
    if not candidates:
        return None, False
    # Descendant preference prevents choosing a small Windows forwarding shim.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    child_selected, rss, proc = candidates[0]
    # If no descendants are available, require real interpreter memory usage.
    # A 5-10 MB Windows shim is not the PyTorch worker.
    verified = bool(child_selected or sys.platform != "win32" or rss >= 32 * 1024 * 1024)
    if target_cache is not None:
        try:
            identity = (proc.pid, proc.create_time())
            if identity not in target_cache:
                target_cache[identity] = proc
            proc = target_cache[identity]  # keep psutil's CPU delta state across samples
        except (psutil.Error, OSError):
            pass
    return proc, verified


def read_host_sample(target, cache, *, launcher_pid=None, verified=True, ffmpeg_pids=None):
    """Nonblocking CPU sampling; process percentages use *one core* as 100%.

    psutil's first process CPU reading is always priming, not a measurement.
    When ffmpeg_pids is None, all host FFmpeg instances are counted. Docker's VM CPU attribution is approximate.
    """
    now = time.time()
    per_core = _safe_call(lambda: psutil.cpu_percent(interval=None, percpu=True), [])
    memory = _safe_call(psutil.virtual_memory)
    frequency = _safe_call(psutil.cpu_freq)
    battery = _safe_call(psutil.sensors_battery)
    python_cpu = _safe_call(lambda: target.cpu_percent(interval=None)) if target and verified else None
    python_threads = _safe_call(target.num_threads) if target and verified else None
    rss = _safe_call(lambda: target.memory_info().rss) if target and verified else None
    affinity = _safe_call(lambda: len(target.cpu_affinity())) if target and verified else None
    ff_cpu, dock_cpu, ff_threads, nff, ndock = 0.0, 0.0, 0, 0, 0
    live = set()
    for proc in psutil.process_iter(attrs=["name"]):
        try:
            name = proc.info.get("name") or ""
            if not (_is_ffmpeg(name) or _is_docker_cpu(name)):
                continue
            if _is_ffmpeg(name) and ffmpeg_pids is not None and proc.pid not in ffmpeg_pids:
                continue
            key = (proc.pid, proc.create_time())
            live.add(key)
            cached = cache.get(key)
            if cached is None:
                cache[key] = proc
                proc.cpu_percent(interval=None)  # prime new process
                usage = 0.0
            else:
                usage = cached.cpu_percent(interval=None)
            if _is_ffmpeg(name):
                nff += 1
                ff_cpu += usage
                ff_threads += proc.num_threads()
            else:
                ndock += 1
                dock_cpu += usage
        except (psutil.Error, OSError):
            continue
    for key in list(cache):
        if key not in live:
            del cache[key]
    return {
        "timestamp_unix_s": round(now, 4),
        "inference_pid": target.pid if target is not None else None,
        "launcher_pid": launcher_pid,
        "inference_pid_verified": bool(verified and target is not None),
        "ffmpeg_scoped_to_publishers": ffmpeg_pids is not None,
        "system_cpu_pct": round(statistics.mean(per_core), 2) if per_core else None,
        "max_core_cpu_pct": round(max(per_core), 2) if per_core else None,
        "python_cpu_pct_one_core": round(python_cpu, 2) if python_cpu is not None else None,
        "ffmpeg_cpu_pct_one_core": round(ff_cpu, 2),
        "docker_cpu_pct_one_core": round(dock_cpu, 2),
        "python_threads": python_threads,
        "python_rss_mb": round(rss / 1048576, 2) if rss is not None else None,
        "python_affinity_count": affinity,
        "ffmpeg_processes": nff,
        "ffmpeg_threads": ff_threads,
        "docker_processes": ndock,
        "memory_used_pct": round(memory.percent, 2) if memory else None,
        # On Windows psutil.cpu_freq often returns nominal, not live, frequency.
        "cpu_freq_reported_mhz": round(frequency.current, 2) if frequency else None,
        "battery_percent": round(battery.percent, 2) if battery else None,
        "ac_power_connected": battery.power_plugged if battery else None,
    }


def prepare_host_monitor(target):
    psutil.cpu_percent(interval=None, percpu=True)
    if target is not None:
        target.cpu_percent(interval=None)
    return {}


def _float(row, key):
    try:
        return float(row[key]) if row.get(key) not in (None, "") else None
    except (ValueError, TypeError):
        return None


def join_batches(rows, samples, measured_start_unix_s, *, max_gap_s=1.0):
    """Closest point-in-time host sample to each batch completion timestamp."""
    import bisect
    ordered = sorted(samples, key=lambda r: float(r["timestamp_unix_s"]))
    timestamps = [float(r["timestamp_unix_s"]) for r in ordered]
    joined = []
    for row in rows:
        elapsed = _float(row, "elapsed_s")
        if elapsed is None:
            continue
        t = measured_start_unix_s + elapsed
        i = bisect.bisect_left(timestamps, t)
        candidates = [ordered[j] for j in (i-1, i) if 0 <= j < len(ordered)]
        chosen = min(candidates, key=lambda s: abs(float(s["timestamp_unix_s"])-t)) if candidates else None
        gap = abs(float(chosen["timestamp_unix_s"]) - t) if chosen else None
        if gap is not None and gap > max_gap_s:
            chosen = None
        result = {k: _float(row, k) for k in JOIN_FIELDS}
        if chosen:
            for k in ("system_cpu_pct", "python_cpu_pct_one_core", "ffmpeg_cpu_pct_one_core",
                      "docker_cpu_pct_one_core", "python_threads", "python_rss_mb"):
                result[k] = _float(chosen, k)
        result["host_sample_gap_s"] = round(gap, 3) if gap is not None else None
        joined.append(result)
    return joined


def build_windows(batch_rows, samples, start_unix_s, end_unix_s, *, width_s=10):
    """Time-aligned full measurement windows, not bootstrap/warmup samples."""
    if width_s <= 0:
        raise ValueError("width_s must be positive")
    n = int(max(0, end_unix_s - start_unix_s) // width_s)
    windows = []
    for index in range(n):
        lo, hi = index * width_s, (index + 1) * width_s
        batch = [r for r in batch_rows if (elapsed := _float(r, "elapsed_s")) is not None
                 and lo < elapsed <= hi]
        host = [s for s in samples if lo <= float(s["timestamp_unix_s"])-start_unix_s < hi]
        def bm(field):
            return median(_float(b, field) for b in batch)
        def hm(field):
            return median(_float(h, field) for h in host)
        windows.append({
            "start_s": lo, "end_s": hi, "batches": len(batch),
            "aggregate_fps": round(4 * len(batch)/width_s, 3),
            "detector_p50_ms": bm("detector_ms"),
            "reid_p50_ms": bm("reid_ms"),
            "boxmot_p50_ms": bm("reid_boxmot_encode_ms"),
            "encoded_crops_p50": bm("reid_encoded_crops"),
            "system_cpu_p50_pct": hm("system_cpu_pct"),
            "python_cpu_p50_one_core": hm("python_cpu_pct_one_core"),
            "ffmpeg_cpu_p50_one_core": hm("ffmpeg_cpu_pct_one_core"),
            "docker_cpu_p50_one_core": hm("docker_cpu_pct_one_core"),
            "python_threads_p50": hm("python_threads"),
            "python_rss_p50_mb": hm("python_rss_mb"),
            "host_samples": len(host),
            "slow_batches": sum((_float(r, "detector_ms") or 0) > 200 or
                                (_float(r, "reid_boxmot_encode_ms") or 0) > 300
                                for r in batch),
        })
    return windows


def analyse(directory, *, window_s=10, sample_interval=0.5, returncode=None):
    directory = Path(directory)
    report = load_json(directory / "report.json") or {}
    raw_batch = load_csv(directory / "per_batch.csv")
    samples = load_csv(directory / "host_samples.csv")
    timing = report.get("measurement_timing") or {}
    start = timing.get("start_unix_s")
    end = timing.get("end_unix_s")
    analysis = {
        "run_status": report.get("run_status", "missing"),
        "complete": bool(report.get("complete")),
        "returncode": returncode,
        "aggregate_fps": report.get("aggregate_processed_fps"),
        "batch_p95_ms": (report.get("processing_ms_per_four_camera_batch") or {}).get("p95"),
        "host_samples_captured": len(samples),
        "measured_batches": len(raw_batch),
        "aligned_windows": 0,
        "warnings": [],
        "measurement_timing": timing,
        "pearson_window_correlations": {},
    }
    if start is None or end is None:
        analysis["warnings"].append(
            "No measurement_timing in report.json: install Step 14.11 live_rtsp.py. "
            "Host samples cannot be aligned reliably to batch latencies.")
        save_json(directory / "analysis.json", analysis)
        return analysis
    batch = join_batches(raw_batch, samples, float(start), max_gap_s=max(1.0, 2*sample_interval))
    windows = build_windows(raw_batch, samples, float(start), float(end), width_s=window_s)
    save_csv(directory / "joined_batches.csv", batch, JOIN_FIELDS)
    save_csv(directory / "windows.csv", windows, WINDOW_FIELDS)
    analysis["aligned_windows"] = sum(w["host_samples"] > 0 for w in windows)
    analysis["inference_process_verified_samples"] = sum(
        str(s.get("inference_pid_verified", "")).lower() == "true" for s in samples)
    analysis["inference_process_unverified_samples"] = len(samples) - analysis["inference_process_verified_samples"]
    analysis["ffmpeg_scoped_to_publishers"] = bool(samples) and all(
        str(s.get("ffmpeg_scoped_to_publishers", "")).lower() == "true" for s in samples)
    if not analysis["inference_process_verified_samples"]:
        analysis["warnings"].append("No verified inference interpreter samples; do not interpret Python CPU or RSS.")
    analysis["windows_below_20fps"] = sum(w["aggregate_fps"] < 20 for w in windows)
    analysis["host_during_measurement"] = {
        "system_cpu_p50_pct": median(w["system_cpu_p50_pct"] for w in windows),
        "python_cpu_p50_one_core": median(w["python_cpu_p50_one_core"] for w in windows),
        "ffmpeg_cpu_p50_one_core": median(w["ffmpeg_cpu_p50_one_core"] for w in windows),
        "docker_cpu_p50_one_core": median(w["docker_cpu_p50_one_core"] for w in windows),
        "python_threads_p50": median(w["python_threads_p50"] for w in windows),
        "python_rss_p50_mb": median(w["python_rss_p50_mb"] for w in windows),
    }
    keys = ("system_cpu_p50_pct", "python_cpu_p50_one_core", "ffmpeg_cpu_p50_one_core",
            "docker_cpu_p50_one_core", "python_threads_p50", "encoded_crops_p50")
    for target in ("detector_p50_ms", "boxmot_p50_ms"):
        analysis["pearson_window_correlations"][target] = {
            key: correlation((w[target] for w in windows), (w[key] for w in windows))
            for key in keys
        }
    if not windows:
        analysis["warnings"].append("No complete measurement windows; increase duration.")
    if any(w["host_samples"] == 0 for w in windows):
        analysis["warnings"].append("At least one window lacks host samples; correlations may be unreliable.")
    analysis["limitations"] = [
        "CPU sampling uses psutil; process CPU percentages are relative to ONE logical core and can exceed 100.",
        "System CPU percentages are averaged over all logical cores (0-100).",
        "FFmpeg totals include all host FFmpeg processes unless explicit publisher PIDs were supplied.",
        "Python process attribution requires an interpreter command line and credible process identity; unverified samples are null.",
        "Docker/WSL process CPU is not a complete attribution of container resources.",
        "Windows psutil.cpu_freq is usually nominal and cannot verify throttling or clock recovery.",
        "Host samples and per-batch timings are aligned by system clock, whose adjustment can affect alignment.",
        "Per-batch nearest host samples are not exact process attribution; use 10-second window comparisons.",
        "Correlation across windows is descriptive and cannot establish causation.",
        "Different RTSP runs do not contain matched camera frames.",
    ]
    save_json(directory / "analysis.json", analysis)
    return analysis


def run(args):
    root = Path(args.root).resolve()
    directory = root / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "src.live_rtsp", "--seconds", str(args.seconds),
               "--warmup-batches", str(args.warmup_batches), "--threads", str(args.threads),
               "--rtsp-base", args.rtsp_base, "--profile-reid", "--output-dir", str(directory)]
    if args.mode == "no_reid":
        command.append("--no-reid")
    print("Starting monitored RTSP run:", args.mode, flush=True)
    print("Host sampling every", args.sample_interval, "seconds", flush=True)
    samples = []
    returncode = None
    child = None
    with (directory / "console.log").open("w", encoding="utf-8") as console:
        child = subprocess.Popen(command, cwd=str(ROOT), stdout=console, stderr=subprocess.STDOUT)
        try:
            launcher = psutil.Process(child.pid)
            target_cache = {}
            target, verified = resolve_inference_process(launcher, target_cache)
            cache = prepare_host_monitor(target)
            target_pid = target.pid if target is not None else None
            publishers = set(getattr(args, "ffmpeg_pids", None) or []) or None
            while child.poll() is None:
                time.sleep(args.sample_interval)
                target, verified = resolve_inference_process(launcher, target_cache)
                if target is not None and target.pid != target_pid:
                    target.cpu_percent(interval=None)
                    target_pid = target.pid
                samples.append(read_host_sample(target, cache, launcher_pid=child.pid,
                                verified=verified, ffmpeg_pids=publishers))
            returncode = child.wait()
        except KeyboardInterrupt:
            print("Interrupted. Saving partial host and pipeline measurements.", flush=True)
            child.terminate()
            try:
                returncode = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                returncode = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    save_csv(directory / "host_samples.csv", samples, FIELDS)
    result = analyse(directory, window_s=args.window_s,
                     sample_interval=args.sample_interval, returncode=returncode)
    print("Status:", result["run_status"], "| FPS:", result["aggregate_fps"],
          "| windows aligned:", result["aligned_windows"], flush=True)
    print("Saved:", directory / "analysis.json", flush=True)
    if result["warnings"]:
        print("Warnings:", *result["warnings"], sep="\n - ", flush=True)
    return 0 if returncode == 0 and result["complete"] and not result["warnings"] else 1


def compare(root):
    root = Path(root)
    records = {}
    for mode in ("reid", "no_reid"):
        obj = load_json(root / mode / "analysis.json")
        if obj is None:
            raise FileNotFoundError(f"Missing {root / mode / 'analysis.json'}")
        records[mode] = {
            "complete": obj.get("complete"),
            "aggregate_fps": obj.get("aggregate_fps"),
            "batch_p95_ms": obj.get("batch_p95_ms"),
            "aligned_windows": obj.get("aligned_windows"),
            "host": obj.get("host_during_measurement"),
            "correlations": obj.get("pearson_window_correlations"),
            "warnings": obj.get("warnings", []),
        }
    out = {
        "runs": records,
        "limitations": "Non-synchronised RTSP content. Different camera activity, host load, "
                       "CPU power state or source publishing can confound comparisons. "
                       "No CPU throttling or root-cause claim without independent evidence.",
    }
    save_json(root / "comparison.json", out)
    print("Saved", root / "comparison.json")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run", help="Monitor a four-camera live_rtsp subprocess")
    runner.add_argument("--mode", choices=("reid", "no_reid"), default="reid")
    runner.add_argument("--seconds", type=float, default=120.0)
    runner.add_argument("--warmup-batches", type=int, default=5)
    runner.add_argument("--threads", type=int, default=4)
    runner.add_argument("--sample-interval", type=float, default=0.5)
    runner.add_argument("--window-s", type=float, default=10.0)
    runner.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    runner.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    runner.add_argument("--ffmpeg-pids", type=int, nargs="*", default=None,
                        help="Optional exact FFmpeg publisher PIDs to monitor; default all FFmpeg processes")
    comparator = sub.add_parser("compare", help="Combine both monitored analyses")
    comparator.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.action == "run":
        if args.seconds <= 0 or args.threads < 1 or args.warmup_batches < 0 or not (0.2 <= args.sample_interval <= 10.0) or args.window_s <= 0:
            parser.error("seconds/threads/window-s must be positive; sampling interval 0.2..10 s; warmup >= 0")
        raise SystemExit(run(args))
    compare(args.root)


if __name__ == "__main__":
    main()
