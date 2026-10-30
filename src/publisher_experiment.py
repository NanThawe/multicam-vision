"""Step 14.12: matched-format RTSP publisher CPU experiment.

First encode each source once, then run a sequential benchmark of:
  live: legacy decode/scale/encode FFmpeg publisher
  copy: replay the prepared H.264 MP4 with -c:v copy

Usage from project root (MediaMTX must already be running):
  python -m src.publisher_experiment prepare
  python -m src.publisher_experiment run --mode live --seconds 120
  python -m src.publisher_experiment run --mode copy --seconds 120
  python -m src.publisher_experiment compare
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from .framesource import ROOT, VIDEOS
from .publish_rtsp import ffmpeg_command
from . import cpu_contention

DEFAULT_ROOT = ROOT / "outputs" / "rtsp" / "step14_12"
CAMERAS = tuple(VIDEOS.keys())


def require_binaries(*names):
    found = {name: shutil.which(name) for name in names}
    missing = [name for name, value in found.items() if value is None]
    if missing:
        raise RuntimeError("Executable(s) not on PATH: " + ", ".join(missing))
    return found


def prepared_video(root, camera):
    return Path(root) / "prepared" / f"{camera}_5fps_960w.mp4"


def preparation_command(source, output, *, fps=5.0, width=960, ffmpeg="ffmpeg"):
    if fps <= 0 or width < 160 or width % 2:
        raise ValueError("fps must be positive; width must be even and >=160")
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "warning", "-y",
            "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", f"fps={fps:g},scale={width}:-2",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps * 2))),
            "-movflags", "+faststart", str(output)]


def copy_command(prepared, camera, *, rtsp_base="rtsp://127.0.0.1:8554", ffmpeg="ffmpeg"):
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "warning", "-re",
            "-stream_loop", "-1", "-i", str(prepared), "-an", "-c:v", "copy",
            "-f", "rtsp", "-rtsp_transport", "tcp", "-muxdelay", "0",
            f"{rtsp_base.rstrip('/')}/{camera}"]


def probe_video(path, *, ffprobe="ffprobe", timeout=20):
    cmd = [ffprobe, "-v", "error"]
    if str(path).startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += ["-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
           "-of", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"No video stream in {path}")
    return streams[0]


def rate(value):
    if not value or value == "0/0":
        return None
    if "/" in str(value):
        a, b = str(value).split("/", 1)
        return float(a) / float(b) if float(b) else None
    return float(value)


def check_video(profile, fps, width):
    return (profile.get("codec_name") == "h264"
            and int(profile.get("width", -1)) == width
            and (actual := rate(profile.get("avg_frame_rate"))) is not None
            and abs(actual - fps) < 0.02)


def prepare(root=DEFAULT_ROOT, *, fps=5, width=960, force=False, sources=VIDEOS,
            ffmpeg=None, ffprobe=None):
    binary = require_binaries("ffmpeg", "ffprobe") if ffmpeg is None or ffprobe is None else {}
    ffmpeg = ffmpeg or binary["ffmpeg"]
    ffprobe = ffprobe or binary["ffprobe"]
    prepared = Path(root) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    previous = cpu_contention.load_json(prepared / "manifest.json") or {}
    manifest = {"fps":fps, "width":width, "cameras":{}}
    for camera, source in sources.items():
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"Missing source video for {camera}: {source}")
        output = prepared / f"{camera}_5fps_960w.mp4"
        signature = {"source":str(source.resolve()), "source_bytes":source.stat().st_size,
                     "source_mtime_ns":source.stat().st_mtime_ns,
                     "fps":fps, "width":width}
        prior = (previous.get("cameras") or {}).get(camera) or {}
        reuse = (not force and output.is_file() and prior.get("source_signature") == signature)
        if reuse:
            profile = probe_video(output, ffprobe=ffprobe)
            reuse = check_video(profile, fps, width)
        if not reuse:
            print(f"Preparing {camera} -> {output} (offline encode)", flush=True)
            command = preparation_command(source, output, fps=fps, width=width, ffmpeg=ffmpeg)
            with (prepared / f"prepare_{camera}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                output.unlink(missing_ok=True)
                raise RuntimeError(f"{camera} offline encode failed; inspect {prepared / f'prepare_{camera}.log'}")
            profile = probe_video(output, ffprobe=ffprobe)
            if not check_video(profile, fps, width):
                raise RuntimeError(f"Prepared video has wrong codec/resolution/FPS: {camera}: {profile}")
        else:
            print(f"Reusing prepared {camera}: {output}", flush=True)
        manifest["cameras"][camera] = {"source_signature":signature, "prepared":str(output),
                                         "profile":profile, "prepared_bytes":output.stat().st_size}
        # Save incrementally, but never reuse if the manifest fails to match the source/settings.
        cpu_contention.save_json(prepared / "manifest.json", manifest)
    return manifest


def verified_prepared(root, *, fps=5, width=960, sources=VIDEOS, ffprobe="ffprobe"):
    manifest = cpu_contention.load_json(Path(root) / "prepared" / "manifest.json") or {}
    if manifest.get("fps") != fps or manifest.get("width") != width:
        raise RuntimeError("Prepared manifest uses different width/FPS; rerun prepare")
    for camera, source in sources.items():
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        p = prepared_video(root, camera)
        sig = {"source":str(source.resolve()), "source_bytes":source.stat().st_size,
               "source_mtime_ns":source.stat().st_mtime_ns, "fps":fps, "width":width}
        old = (manifest.get("cameras") or {}).get(camera) or {}
        if not p.is_file() or old.get("source_signature") != sig:
            raise RuntimeError(f"Missing/stale prepared {camera}. Run: python -m src.publisher_experiment prepare")
        if not check_video(probe_video(p, ffprobe=ffprobe), fps, width):
            raise RuntimeError(f"Invalid codec/FPS/width in {p}; rerun prepare")
    return manifest


def publisher_commands(mode, root, sources=VIDEOS, *, rtsp_base, ffmpeg="ffmpeg", fps=5, width=960):
    if mode not in ("live", "copy"):
        raise ValueError("mode must be live or copy")
    return {
        camera: ffmpeg_command(source, camera, fps=fps, width=width,
                               rtsp_base=rtsp_base, binary=ffmpeg) if mode == "live"
        else copy_command(prepared_video(root, camera), camera, rtsp_base=rtsp_base, ffmpeg=ffmpeg)
        for camera, source in sources.items()
    }


def start_publishers(commands, logs):
    """Launch each publisher independently, preserving detailed error logs."""
    procs, handles = {}, []
    try:
        for cam, cmd in commands.items():
            path = Path(logs) / f"publisher_{cam}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w", encoding="utf-8")
            handles.append(handle)
            procs[cam] = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        return procs, handles
    except Exception:
        stop_publishers(procs, handles)
        raise


def stop_publishers(procs, handles):
    for proc in procs.values():
        if proc.poll() is None:
            proc.terminate()
    for proc in procs.values():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for handle in handles:
        handle.close()


def verify_live_endpoints(procs, rtsp_base, *, ffprobe, fps=5, width=960,
                          attempts=6, delay=1.5):
    for camera, proc in procs.items():
        url = f"{rtsp_base.rstrip('/')}/{camera}"
        error = None
        for _ in range(attempts):
            if proc.poll() is not None:
                raise RuntimeError(f"Publisher {camera} exited: code={proc.returncode}; inspect publisher log")
            try:
                profile = probe_video(url, ffprobe=ffprobe, timeout=8)
                # On some RTSP demuxers avg_frame_rate is 0/0 even when the
                # original 5 FPS encoded file is correct. Verify the stream
                # is H.264 and 960 px; the publisher command supplies FPS.
                if profile.get("codec_name") == "h264" and int(profile.get("width", -1)) == width:
                    reported_fps = rate(profile.get("avg_frame_rate"))
                    if reported_fps is None:
                        print(f"{camera}: live RTSP FPS metadata unavailable; source command is configured for {fps} FPS", flush=True)
                    elif abs(reported_fps - fps) >= 0.1:
                        print(f"{camera}: RTSP reports {reported_fps:g} FPS (requested {fps:g}); inspect publisher metadata", flush=True)
                    break
                error = f"Stream codec/width mismatch: {camera}: {profile}"
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                error = str(exc)
            time.sleep(delay)
        else:
            raise RuntimeError(f"RTSP stream {camera} not ready/incorrect: {error}")


def run_trial(args, *, sources=VIDEOS):
    bins = require_binaries("ffmpeg", "ffprobe")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "copy":
        verified_prepared(root, fps=args.fps, width=args.width, sources=sources,
                          ffprobe=bins["ffprobe"])
    for camera, path in sources.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {camera}: {path}")
    commands = publisher_commands(args.mode, root, sources, rtsp_base=args.rtsp_base,
                                  ffmpeg=bins["ffmpeg"], fps=args.fps, width=args.width)
    output = root / args.mode
    output.mkdir(parents=True, exist_ok=True)
    procs, handles = start_publishers(commands, output)
    state = {"mode":args.mode,"publisher_pids":{c:p.pid for c,p in procs.items()},
             "fps":args.fps,"width":args.width,"rtsp_base":args.rtsp_base,
             "requested_seconds":args.seconds,"returncode":None,
             "completed":False,"error":None}
    cpu_contention.save_json(output / "publisher_trial.json", state)
    try:
        verify_live_endpoints(procs, args.rtsp_base, ffprobe=bins["ffprobe"],
                              fps=args.fps, width=args.width)
        print(f"Four {args.mode} publishers ready. Starting monitored ReID trial.", flush=True)
        monitor_args = SimpleNamespace(root=output, mode="reid", seconds=args.seconds,
                                       warmup_batches=args.warmup_batches, threads=args.threads,
                                       sample_interval=args.sample_interval, window_s=10,
                                       rtsp_base=args.rtsp_base,
                                       ffmpeg_pids=[p.pid for p in procs.values()])
        code = cpu_contention.run(monitor_args)
        state["returncode"] = code
        state["completed"] = code == 0
        return code
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_publishers(procs, handles)
        state["publisher_exit_codes"] = {c:p.returncode for c,p in procs.items()}
        cpu_contention.save_json(output / "publisher_trial.json", state)


def compare(root=DEFAULT_ROOT):
    root = Path(root)
    records = {}
    for mode in ("live", "copy"):
        analysis = cpu_contention.load_json(root / mode / "reid" / "analysis.json")
        trial = cpu_contention.load_json(root / mode / "publisher_trial.json")
        if not analysis or not trial:
            raise FileNotFoundError(f"Missing {mode} trial outputs; run both modes first")
        host = analysis.get("host_during_measurement") or {}
        if not analysis.get("aligned_windows"):
            raise RuntimeError(f"{mode} did not produce aligned windows")
        records[mode] = {
            "run_complete":bool(trial.get("completed")) and analysis.get("run_status") == "completed",
            "aggregate_fps":analysis.get("aggregate_fps"),
            "batch_p95_ms":analysis.get("batch_p95_ms"),
            "aligned_windows":analysis.get("aligned_windows"),
            "windows_below_20fps":analysis.get("windows_below_20fps"),
            "publisher_ffmpeg_cpu_p50_one_core":host.get("ffmpeg_cpu_p50_one_core"),
            "inference_python_cpu_p50_one_core":host.get("python_cpu_p50_one_core"),
            "inference_python_rss_mb":host.get("python_rss_p50_mb"),
            "verified_inference_samples":analysis.get("inference_process_verified_samples"),
            "ffmpeg_scoped_to_publishers":analysis.get("ffmpeg_scoped_to_publishers"),
            "measurement_warnings":analysis.get("warnings", []),
        }
    a, b = records["live"], records["copy"]
    result = {
        "purpose":"Compare live transcoding publishers vs prepared H.264 stream copy",
        "runs":records,
        "delta_copy_minus_live_fps":round(b["aggregate_fps"]-a["aggregate_fps"],3)
            if a["aggregate_fps"] is not None and b["aggregate_fps"] is not None else None,
        "delta_copy_minus_live_publisher_cpu_one_core":round(
            b["publisher_ffmpeg_cpu_p50_one_core"]-a["publisher_ffmpeg_cpu_p50_one_core"],3)
            if a["publisher_ffmpeg_cpu_p50_one_core"] is not None and b["publisher_ffmpeg_cpu_p50_one_core"] is not None else None,
        "interpretation":"Compare host windows, repeated trials, and model timing before inferring causality.",
        "limitations":[
            "Modes replay the same four original recordings, but frames and timestamps are not perfectly synchronized.",
            "Re-encoding offline can alter compression or source looping behavior; compare against source profiles.",
            "Both cases also decode RTSP inside inference; only publishing-side decode/filter/encode is removed.",
            "Live publisher CPU is measured from exact FFmpeg PIDs, not all system FFmpeg processes.",
            "System/CPU load, Windows power state and model workload can vary between sequential trials.",
            "No official accuracy or camera-to-display latency is computed.",
        ],
    }
    cpu_contention.save_json(root / "comparison.json", result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prep = subs.add_parser("prepare", help="Offline encode all sources into 5 FPS H.264 MP4s")
    prep.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prep.add_argument("--force", action="store_true")
    runner = subs.add_parser("run", help="Start publishers, monitor ReID inference, stop publishers")
    runner.add_argument("--mode", choices=("live","copy"), required=True)
    runner.add_argument("--seconds", type=float, default=120)
    runner.add_argument("--threads", type=int, default=4)
    runner.add_argument("--warmup-batches", type=int, default=5)
    runner.add_argument("--sample-interval", type=float, default=.5)
    runner.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    runner.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    comp = subs.add_parser("compare", help="Compare completed publisher mode trials")
    comp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.root, force=args.force)
    elif args.action == "run":
        if args.seconds <= 0 or args.threads < 1 or args.warmup_batches < 0 or not .2 <= args.sample_interval <= 10:
            parser.error("seconds/threads must be positive, warmup >=0, sampling interval 0.2..10")
        # Fixed parameters intentionally match the existing src.publish_rtsp defaults.
        args.fps = 5.0
        args.width = 960
        raise SystemExit(run_trial(args))
    else:
        compare(args.root)


if __name__ == "__main__":
    main()
