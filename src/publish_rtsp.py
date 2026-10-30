"""Replay four MP4 recordings as paced RTSP streams using FFmpeg.

Run after: docker compose -f compose.rtsp.yaml up -d
    python -m src.publish_rtsp --width 960 --fps 5
Requires FFmpeg on PATH. Press Ctrl+C to stop all publishers.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from .framesource import ROOT, VIDEOS

LOG_DIR = ROOT / "outputs" / "rtsp" / "publishers"


def ffmpeg_command(input_path, camera, *, fps=5.0, width=960,
                   rtsp_base="rtsp://127.0.0.1:8554", binary="ffmpeg"):
    """Return a real-time, looping H.264 publisher command for one camera."""
    if fps <= 0 or width < 160 or width % 2:
        raise ValueError("fps must be positive; width must be even and >= 160")
    return [
        binary, "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-re", "-stream_loop", "-1", "-i", str(input_path),
        "-an", "-vf", f"fps={fps:g},scale={width}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps * 2))),
        "-f", "rtsp", "-rtsp_transport", "tcp", "-muxdelay", "0",
        f"{rtsp_base.rstrip('/')}/{camera}",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Published FPS per camera (default 5)")
    parser.add_argument("--width", type=int, default=960,
                        help="Published video width in pixels, even (default 960)")
    parser.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg not found on PATH. Install FFmpeg and check: ffmpeg -version")
    missing = [str(path) for path in VIDEOS.values() if not Path(path).is_file()]
    if missing:
        parser.error("Missing camera recording(s): " + ", ".join(missing))
    try:
        # Validate all settings before launching any process.
        commands = {camera: ffmpeg_command(path, camera, fps=args.fps,
                    width=args.width, rtsp_base=args.rtsp_base, binary=ffmpeg)
                    for camera, path in VIDEOS.items()}
    except ValueError as exc:
        parser.error(str(exc))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = {}
    logs = []
    try:
        for camera, cmd in commands.items():
            log_path = LOG_DIR / f"{camera}.log"
            handle = log_path.open("w", encoding="utf-8")
            logs.append(handle)
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            processes[camera] = proc
            print(f"{camera}: {args.rtsp_base.rstrip('/')}/{camera} "
                  f"(pid={proc.pid}, log={log_path})", flush=True)
        print("RTSP streams starting. Run src.live_rtsp in another terminal. Ctrl+C stops publishers.")
        while True:
            time.sleep(0.5)
            for camera, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{camera} FFmpeg exited ({process.returncode}); "
                        f"inspect {LOG_DIR / (camera + '.log')}"
                    )
    except KeyboardInterrupt:
        print("Stopping RTSP publishers...")
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()
