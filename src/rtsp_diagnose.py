"""Check four RTSP feeds without loading YOLO or OSNet.

Run with MediaMTX and all FFmpeg publishers already running:
    python -m src.rtsp_diagnose --seconds 8
"""

import argparse
import json
import os
import time
from pathlib import Path

from .framesource import ROOT
from .live_rtsp import CAMERAS, RTSPReader


OUTPUT = ROOT / "outputs" / "rtsp" / "diagnostic.json"


def probe(readers, seconds=8.0, sleep=time.sleep, clock=time.perf_counter):
    """Return each camera's decoded count and reader state over a short window."""
    # Wait for one decoded frame before opening the next camera.
    # Some Windows OpenCV/FFmpeg builds serialize RTSP VideoCapture.open().
    for reader in readers.values():
        try:
            print(f"Connecting {reader.camera}: {reader.url}", flush=True)
            reader.start(timeout=12.0)
            print(f"{reader.camera}: first frame decoded", flush=True)
        except Exception as exc:
            print(f"{reader.camera}: startup failed: {exc}", flush=True)
    started = clock()
    try:
        while clock() - started < seconds:
            sleep(0.1)
    finally:
        snapshot = {}
        now = clock()
        for camera, reader in readers.items():
            received, overwritten = reader.slot.counters()
            latest = reader.slot.latest
            snapshot[camera] = {
                "url": reader.url,
                "decoded_frames": received,
                "latest_frame_age_s": round(now - latest.received_at, 2)
                if latest is not None else None,
                "reader_running": bool(reader.thread and reader.thread.is_alive()),
                "error": str(reader.error) if reader.error is not None else None,
            }
        for reader in readers.values():
            reader.stop()
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    readers = {camera: RTSPReader(camera, f"{args.rtsp_base.rstrip('/')}/{camera}")
               for camera in CAMERAS}
    print("Probing four RTSP feeds; no AI models are loaded...", flush=True)
    report = probe(readers, seconds=args.seconds)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for camera, info in report.items():
        print(f"{camera}: decoded={info['decoded_frames']}, "
              f"reader_running={info['reader_running']}, "
              f"last_frame_age_s={info['latest_frame_age_s']}, "
              f"error={info['error']}")
    print(f"Saved {OUTPUT}")
    if any(info["decoded_frames"] == 0 for info in report.values()):
        print("At least one feed delivered no decoded frames. "
              "Check its publisher log and MediaMTX container logs.")
        return 1
    print("All cameras delivered decoded frames. Retry the short live benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
