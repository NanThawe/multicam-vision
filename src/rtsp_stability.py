"""Step 14.8: repeated RTSP A/B stability trials and independent MQTT smoke test.

Usage (project root):
    python -m src.rtsp_stability run --seconds 120 --repeats 3
    python -m src.rtsp_stability report --root outputs/rtsp/step14_8
    python -m src.rtsp_stability mqtt-check

Runs the existing live_rtsp as separate subprocesses, keeping publishers
external to the experiment. No YOLO, ByteTrack or ReID settings change.
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .framesource import ROOT

DEFAULT_DIR = ROOT / "outputs" / "rtsp" / "step14_8"
CAMERAS = ("C1", "C2", "C3", "C4")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _median(values):
    return round(statistics.median(values), 3) if values else None


def _load_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ten_second_windows(rows, window_s=10.0):
    """Per-complete-window throughput from actual measured elapsed_s, not frame IDs.

    The first interval is (0, window_s], following intervals are left-open.
    Ignore a partial final window to avoid overstating/understating throughput.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    elapsed = [float(row["elapsed_s"]) for row in rows if row.get("elapsed_s")]
    if not elapsed:
        return []
    completed_windows = math.floor(max(elapsed) / window_s)
    out = []
    for index in range(completed_windows):
        start, finish = index * window_s, (index + 1) * window_s
        in_window = [r for r in rows if r.get("elapsed_s") and
                     start < float(r["elapsed_s"]) <= finish]
        if not in_window:
            fps = 0.0
            p95_ms = None
        else:
            fps = 4 * len(in_window) / window_s
            p95_ms = _percentile([float(r["processing_ms"]) for r in in_window], .95)
        out.append({"start_s": round(start, 2), "end_s": round(finish, 2),
                    "batches": len(in_window), "aggregate_fps": round(fps, 3),
                    "batch_p95_ms": p95_ms})
    return out


def _percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    place = (len(values)-1)*p
    a = math.floor(place)
    b = math.ceil(place)
    return round(values[a] + (values[b]-values[a])*(place-a), 3)


def inspect_run(directory, *, minimum_fps=20.0, window_s=10.0):
    """Read independent run artifacts and derive conservative stability checks."""
    directory = Path(directory)
    report = _read_json(directory / "report.json")
    status = _read_json(directory / "run_status.json") or {}
    row = {
        "run": directory.name,
        "run_status": status.get("status", "missing"),
        "complete": bool(report and report.get("complete") and status.get("complete")),
        "aggregate_fps": None,
        "p50_processing_ms": None,
        "p95_processing_ms": None,
        "total_overwrites": None,
        "decoded_received": None,
        "overwrite_per_received": None,
        "worst_complete_window_fps": None,
        "windows_below_target": None,
        "meets_run_target": False,
        "stage_timings_ms": {},
        "mqtt_enabled": None,
        "mqtt_qos1_acknowledged": None,
        "error": status.get("error"),
    }
    windows = ten_second_windows(_load_rows(directory / "per_batch.csv"), window_s)
    if report is None:
        return row, windows
    row["aggregate_fps"] = report.get("aggregate_processed_fps")
    row["p50_processing_ms"] = report.get("processing_ms_per_four_camera_batch", {}).get("p50")
    row["p95_processing_ms"] = report.get("processing_ms_per_four_camera_batch", {}).get("p95")
    cameras = report.get("per_camera", {})
    row["total_overwrites"] = sum(c.get("application_buffer_overwrites", 0)
                                  for c in cameras.values())
    row["decoded_received"] = sum(c.get("decoded_frames_received", 0)
                                   for c in cameras.values())
    if row["decoded_received"]:
        row["overwrite_per_received"] = round(row["total_overwrites"] / row["decoded_received"], 5)
    row["stage_timings_ms"] = report.get("stage_timings_ms", {})
    row["mqtt_enabled"] = report.get("mqtt_enabled")
    row["mqtt_qos1_acknowledged"] = report.get("mqtt_qos1_acknowledged")
    row["error"] = report.get("error") or row["error"]
    if windows:
        row["worst_complete_window_fps"] = min(w["aggregate_fps"] for w in windows)
        row["windows_below_target"] = sum(w["aggregate_fps"] < minimum_fps for w in windows)
    # A completed run meeting average FPS is a short-run throughput pass;
    # sustained-window performance and overwrites are independent diagnostics.
    row["meets_run_target"] = bool(row["complete"] and
                                   row["aggregate_fps"] is not None and
                                   row["aggregate_fps"] >= minimum_fps)
    return row, windows


def compare(runs):
    """Do not compare unmatched sequences as a frame-aligned latency experiment."""
    groups = {"baseline": [], "overlay": []}
    for r in runs:
        mode = r["mode"]
        if mode not in groups:
            continue
        if r["complete"] and r["aggregate_fps"] is not None:
            groups[mode].append(r)
    result = {}
    for mode, records in groups.items():
        result[mode] = {
            "completed_runs": len(records),
            "median_aggregate_fps": _median([r["aggregate_fps"] for r in records]),
            "median_batch_p95_ms": _median([r["p95_processing_ms"] for r in records
                                             if r["p95_processing_ms"] is not None]),
            "median_overwrites": _median([r["total_overwrites"] for r in records]),
            "runs_meeting_20fps": sum(r["meets_run_target"] for r in records),
            "runs_with_window_below_20fps": sum((r["windows_below_target"] or 0) > 0
                                                for r in records),
        }
    base = result["baseline"]["median_aggregate_fps"]
    overlay = result["overlay"]["median_aggregate_fps"]
    result["overlay_minus_baseline_median_fps"] = (round(overlay-base, 3)
                                                   if base is not None and overlay is not None
                                                   else None)
    result["limitations"] = [
        "Different trials may contain different people and camera frames.",
        "Four cameras share CPU with RTSP publishing and MP4 encoding.",
        "FPS measures decoded-frame processing, not camera-to-display latency.",
        "Overwrites count only drops in the Python latest-frame buffer.",
        "Per-stage percentiles do not add to total-batch percentiles.",
        "QoS1 broker acknowledgement does not establish subscriber receipt.",
    ]
    return result


def write_csv(path, data, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def save_analysis(root, schedule, *, minimum_fps=20.0, window_s=10.0):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    runs, windows_out = [], []
    for slot in schedule:
        row, windows = inspect_run(root / slot["run"], minimum_fps=minimum_fps,
                                   window_s=window_s)
        row["mode"] = slot["mode"]
        row["returncode"] = slot.get("returncode")
        runs.append(row)
        for w in windows:
            windows_out.append({"run": slot["run"], "mode": slot["mode"], **w})
    metrics = compare(runs)
    summary = {"runs": runs, "comparison": metrics, "minimum_fps": minimum_fps,
               "window_s": window_s, "schedule": schedule}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ["run", "mode", "complete", "run_status", "returncode", "aggregate_fps",
              "p50_processing_ms", "p95_processing_ms", "total_overwrites",
              "decoded_received", "overwrite_per_received", "worst_complete_window_fps",
              "windows_below_target", "meets_run_target", "mqtt_enabled",
              "mqtt_qos1_acknowledged", "error"]
    write_csv(root / "per_run.csv", runs, fields)
    write_csv(root / "time_windows.csv", windows_out,
              ["run", "mode", "start_s", "end_s", "batches", "aggregate_fps", "batch_p95_ms"])
    return summary


def _existing_schedule(root):
    manifest = _read_json(Path(root) / "manifest.json")
    if manifest and isinstance(manifest.get("schedule"), list):
        return manifest["schedule"]
    candidates = sorted(d for d in Path(root).iterdir() if d.is_dir() and
                        (d.name.startswith("baseline_") or d.name.startswith("overlay_")))
    return [{"run": d.name, "mode": d.name.split("_")[0]} for d in candidates]


def run_trials(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already contains a trial manifest: {manifest_path}. Use a new --root to prevent overwritten evidence.")
    schedule = []
    for repetition in range(args.repeats):
        modes = ("baseline", "overlay") if repetition % 2 == 0 else ("overlay", "baseline")
        for mode in modes:
            schedule.append({"run": f"{mode}_{repetition+1:02d}", "mode": mode,
                             "repetition": repetition+1, "returncode": None})
    manifest = {"schedule": schedule, "seconds": args.seconds,
                "repeats": args.repeats, "threads": args.threads,
                "warmup_batches": args.warmup_batches,
                "rtsp_base": args.rtsp_base,
                "note": "Publisher must remain running outside this runner. Do not parallelize trials."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for slot in schedule:
        folder = root / slot["run"]
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "src.live_rtsp", "--seconds", str(args.seconds),
                   "--warmup-batches", str(args.warmup_batches), "--threads", str(args.threads),
                   "--rtsp-base", args.rtsp_base, "--profile-stages",
                   "--output-dir", str(folder)]
        if slot["mode"] == "overlay":
            command.append("--provisional-preview")
        (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        print(f"Running {slot['run']} ...", flush=True)
        began = time.time()
        with (folder / "console.log").open("w", encoding="utf-8") as output:
            try:
                process = subprocess.run(command, cwd=str(ROOT), stdout=output,
                                         stderr=subprocess.STDOUT, check=False,
                                         timeout=max(args.seconds * 3 + 120, 180))
                slot["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                slot["returncode"] = -1
                output.write("\nTimed out in stability runner.\n")
        slot["elapsed_wall_s"] = round(time.time() - began, 2)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        save_analysis(root, schedule, minimum_fps=args.minimum_fps,
                      window_s=args.window_s)
        print(f"{slot['run']}: exit code {slot['returncode']}; "+
              f"see {folder / 'report.json'} and {folder / 'console.log'}", flush=True)
    print(f"Saved {root / 'summary.json'}", flush=True)


def mqtt_smoke(host, port, timeout=6.0):
    """Test actual subscriber delivery and publish QoS1 broker acknowledgment."""
    from .mqtt_events import mqtt_client
    mqtt, subscriber = mqtt_client()
    _, publisher = mqtt_client()
    nonce = uuid.uuid4().hex
    topic = "multicam/diagnostics/step14_8/" + nonce
    ready = threading.Event()
    subscribed = threading.Event()
    delivered = threading.Event()
    connected_publisher = threading.Event()
    errors = []

    def on_sub_connect(client, _userdata, _flags, reason, _properties):
        if reason.is_failure:
            errors.append(f"Subscriber refused: {reason}")
        else:
            ready.set()
            client.subscribe(topic, qos=1)

    def on_subscribe(_client, _userdata, _mid, reasons, _properties):
        if any(getattr(x, "is_failure", False) for x in reasons):
            errors.append("Subscribe refused")
        else:
            subscribed.set()

    def on_message(_client, _userdata, message):
        if message.topic == topic and message.payload.decode("utf-8") == nonce:
            delivered.set()

    def on_pub_connect(_client, _userdata, _flags, reason, _properties):
        if reason.is_failure:
            errors.append(f"Publisher refused: {reason}")
        else:
            connected_publisher.set()

    subscriber.on_connect = on_sub_connect
    subscriber.on_subscribe = on_subscribe
    subscriber.on_message = on_message
    publisher.on_connect = on_pub_connect
    sub_started = pub_started = False
    try:
        subscriber.connect(host, port, keepalive=30)
        subscriber.loop_start()
        sub_started = True
        if not ready.wait(timeout) or not subscribed.wait(timeout):
            raise TimeoutError("Subscriber connection/subscription not confirmed")
        publisher.connect(host, port, keepalive=30)
        publisher.loop_start()
        pub_started = True
        if not connected_publisher.wait(timeout):
            raise TimeoutError("Publisher connection not confirmed")
        info = publisher.publish(topic, nonce, qos=1, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"Publish failed with code {info.rc}")
        info.wait_for_publish(timeout=timeout)
        pubacked = bool(info.is_published())
        received = delivered.wait(timeout)
        if errors or not pubacked or not received:
            raise ConnectionError(f"Broker ack: {pubacked}; subscriber receive: {received}; errors: {errors}")
        return {"mqtt_connected": True, "qos1_broker_ack": True,
                "subscriber_delivered": True, "topic": topic}
    finally:
        for client, started in ((publisher, pub_started), (subscriber, sub_started)):
            if started:
                client.disconnect()
                client.loop_stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Alternating repeated baseline and overlay RTSP trials")
    run.add_argument("--root", type=Path, default=DEFAULT_DIR)
    run.add_argument("--seconds", type=float, default=120.0)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--warmup-batches", type=int, default=5)
    run.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    run.add_argument("--minimum-fps", type=float, default=20.0)
    run.add_argument("--window-s", type=float, default=10.0)
    report = sub.add_parser("report", help="Rebuild summary from existing trial folders")
    report.add_argument("--root", type=Path, default=DEFAULT_DIR)
    report.add_argument("--minimum-fps", type=float, default=20.0)
    report.add_argument("--window-s", type=float, default=10.0)
    mqtt = sub.add_parser("mqtt-check", help="Independent MQTT QoS1 + subscriber smoke test")
    mqtt.add_argument("--host", default="127.0.0.1")
    mqtt.add_argument("--port", type=int, default=1883)
    mqtt.add_argument("--timeout", type=float, default=6.0)
    mqtt.add_argument("--output", type=Path, default=DEFAULT_DIR / "mqtt_smoke.json")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.seconds <= 0 or args.repeats < 1 or args.threads < 1 or args.warmup_batches < 0 or args.window_s <= 0:
            parser.error("Invalid trial duration, repetitions, threads, warmup or window size")
        run_trials(args)
    elif args.command == "report":
        if args.window_s <= 0:
            parser.error("--window-s must be positive")
        summary = save_analysis(args.root, _existing_schedule(args.root),
                                minimum_fps=args.minimum_fps, window_s=args.window_s)
        print(json.dumps(summary["comparison"], indent=2))
    else:
        if args.timeout <= 0 or not 1 <= args.port <= 65535:
            parser.error("Invalid MQTT timeout/port")
        try:
            result = mqtt_smoke(args.host, args.port, args.timeout)
            result["status"] = "pass"
        except Exception as exc:
            result = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        if result["status"] != "pass":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
