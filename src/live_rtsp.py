"""Continuously process four RTSP feeds with one-slot, latest-frame buffers.

Example:
    python -m src.live_rtsp --seconds 60 --warmup-batches 5

Queue-drop counts cover decoded frames overwritten in this Python process;
they do not include camera/network/FFmpeg decoder loss. Frame age is measured
from Python decode receipt, NOT from camera capture time.
"""

import argparse
import csv
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .framesource import ROOT

CAMERAS = ("C1", "C2", "C3", "C4")
OUTPUT_DIR = ROOT / "outputs" / "rtsp"


@dataclass(frozen=True)
class Sample:
    sequence: int
    received_at: float
    image: object


class LatestSlot:
    """A one-frame buffer. Overwritten frames are an application-level drop."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.received = 0
        self.overwritten = 0

    def put(self, image, timestamp):
        with self.lock:
            self.received += 1
            if self.latest is not None:
                self.overwritten += 1
            self.latest = Sample(self.received, timestamp, image)

    def has_frame(self):
        with self.lock:
            return self.latest is not None

    def take(self):
        with self.lock:
            sample = self.latest
            self.latest = None
            return sample

    def counters(self):
        with self.lock:
            return self.received, self.overwritten


class RTSPReader:
    """Decode a feed continuously on one background thread."""

    def __init__(self, camera, url, *, capture_factory=None):
        self.camera = camera
        self.url = url
        self.slot = LatestSlot()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.capture_factory = capture_factory
        self._first_frame = threading.Event()

    def start(self, *, timeout=None):
        """Start decoding; optionally wait for this camera's first frame.

        Calling start(timeout=...) in camera order serialises RTSP startup,
        avoiding overlapping VideoCapture.open() calls on Windows.
        """
        if self.thread is not None:
            raise RuntimeError(f"{self.camera}: reader already started")
        self.thread = threading.Thread(target=self._run, name=f"rtsp-{self.camera}",
                                       daemon=True)
        self.thread.start()
        if timeout is not None:
            if not self._first_frame.wait(timeout):
                error = TimeoutError(
                    f"{self.camera}: no first decoded frame after {timeout:g}s "
                    f"from {self.url}"
                )
                self.error = error
                self.stop()
                raise error
            if self.error is not None:
                self.stop()
                raise OSError(f"{self.camera}: {self.error}")

    def _run(self):
        capture = None
        try:
            if self.capture_factory is not None:
                capture = self.capture_factory(self.url)
            else:
                capture = cv2.VideoCapture(
                    self.url, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000],
                )
            if not capture.isOpened():
                raise OSError(f"Cannot open {self.url}; confirm FFmpeg publisher and MediaMTX are running")
            # Some OpenCV backends ignore this; actual application buffering is our LatestSlot.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop_event.is_set():
                ok, image = capture.read()
                if not ok:
                    raise OSError(f"Stream read stopped: {self.camera}. Restart after checking publisher logs")
                self.slot.put(image, time.perf_counter())
                self._first_frame.set()
        except Exception as exc:
            if not self.stop_event.is_set() and self.error is None:
                self.error = exc
            self._first_frame.set()
        finally:
            if capture is not None:
                capture.release()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)


def wait_for_batch(readers, *, timeout=10.0, clock=time.perf_counter,
                   sleep=time.sleep):
    """Retrieve one fresh decoded frame per camera or fail after timeout.

    The camera frames are not assumed hardware-synchronised. This function
    only checks that each camera supplies a new frame for this batch.
    """
    deadline = clock() + timeout
    while clock() < deadline:
        for reader in readers.values():
            if reader.error is not None:
                raise OSError(f"{reader.camera}: {reader.error}")
        if all(reader.slot.has_frame() for reader in readers.values()):
            batch = {camera: reader.slot.take() for camera, reader in readers.items()}
            if all(sample is not None for sample in batch.values()):
                return batch
        sleep(0.005)
    details = []
    for camera, reader in readers.items():
        received, _ = reader.slot.counters()
        has_fresh = reader.slot.has_frame()
        thread = getattr(reader, "thread", None)
        running = thread.is_alive() if thread is not None else "unknown"
        error = str(reader.error) if reader.error is not None else "none"
        details.append(
            f"{camera}: fresh={has_fresh}, decoded={received}, "
            f"reader_alive={running}, error={error}"
        )
    raise TimeoutError(
        "Timed out waiting for fresh frames from all RTSP cameras. "
        + " | ".join(details)
    )


def timeline_frame_id(now, origin, previous, nominal_source_fps=59.94):
    """Map increasing live elapsed time into EventEngine's source-frame units."""
    return max(previous + 1, round((now - origin) * nominal_source_fps))


def percentile(values, proportion):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * proportion
    lo = math.floor(index)
    hi = math.ceil(index)
    return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 2)


def summarize(samples, counters_start, counters_end, wall_s, *, event_count,
              mqtt_acknowledged=0, warmup_batches=0, threads=4):
    """Reported FPS includes waiting for fresh RTSP frames, not just inference."""
    if not samples or wall_s <= 0:
        raise ValueError("No measured streaming batches")
    cameras = sorted(counters_end)
    count = len(samples)
    summary = {
        "measured_batches": count,
        "warmup_batches": warmup_batches,
        "wall_time_s": round(wall_s, 3),
        "cameras": cameras,
        "aggregate_processed_fps": round(count * len(cameras) / wall_s, 3),
        "processed_fps_per_camera": round(count / wall_s, 3),
        "decode_receipt_to_processing_start_ms": {
            "p50": percentile([age for s in samples for age in s["frame_age_ms"].values()], .5),
            "p95": percentile([age for s in samples for age in s["frame_age_ms"].values()], .95),
        },
        "processing_ms_per_four_camera_batch": {
            "p50": percentile([s["processing_ms"] for s in samples], .5),
            "p95": percentile([s["processing_ms"] for s in samples], .95),
        },
        "per_camera": {},
        "events_generated": event_count,
        "mqtt_qos1_acknowledged": mqtt_acknowledged,
        "torch_intraop_threads": threads,
        "notes": (
            "Four RTSP publishers simulated from recorded CCTV. These are processing throughput "
            "and application-buffer overwrite metrics, not capture-to-display latency or total "
            "network/decoder frame loss. Camera images are not time-synchronised. "
            "Any FFmpeg fps filter reduction is intentional source sampling. "
            "Video publishing/transcoding also consumes host CPU."
        ),
    }
    for camera in cameras:
        before_rx, before_drop = counters_start[camera]
        after_rx, after_drop = counters_end[camera]
        received = max(0, after_rx - before_rx)
        overwritten = max(0, after_drop - before_drop)
        summary["per_camera"][camera] = {
            "decoded_frames_received": received,
            "decoded_receipt_fps": round(received / wall_s, 3),
            "processed_frames": count,
            "application_buffer_overwrites": overwritten,
        }
    return summary


# ReID counters and timers are cumulative in CrossCameraMatcher / AppearanceEncoder.
# Snapshot-delta accounting avoids mixing warmup and measurement observations.
REID_FIELDS = (
    "eligible_crops", "encoded_crops", "reused_crops", "encoder_calls",
    "new_local_states", "new_global_ids", "input_prep_ms",
    "boxmot_encode_ms", "output_convert_ms", "other_reid_ms",
)


def reid_snapshot(matcher):
    """Read-only snapshot of cached matcher statistics and encoder timings.

    BoxMOT's ``model_encode`` timer covers the whole BoxMOT encode call,
    including its internal preprocessing; it is NOT a pure neural-network timer.
    """
    if matcher is None:
        return ({key: 0 for key in ("eligible_crops", "encoded_crops",
                 "reused_crops", "encoder_calls", "local_states", "global_ids")}
                | {key: 0.0 for key in ("input_prep", "model_encode", "output_convert")})
    needed = ("eligible_crops", "encoded_crops", "reused_crops", "encoder_calls")
    timings = getattr(matcher.encoder, "timings_ms", None)
    if timings is None or not all(k in timings for k in ("input_prep", "model_encode", "output_convert")):
        raise RuntimeError("--profile-reid requires the Step 11+ instrumented src/reid.py "
                           "AppearanceEncoder with timings_ms")
    if not hasattr(matcher, "stats") or not all(k in matcher.stats for k in needed):
        raise RuntimeError("--profile-reid requires the Step 9+ matcher.stats counters")
    snapshot = {key: int(matcher.stats[key]) for key in needed}
    snapshot.update(local_states=len(matcher.local), global_ids=len(matcher.global_ids))
    snapshot.update({key: float(timings[key]) for key in
                     ("input_prep", "model_encode", "output_convert")})
    return snapshot


def reid_delta(before, after, measured_stage_ms):
    """Per-batch crop/cache activity and sub-stage wall-clock measurements."""
    keys = ("eligible_crops", "encoded_crops", "reused_crops", "encoder_calls")
    counts = {key: after[key] - before[key] for key in keys}
    if any(value < 0 for value in counts.values()):
        raise ValueError("ReID counters must be monotonic")
    if counts["encoded_crops"] + counts["reused_crops"] != counts["eligible_crops"]:
        raise ValueError("ReID cache accounting does not balance")
    counts["new_local_states"] = after["local_states"] - before["local_states"]
    counts["new_global_ids"] = after["global_ids"] - before["global_ids"]
    counts["input_prep_ms"] = max(0.0, after["input_prep"] - before["input_prep"])
    counts["boxmot_encode_ms"] = max(0.0, after["model_encode"] - before["model_encode"])
    counts["output_convert_ms"] = max(0.0, after["output_convert"] - before["output_convert"])
    counts["other_reid_ms"] = max(0.0, measured_stage_ms - sum(
        counts[k] for k in ("input_prep_ms", "boxmot_encode_ms", "output_convert_ms")))
    return counts


def summarize_reid_details(samples, *, enabled=True):
    """Summarise only measured batches; encoding buckets expose crop bursts."""
    rows = [rec["reid_detail"] for rec in samples]
    summed = {key: sum(row[key] for row in rows) for key in REID_FIELDS[:6]}
    eligible = summed["eligible_crops"]
    encoded = summed["encoded_crops"]
    groups = {"zero": [], "1_to_4": [], "5_to_9": [], "10_plus": []}
    for rec in samples:
        count = rec["reid_detail"]["encoded_crops"]
        bucket = ("zero" if count == 0 else "1_to_4" if count <= 4
                  else "5_to_9" if count <= 9 else "10_plus")
        groups[bucket].append(rec["stage_ms"]["reid"])
    active = [r for r in rows if r["encoded_crops"] > 0]
    return {
        "enabled": enabled,
        "measured_batches": len(rows),
        "totals": summed,
        "cache_reuse_pct": round(100.0 * summed["reused_crops"] / eligible, 2) if eligible else None,
        "encoded_per_active_batch_p50": percentile([r["encoded_crops"] for r in active], .5),
        "encoded_per_active_batch_p95": percentile([r["encoded_crops"] for r in active], .95),
        "timings_per_active_encoding_batch_ms": {
            name: {"p50": percentile([r[name] for r in active], .5),
                   "p95": percentile([r[name] for r in active], .95)}
            for name in ("input_prep_ms", "boxmot_encode_ms", "output_convert_ms")
        },
        "reid_stage_ms_by_encoded_crop_count": {
            name: {"batches": len(values), "p50": percentile(values, .5),
                   "p95": percentile(values, .95)} for name, values in groups.items()
        },
        "notes": (
            "Cumulative matcher counters are differenced per measured batch; warmup is excluded. "
            "boxmot_encode_ms measures the entire BoxMOT encode call, including internal crop preprocessing, "
            "not pure OSNet neural-network inference. other_reid_ms includes cache selection, "
            "association and instrumentation overhead. Crops are observations, not unique pedestrians. "
            "Batches without new encodings still include association work. "
            "These statistics do not establish identity accuracy or causality."
        ),
    }


def save_run_results(samples, counters_start, counters_end, wall_s, *,
                     event_count, mqtt_acknowledged, warmup_batches, threads,
                     status, error=None, output_dir=OUTPUT_DIR, profile_stages=False,
                     profile_reid=False, reid_enabled=True):
    """Save a final or partial report, plus an explicit run status.

    Status is always saved. A performance report needs at least one measured
    batch; otherwise generating FPS or latency statistics would be misleading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_record = {
        "status": status,
        "complete": status == "completed",
        "measured_batches": len(samples),
        "error": error,
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(status_record, indent=2) + "\n", encoding="utf-8"
    )
    if not samples:
        print(f"No measured batches. Run status: {status}. "
              f"See {output_dir / 'run_status.json'}", flush=True)
        return None

    report = summarize(samples, counters_start, counters_end, max(wall_s, 1e-9),
                       event_count=event_count,
                       mqtt_acknowledged=mqtt_acknowledged,
                       warmup_batches=warmup_batches, threads=threads)
    report["run_status"] = status
    report["complete"] = status == "completed"
    report["error"] = error
    if profile_stages:
        names = ("wait_for_batch", "detector", "tracker", "provisional", "reid",
                 "events", "event_io_mqtt", "preview", "thread_check")
        report["stage_timings_ms"] = {
            name: {"p50": percentile([rec["stage_ms"][name] for rec in samples], .5),
                   "p95": percentile([rec["stage_ms"][name] for rec in samples], .95)}
            for name in names
        }
        report["stage_timing_notes"] = (
            "Wall-clock per four-camera batch, stages measured sequentially. "
            "wait_for_batch is outside processing_ms; event_io_mqtt includes JSONL "
            "writes and any synchronous QoS1 waits. Percentiles are across batches, "
            "so stage percentiles must not be added to obtain total-batch percentiles."
        )
    if profile_reid:
        report["reid_cache_diagnostics"] = summarize_reid_details(samples, enabled=reid_enabled)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_batch.csv").open("w", newline="", encoding="utf-8") as file:
        stage_names = ("wait_for_batch", "detector", "tracker", "provisional", "reid",
                       "events", "event_io_mqtt", "preview", "thread_check")
        fields = ["frame_id", "processing_ms", "event_count",
                  *[f"age_{cam}_ms" for cam in CAMERAS]]
        if profile_stages:
            fields += ["elapsed_s", *[f"{name}_ms" for name in stage_names],
                       *[f"overwrites_{cam}" for cam in CAMERAS],
                       *[f"received_{cam}" for cam in CAMERAS]]
        if profile_reid:
            fields += [f"reid_{name}" for name in REID_FIELDS]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in samples:
            row = {"frame_id": record["frame_id"],
                   "processing_ms": round(record["processing_ms"], 3),
                   "event_count": record["event_count"],
                   **{f"age_{cam}_ms": round(record["frame_age_ms"][cam], 3)
                      for cam in CAMERAS}}
            if profile_stages:
                row.update({"elapsed_s": round(record["elapsed_s"], 4),
                            **{f"{name}_ms": round(record["stage_ms"][name], 3)
                               for name in stage_names},
                            **{f"overwrites_{cam}": record["counters"][cam][1]
                               for cam in CAMERAS},
                            **{f"received_{cam}": record["counters"][cam][0]
                               for cam in CAMERAS}})
            if profile_reid:
                row.update({f"reid_{key}": round(record["reid_detail"][key], 3)
                            if key.endswith("_ms") else record["reid_detail"][key]
                            for key in REID_FIELDS})
            writer.writerow(row)
    print(f"Run status: {status} | Measured batches: {len(samples)} | "
          f"Aggregate FPS: {report['aggregate_processed_fps']:.2f}", flush=True)
    print(f"Saved {output_dir / 'report.json'} and {output_dir / 'per_batch.csv'}",
          flush=True)
    if error:
        print(f"Run error: {error}", flush=True)
    return report


class MQTTEventSink:
    """Optional local broker publisher; waits for QoS 1 acknowledgements."""

    def __init__(self, host="127.0.0.1", port=1883):
        from .mqtt_events import mqtt_client
        self.mqtt, self.client = mqtt_client()
        self.started = False
        connected = threading.Event()
        status = {"error": None}

        def on_connect(_client, _userdata, _flags, reason_code, _properties):
            if reason_code.is_failure:
                status["error"] = str(reason_code)
            connected.set()

        self.client.on_connect = on_connect
        try:
            self.client.connect(host, port, keepalive=30)
            self.client.loop_start()
            self.started = True
            if not connected.wait(timeout=5):
                raise TimeoutError("MQTT broker did not acknowledge connection")
            if status["error"]:
                raise ConnectionError(f"MQTT broker refused connection: {status['error']}")
        except Exception:
            self.close()
            raise

    def publish(self, event):
        from .mqtt_events import TOPIC_ROOT
        topic = f"{TOPIC_ROOT}/{event['camera']}/{event['event_type']}"
        info = self.client.publish(topic, json.dumps(event, separators=(",", ":")),
                                   qos=1, retain=False)
        if info.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"MQTT publish failed: {self.mqtt.error_string(info.rc)}")
        info.wait_for_publish(timeout=2)
        if not info.is_published():
            raise TimeoutError(f"MQTT QoS1 acknowledgement timeout: {topic}")

    def close(self):
        if self.started:
            self.client.disconnect()
            self.client.loop_stop()
            self.started = False


def run_stream(args):
    """Start independent camera reads, then run the existing vision stages."""
    # Set tcp transport before the FFmpeg decoder is initialised by OpenCV.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    import torch
    from ultralytics.utils import torch_utils
    from .benchmark import configure_cpu_threads, check_intraop_threads
    from .detector import PersonDetector
    from .events import EventEngine
    from .reid import AppearanceEncoder, CrossCameraMatcher
    from .tracker import MultiCameraTracker
    from .provisional_overlay import collect_provisional, render_grid

    configure_cpu_threads(torch, torch_utils, args.threads)
    detector = PersonDetector(device="cpu")
    tracker = MultiCameraTracker()
    matcher = None
    if not args.no_reid:
        matcher = CrossCameraMatcher(AppearanceEncoder(backend="pytorch"),
                                     refresh_every=5, batch_across_cameras=True)
    event_engine = None
    publisher = MQTTEventSink(args.mqtt_host, args.mqtt_port) if args.mqtt else None
    readers = {cam: RTSPReader(cam, f"{args.rtsp_base.rstrip('/')}/{cam}")
               for cam in CAMERAS}
    output_dir = Path(getattr(args, "output_dir", OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each run overwrites its outputs within the selected directory.
    for filename in ("report.json", "per_batch.csv", "run_status.json", "provisional_counts.csv"):
        (output_dir / filename).unlink(missing_ok=True)
    use_preview = bool(getattr(args, "provisional_preview", False))
    profile_reid = bool(getattr(args, "profile_reid", False))
    profile_stages = bool(getattr(args, "profile_stages", False)) or profile_reid
    preview_path = output_dir / "provisional_preview.mp4"
    preview_writer = None
    preview_rows = []
    preview_elapsed_ms = []
    if use_preview:
        # Fixed playback FPS; actual processing timing is reported separately.
        preview_writer = cv2.VideoWriter(str(preview_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                                         (1280, 720))
        if not preview_writer.isOpened():
            preview_writer.release()
            raise OSError(f"Cannot open preview output: {preview_path}")
    measurements = []
    events_seen = 0
    ack_count = 0
    previous_id = -1
    origin = None
    warmup_done = 0
    measured_start = None
    measured_start_unix_s = None
    counters_start = None
    duration = 0
    status = "completed"
    failure = None
    report = None
    try:
        for reader in readers.values():
            print(f"Connecting {reader.camera}: {reader.url}", flush=True)
            reader.start(timeout=args.startup_timeout)
            print(f"{reader.camera}: first frame decoded", flush=True)
        with (output_dir / "events.jsonl").open("w", encoding="utf-8") as out:
            while True:
                if warmup_done >= args.warmup_batches and counters_start is None:
                    counters_start = {cam: reader.slot.counters()
                                      for cam, reader in readers.items()}
                waiting_start = time.perf_counter() if profile_stages else None
                batch = wait_for_batch(readers, timeout=args.timeout)
                processing_start = time.perf_counter()
                stage_ms = {}
                stage_tick = processing_start
                if profile_stages:
                    stage_ms["wait_for_batch"] = 1000 * (processing_start - waiting_start)
                def stamp(name):
                    nonlocal stage_tick
                    if profile_stages:
                        now = time.perf_counter()
                        stage_ms[name] = 1000 * (now - stage_tick)
                        stage_tick = now
                if origin is None:
                    origin = processing_start
                frame_id = timeline_frame_id(processing_start, origin, previous_id)
                previous_id = frame_id
                images = {camera: sample.image for camera, sample in batch.items()}
                ages = {camera: max(0.0, 1000 * (processing_start - sample.received_at))
                        for camera, sample in batch.items()}

                if warmup_done >= args.warmup_batches and measured_start is None:
                    measured_start = processing_start
                    measured_start_unix_s = time.time()

                detections = detector.detect(images)
                stamp("detector")
                tracks = tracker.update(images, detections)
                stamp("tracker")
                # Separate, read-only overlay branch. Never append provisional
                # observations to tracks, which feed ReID and event publishing.
                provisional = (collect_provisional(tracker.trackers, tracks)
                               if use_preview else None)
                stamp("provisional")
                reid_before = reid_snapshot(matcher) if profile_reid else None
                if matcher is not None:
                    tracks, _ = matcher.update(frame_id, images, tracks)
                stamp("reid")
                reid_detail = (reid_delta(reid_before, reid_snapshot(matcher),
                                          stage_ms["reid"]) if profile_reid else None)
                if event_engine is None:
                    h, w = next(iter(images.values())).shape[:2]
                    event_engine = EventEngine(width=w, height=h)
                generated = []
                for camera in CAMERAS:
                    generated.extend(event_engine.update(frame_id, camera, tracks[camera]))
                stamp("events")
                if warmup_done >= args.warmup_batches:
                    for event in generated:
                        # The source is live playback; frame_id is a virtual elapsed-time index.
                        event["elapsed_s"] = round(processing_start - origin, 3)
                        out.write(json.dumps(event) + "\n")
                        if publisher is not None:
                            publisher.publish(event)
                            ack_count += 1
                stamp("event_io_mqtt")
                if use_preview:
                    preview_start = time.perf_counter()
                    canvas = render_grid(images, tracks, provisional)
                    preview_writer.write(canvas)
                    preview_elapsed_ms.append(1000 * (time.perf_counter() - preview_start))
                    if warmup_done >= args.warmup_batches:
                        for camera in CAMERAS:
                            preview_rows.append({"frame_id": frame_id, "camera": camera,
                                                 "confirmed": len(tracks[camera]),
                                                 "provisional": len(provisional[camera])})
                stamp("preview")
                check_intraop_threads(torch, args.threads)
                stamp("thread_check")
                end = time.perf_counter()

                if warmup_done < args.warmup_batches:
                    warmup_done += 1
                    continue
                events_seen += len(generated)
                record = {
                    "frame_id": frame_id,
                    "frame_age_ms": ages,
                    "processing_ms": 1000 * (end - processing_start),
                    "event_count": len(generated),
                }
                if profile_stages:
                    record["stage_ms"] = stage_ms
                    if profile_reid:
                        record["reid_detail"] = reid_detail
                    record["elapsed_s"] = end - measured_start
                    sampled_counters = {cam: reader.slot.counters()
                                        for cam, reader in readers.items()}
                    record["counters"] = {
                        cam: (max(0, sampled_counters[cam][0] - counters_start[cam][0]),
                              max(0, sampled_counters[cam][1] - counters_start[cam][1]))
                        for cam in CAMERAS
                    }
                measurements.append(record)
                duration = time.perf_counter() - measured_start
                if len(measurements) % 25 == 0:
                    print(f"{duration:.1f}s: {len(measurements)} batches, "
                          f"{len(measurements)*4/max(duration, 1e-9):.2f} aggregate FPS",
                          flush=True)
                if duration >= args.seconds or (args.max_batches and
                        len(measurements) >= args.max_batches):
                    break
    except KeyboardInterrupt:
        status = "interrupted"
        failure = "KeyboardInterrupt: stopped by user"
        print("Interrupted; saving available measurements.", flush=True)
    except Exception as exc:
        status = "failed"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Save even when a camera disconnects, a stage raises, or Ctrl+C occurs.
        # Take the time and counters before reader shutdown adds cleanup delays.
        measured_end = time.perf_counter()
        measured_end_unix_s = time.time()
        counters_end = {cam: reader.slot.counters() for cam, reader in readers.items()}
        for reader in readers.values():
            reader.stop()
        if publisher is not None:
            publisher.close()
        if preview_writer is not None:
            preview_writer.release()
        if counters_start is None:
            counters_start = counters_end
        wall_s = (measured_end - measured_start) if measured_start is not None else 0.0
        report = save_run_results(
            measurements, counters_start, counters_end, wall_s,
            event_count=events_seen, mqtt_acknowledged=ack_count,
            warmup_batches=warmup_done, threads=args.threads,
            status=status, error=failure, output_dir=output_dir,
            profile_stages=profile_stages, profile_reid=profile_reid,
            reid_enabled=matcher is not None,
        )
        if report is not None:
            report["measurement_timing"] = {
                "start_unix_s": measured_start_unix_s,
                "end_unix_s": measured_end_unix_s,
                "notes": "System clock timestamps for correlating external host telemetry. "
                         "elapsed_s in per_batch.csv is relative to measured start.",
            }
            report["mqtt_enabled"] = bool(args.mqtt)
            report["mqtt_ack_note"] = (
                "MQTT disabled; zero QoS1 acknowledgements is expected."
                if not args.mqtt else
                "QoS1 acknowledgements confirm broker receipt, not subscriber delivery."
            )
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if use_preview:
            with (output_dir / "provisional_counts.csv").open(
                    "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file,
                                        fieldnames=["frame_id", "camera", "confirmed", "provisional"])
                writer.writeheader()
                writer.writerows(preview_rows)
            if report is not None:
                report["provisional_preview"] = {
                    "enabled": True,
                    "video": str(preview_path),
                    "video_playback_fps": 5,
                    "rendered_batches_including_warmup": len(preview_elapsed_ms),
                    "measured_confirmed_observations": sum(r["confirmed"] for r in preview_rows),
                    "measured_provisional_observations": sum(r["provisional"] for r in preview_rows),
                    "preview_render_write_ms_p50": percentile(preview_elapsed_ms, 0.5),
                    "preview_render_write_ms_p95": percentile(preview_elapsed_ms, 0.95),
                    "notes": "Counts describe visual observations, not unique people. "
                             "Preview render/write time includes warmup; RTSP latency is not end-to-end. "
                             "Confirmed-only tracks still feed ReID, EventEngine and MQTT.",
                }
                (output_dir / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Measured streaming duration after warm-up (default 60)")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Optional cap on measured four-camera batches (0 = no cap)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=12.0,
                        help="Seconds allowed for each camera to deliver its first frame")
    parser.add_argument("--rtsp-base", default="rtsp://127.0.0.1:8554")
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--provisional-preview", action="store_true",
                        help="Render read-only unconfirmed ByteTrack observations in an MP4")
    parser.add_argument("--profile-stages", action="store_true",
                        help="Write per-stage timings and cumulative buffer counters for stability analysis")
    parser.add_argument("--profile-reid", action="store_true",
                        help="Capture per-batch OSNet/BoxMOT encode counts, cache reuse and timings; implies --profile-stages")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Save RTSP metrics, events and optional preview to this folder")
    parser.add_argument("--mqtt", action="store_true", help="Publish events live to local MQTT broker")
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    args = parser.parse_args(argv)
    if (args.seconds <= 0 or args.warmup_batches < 0 or args.threads < 1
            or args.timeout <= 0 or args.startup_timeout <= 0):
        parser.error("seconds, threads and timeouts must be positive; warmup-batches >= 0")
    try:
        run_stream(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
