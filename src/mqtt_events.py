"""Replay saved pedestrian events to a local MQTT broker.

From the project root:
    python -m src.mqtt_events --dry-run
    python -m src.mqtt_events --listen   # separate terminal, run this first
    python -m src.mqtt_events

No detector or ReID inference is performed here.
"""

import argparse
import json
import math
import time
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "outputs" / "events" / "events.jsonl"
TOPIC_ROOT = "multicam/events"
CAMERAS = {"C1", "C2", "C3", "C4"}
EVENT_TYPES = {"zone_entry", "zone_exit", "line_crossing"}


def read_events(path):
    """Yield validated records from Step 5, preserving the original schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run 'python -m src.events' first.")
    last_time = -1.0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}") from exc
            if not isinstance(event, dict) or not all(
                key in event for key in ("event_id", "event_type", "camera", "elapsed_s")
            ):
                raise ValueError(f"Missing event fields on line {line_no}")
            if not isinstance(event["event_id"], str) or not event["event_id"]:
                raise ValueError(f"Invalid event_id on line {line_no}")
            if event["camera"] not in CAMERAS or event["event_type"] not in EVENT_TYPES:
                raise ValueError(f"Unknown camera or event type on line {line_no}")
            try:
                elapsed = float(event["elapsed_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid elapsed_s on line {line_no}") from exc
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed < last_time:
                raise ValueError(f"Event times must be finite and ordered (line {line_no})")
            last_time = elapsed
            yield event


def replay_events(events, send, speed=1.0, sleep=time.sleep):
    """Send records in order, optionally pacing by video-relative elapsed_s."""
    if not math.isfinite(speed) or speed < 0:
        raise ValueError("speed must be finite and >= 0")
    count, previous_time = 0, None
    for event in events:
        current_time = float(event["elapsed_s"])
        if speed and previous_time is not None:
            sleep(max(0, current_time - previous_time) / speed)
        topic = f"{TOPIC_ROOT}/{event['camera']}/{event['event_type']}"
        send(topic, json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        previous_time = current_time
        count += 1
    return count


def mqtt_client():
    """Import MQTT lazily so --dry-run and data tests need no MQTT package."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise RuntimeError("Install the MQTT dependency: pip install -r requirements.txt") from exc
    return mqtt, mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             protocol=mqtt.MQTTv311, reconnect_on_failure=False)


def publish_file(path=EVENTS_PATH, host="127.0.0.1", port=1883, speed=1.0):
    """Publish with QoS 1 and wait for broker acknowledgements."""
    # Check that the file exists before trying the broker connection.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run 'python -m src.events' first.")

    mqtt, client = mqtt_client()
    connected = Event()
    status = {"error": None}

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            status["error"] = f"Broker rejected connection: {reason_code}"
        connected.set()

    client.on_connect = on_connect
    loop_running = False
    try:
        client.connect(host, port, keepalive=30)
        client.loop_start()
        loop_running = True
        if not connected.wait(timeout=5):
            raise TimeoutError("No MQTT connection acknowledgement in 5 seconds")
        if status["error"]:
            raise ConnectionError(status["error"])

        def send(topic, payload):
            info = client.publish(topic, payload, qos=1, retain=False)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise ConnectionError(f"MQTT publish failed: {mqtt.error_string(info.rc)}")
            info.wait_for_publish(timeout=10)
            if not info.is_published():
                raise TimeoutError(f"No broker acknowledgement for {topic}")

        count = replay_events(read_events(path), send, speed=speed)
        return count
    finally:
        if loop_running:
            client.disconnect()
            client.loop_stop()


def listen(host="127.0.0.1", port=1883):
    """Print incoming events to verify the publisher in another terminal."""
    _, client = mqtt_client()

    def on_connect(c, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            print(f"Broker rejected connection: {reason_code}")
            c.disconnect()
            return
        c.subscribe(f"{TOPIC_ROOT}/#", qos=1)
        print(f"Subscribed to {TOPIC_ROOT}/# (Ctrl+C to exit)", flush=True)

    def on_message(_client, _userdata, message):
        print(f"{message.topic} {message.payload.decode('utf-8')}", flush=True)

    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(host, port, keepalive=30)
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EVENTS_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="1=realtime replay, 0=as fast as possible")
    parser.add_argument("--dry-run", action="store_true", help="Print events; no broker")
    parser.add_argument("--listen", action="store_true", help="Subscribe and print MQTT events")
    args = parser.parse_args()

    try:
        if args.listen:
            listen(args.host, args.port)
        elif args.dry_run:
            count = replay_events(read_events(args.input),
                                  lambda topic, payload: print(topic, payload), speed=0)
            print(f"Validated {count} events (not published)")
        else:
            count = publish_file(args.input, args.host, args.port, args.speed)
            print(f"Published {count} events with QoS 1 (broker acknowledged)")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
