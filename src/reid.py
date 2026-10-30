"""Associate ByteTrack identities across cameras using OSNet appearance features.

Run from project root: python -m src.reid
These IDs are appearance-based hypotheses, not verified person identities.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "reid"
MAX_BATCHES = 30  # Set to None to run the full recordings.
MIN_OBSERVATIONS = 3
MATCH_THRESHOLD = 0.85  # Initial heuristic; calibrate using labelled examples.
MATCH_MARGIN = 0.05

# BoxMOT 25 requires a resolved local checkpoint and its SHA-256 identity.
MODEL_WEIGHTS = ROOT / "models" / "osnet_x0_25_msmt17.pt"
MODEL_URL = (
    "https://huggingface.co/spaces/xfys/yolov5_tracking/resolve/main/"
    "weights/osnet_x0_25_msmt17.pt"
)
MODEL_SHA256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
ONNX_WEIGHTS = MODEL_WEIGHTS.with_suffix(".onnx")


def unit(vector):
    """Return an L2-normalised appearance vector."""
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class AppearanceEncoder:
    """BoxMOT OSNet adapter, with optional cross-camera crop batching.

    The BoxMOT v25 encoder accepts parallel lists of Frames and Detections.
    We call it once for all cameras in a batch instead of once per camera.
    """

    def __init__(self, backend="pytorch"):
        if backend not in ("pytorch", "onnx"):
            raise ValueError("ReID backend must be pytorch or onnx")

        from boxmot.components.artifacts import resolve_artifact
        from boxmot.reid import ReIDEncoderSpec, create_reid_encoder

        if backend == "pytorch":
            MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            artifact = resolve_artifact(
                MODEL_WEIGHTS,
                source_uri=MODEL_URL,
                expected_sha256=MODEL_SHA256,
                allow_download=True,
            )
            model_path, digest = artifact.path, artifact.sha256
        else:
            if not ONNX_WEIGHTS.is_file():
                raise FileNotFoundError(
                    f"Missing ONNX model: {ONNX_WEIGHTS}\n"
                    "Export it first using: boxmot export --weights "
                    "models/osnet_x0_25_msmt17.pt --include onnx --device cpu --dynamic"
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("Install ONNX Runtime: pip install onnxruntime") from exc
            if "CPUExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CPUExecutionProvider unavailable")
            from boxmot.components.artifacts import sha256_artifact
            model_path, digest = ONNX_WEIGHTS, sha256_artifact(ONNX_WEIGHTS)

        self.backend = backend
        self.model = create_reid_encoder(
            ReIDEncoderSpec(
                backend=backend,
                artifact=str(model_path),
                artifact_sha256=digest,
                device="cpu",
                precision="fp32",
            )
        )
        self.timings_ms = {"input_prep": 0.0, "model_encode": 0.0,
                           "output_convert": 0.0}

    def encode(self, camera, frame_id, image, people):
        """Backward-compatible single-camera API."""
        if not people:
            return []
        return self.encode_batch([(camera, frame_id, image, people)])[0]

    def encode_batch(self, requests):
        """Encode several cameras in one BoxMOT call, preserving their order.

        requests: [(camera, frame_id, BGR_image, people), ...]
        returns:  [NxD appearance embeddings for each request, ...]
        Empty requests are allowed; empty people lists are omitted by caller.
        """
        if not requests:
            return []

        import torch
        from boxmot.structures import Boxes, Detections, Frame

        start = time.perf_counter()
        frames, detections = [], []
        for camera, frame_id, image, people in requests:
            if not people:
                raise ValueError("encode_batch accepts only nonempty pedestrian lists")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample_id = f"{camera}:{frame_id:08d}"
            frames.append(Frame(
                image=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                sample_id=sample_id,
                sequence_id=camera,
                frame_index=int(frame_id),
                timestamp_s=None,
                source_uri=camera,
            ))
            detections.append(Detections(
                geometry=Boxes(torch.tensor([p["xyxy"] for p in people],
                                            dtype=torch.float32)),
                scores=torch.tensor([p["confidence"] for p in people],
                                    dtype=torch.float32),
                class_ids=torch.zeros(len(people), dtype=torch.int64),
                sample_id=sample_id,
            ))
        built = time.perf_counter()
        results = self.model.encode(frames, detections)
        encoded = time.perf_counter()
        if len(results) != len(requests):
            raise ValueError("ReID encoder returned the wrong number of camera batches")

        outputs = []
        for result, (_, _, _, people) in zip(results, requests):
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            result = np.asarray(result, dtype=np.float32)
            if result.ndim != 2 or result.shape[0] != len(people):
                raise ValueError("ReID encoder returned unexpected crop embeddings")
            outputs.append(result)
        finished = time.perf_counter()
        self.timings_ms["input_prep"] += (built - start) * 1000
        self.timings_ms["model_encode"] += (encoded - built) * 1000
        self.timings_ms["output_convert"] += (finished - encoded) * 1000
        return outputs


class CrossCameraMatcher:
    """Assign tentative global IDs while reusing features on stable local tracks.

    The first `min_observations` valid crops are always encoded. Afterwards,
    a local track gets a new embedding at most once per `refresh_every` batches.
    `refresh_every=1` preserves the original full-rate reference behaviour.
    """

    def __init__(self, encoder, min_observations=MIN_OBSERVATIONS,
                 threshold=MATCH_THRESHOLD, margin=MATCH_MARGIN,
                 refresh_every=1, batch_across_cameras=False):
        if refresh_every < 1 or min_observations < 1:
            raise ValueError("refresh_every and min_observations must be >= 1")
        self.refresh_every = refresh_every
        self.batch_across_cameras = batch_across_cameras
        self.batch_index = -1
        self.stats = {"eligible_crops": 0, "encoded_crops": 0,
                      "reused_crops": 0, "encoder_calls": 0}
        self.encoder = encoder
        self.min_observations = min_observations
        self.threshold = threshold
        self.margin = margin
        self.local = {}  # (camera, track_id) -> feature, count, global_id, last_encoded
        self.global_ids = {}  # global_id -> feature, cameras assigned
        self.next_id = 1

    def update(self, frame_id, frames, tracks):
        """Return (tracks annotated with global IDs, new association decisions)."""
        self.batch_index += 1
        output = {camera: [dict(person, global_id=None) for person in people]
                  for camera, people in tracks.items()}
        decisions = []

        # Select crops first, then use either the original per-camera path or
        # one cross-camera request. Local track schedules remain unchanged.
        requests = []
        for camera, people in output.items():
            to_encode = []
            for person in people:
                x1, y1, x2, y2 = person["xyxy"]
                if x2 - x1 < 16 or y2 - y1 < 32:
                    continue
                self.stats["eligible_crops"] += 1
                state = self.local.get((camera, person["track_id"]))
                if (state is None or state["count"] < self.min_observations
                        or self.batch_index - state["last_encoded"] >= self.refresh_every):
                    to_encode.append(person)
                else:
                    self.stats["reused_crops"] += 1
            if to_encode:
                requests.append((camera, frame_id, frames[camera], to_encode))

        results = []
        if requests and self.batch_across_cameras:
            if not hasattr(self.encoder, "encode_batch"):
                raise TypeError("Encoder must provide encode_batch for cross-camera batching")
            results = self.encoder.encode_batch(requests)
            self.stats["encoder_calls"] += 1
        else:
            for camera, index, image, people in requests:
                results.append(self.encoder.encode(camera, index, image, people))
                self.stats["encoder_calls"] += 1

        if len(results) != len(requests):
            raise ValueError("Wrong number of encoder results")
        for (camera, _, _, people), embeddings in zip(requests, results):
            if len(embeddings) != len(people):
                raise ValueError("Encoder did not return one embedding per input crop")
            self.stats["encoded_crops"] += len(people)
            for person, embedding in zip(people, embeddings):
                key = (camera, person["track_id"])
                feature = unit(embedding)
                if key not in self.local:
                    self.local[key] = {"feature": feature, "count": 1,
                                       "global_id": None, "last_encoded": self.batch_index}
                else:
                    state = self.local[key]
                    state["feature"] = unit(0.8 * state["feature"] + 0.2 * feature)
                    state["count"] += 1
                    state["last_encoded"] = self.batch_index

        # Then assign identities. Deterministic camera order permits overlapping views.
        for camera, people in output.items():
            for person in people:
                key = (camera, person["track_id"])
                state = self.local.get(key)
                if state is None or state["count"] < self.min_observations:
                    continue

                if state["global_id"] is None:
                    candidates = sorted(
                        (
                            (float(np.dot(state["feature"], identity["feature"])), gid)
                            for gid, identity in self.global_ids.items()
                            if camera not in identity["cameras"]
                        ),
                        reverse=True,
                    )
                    best = candidates[0][0] if candidates else -1.0
                    second = candidates[1][0] if len(candidates) > 1 else -1.0
                    matched = best >= self.threshold and best - second >= self.margin

                    if matched:
                        gid = candidates[0][1]
                        decision = "cross_camera_match"
                    else:
                        gid = self.next_id
                        self.next_id += 1
                        self.global_ids[gid] = {"feature": state["feature"].copy(),
                                                "cameras": {}}
                        decision = "new_identity"

                    state["global_id"] = gid
                    self.global_ids[gid]["cameras"][camera] = person["track_id"]
                    decisions.append({
                        "frame_id": frame_id, "camera": camera,
                        "track_id": person["track_id"], "global_id": gid,
                        "decision": decision,
                        "similarity": round(best, 3) if matched else None,
                    })

                gid = state["global_id"]
                identity = self.global_ids[gid]
                identity["feature"] = unit(0.9 * identity["feature"] + 0.1 * state["feature"])
                person["global_id"] = gid

        return output, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-every", type=int, default=1,
                        help="Encode stable tracks every N sampled batches (default: 1)")
    parser.add_argument("--reid-backend", choices=("pytorch", "onnx"), default="pytorch",
                        help="OSNet inference backend (default: pytorch)")
    parser.add_argument("--batch-across-cameras", action="store_true",
                        help="Encode crops from all cameras in one BoxMOT call")
    parser.add_argument("--batches", type=int, default=MAX_BATCHES,
                        help="Number of sampled batches to process (default: 30)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Save outputs in this folder (default: outputs/reid)")
    args = parser.parse_args(argv)
    if args.refresh_every < 1 or args.batches < 1:
        parser.error("--refresh-every and --batches must be positive integers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "preview.mp4"
    tracks_path = args.output_dir / "global_tracks.jsonl"
    decisions_path = args.output_dir / "associations.jsonl"

    from .detector import PersonDetector
    from .framesource import iter_frames
    from .tracker import MultiCameraTracker, draw_grid, SAMPLE_FPS

    detector = PersonDetector()
    tracker = MultiCameraTracker()
    matcher = CrossCameraMatcher(AppearanceEncoder(backend=args.reid_backend), refresh_every=args.refresh_every,
                                 batch_across_cameras=args.batch_across_cameras)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             SAMPLE_FPS, (1280, 720))
    if not writer.isOpened():
        raise OSError(f"Cannot write preview: {video_path}")

    reader = iter_frames(sample_fps=SAMPLE_FPS)
    batches = 0
    try:
        with tracks_path.open("w", encoding="utf-8") as track_file, \
                decisions_path.open("w", encoding="utf-8") as decision_file:
            for frame_id, frames in reader:
                tracks = tracker.update(frames, detector.detect(frames))
                people, decisions = matcher.update(frame_id, frames, tracks)
                writer.write(draw_grid(frames, people))
                for camera, persons in people.items():
                    track_file.write(json.dumps({"frame_id": frame_id,
                                                 "camera": camera,
                                                 "tracks": persons}) + "\n")
                for decision in decisions:
                    decision_file.write(json.dumps(decision) + "\n")
                batches += 1
                print(f"Frame {frame_id}: {sum(len(p) for p in people.values())} local tracks, "
                      f"{len(matcher.global_ids)} global IDs, {len(decisions)} new decisions")
                if batches >= args.batches:
                    break
    finally:
        reader.close()
        writer.release()

    print(f"ReID statistics: {matcher.stats}")
    print(f"Saved {video_path}, {tracks_path}, and {decisions_path}")


if __name__ == "__main__":
    main()
