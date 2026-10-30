"""Check PyTorch and ONNX OSNet embeddings on identical WILDTRACK detections.

Run after export: python -m src.validate_onnx
This tests numerical agreement, NOT labelled ReID accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .reid import AppearanceEncoder, ROOT


def embedding_agreement(reference, candidate, min_cosine=0.99):
    """Compare two NxD embedding matrices without treating cosine as accuracy."""
    ref = np.asarray(reference, dtype=np.float32)
    other = np.asarray(candidate, dtype=np.float32)
    if ref.shape != other.shape or ref.ndim != 2 or ref.shape[0] == 0:
        raise ValueError("Embedding arrays must have identical, nonempty NxD shapes")
    if not np.isfinite(ref).all() or not np.isfinite(other).all():
        raise ValueError("Non-finite embedding values")
    ref_norm = np.linalg.norm(ref, axis=1)
    other_norm = np.linalg.norm(other, axis=1)
    if np.any(ref_norm < 1e-8) or np.any(other_norm < 1e-8):
        raise ValueError("Zero-norm embeddings")
    similarities = np.sum(ref * other, axis=1) / (ref_norm * other_norm)
    scores = {
        "crops": int(len(ref)),
        "min_cosine": round(float(similarities.min()), 6),
        "mean_cosine": round(float(similarities.mean()), 6),
        "max_absolute_difference": round(float(np.max(np.abs(ref - other))), 6),
        "passed": bool((similarities >= min_cosine).all()),
        "min_cosine_required": min_cosine,
        "note": "Numerical backend agreement on identical crops, not ReID accuracy.",
    }
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-crops", type=int, default=24)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "benchmarks" / "step11")
    args = parser.parse_args(argv)
    if args.max_crops < 1 or args.max_batches < 1 or not 0 < args.min_cosine <= 1:
        parser.error("Invalid max-crops, max-batches or min-cosine")

    from .detector import PersonDetector
    from .framesource import iter_frames

    reader = iter_frames()
    requests = []
    selected = 0
    detector = PersonDetector()
    try:
        for batch_number, (frame_id, frames) in enumerate(reader, start=1):
            detections = detector.detect(frames)
            for camera, people in detections.items():
                valid = [p for p in people
                         if p["xyxy"][2] - p["xyxy"][0] >= 16
                         and p["xyxy"][3] - p["xyxy"][1] >= 32]
                valid = valid[:args.max_crops - selected]
                if valid:
                    requests.append((camera, frame_id, frames[camera], valid))
                    selected += len(valid)
                if selected >= args.max_crops:
                    break
            if selected >= args.max_crops or batch_number >= args.max_batches:
                break
    finally:
        reader.close()

    if not requests:
        raise RuntimeError("No valid person detections found for backend comparison")

    baseline = AppearanceEncoder(backend="pytorch").encode_batch(requests)
    deployed = AppearanceEncoder(backend="onnx").encode_batch(requests)
    result = embedding_agreement(np.concatenate(baseline), np.concatenate(deployed),
                                 min_cosine=args.min_cosine)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "parity.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {path}")
    if not result["passed"]:
        raise SystemExit("ONNX embeddings differ from PyTorch beyond chosen threshold; review before benchmarking")


if __name__ == "__main__":
    main()
