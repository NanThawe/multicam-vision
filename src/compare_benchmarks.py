"""Compare repeated Step 10 benchmark runs; NOT an accuracy evaluation.

Run: python -m src.compare_benchmarks --root outputs/benchmarks/step10
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def summarise(root, runs=3):
    results = {}
    for mode in ("per_camera", "batched"):
        reports = []
        for index in range(1, runs + 1):
            path = root / f"{mode}_{index}" / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["configuration"]["reid_batch_across_cameras"] != (mode == "batched"):
                raise ValueError(f"Incorrect batching mode recorded in {path}")
            reports.append(report)
        results[mode] = {
            "aggregate_fps_median": round(statistics.median(
                r["aggregate_fps"] for r in reports), 3),
            "aggregate_fps_min": min(r["aggregate_fps"] for r in reports),
            "aggregate_fps_max": max(r["aggregate_fps"] for r in reports),
            "total_p95_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["total"]["p95"]
                for r in reports), 2),
            "reid_p50_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["reid"]["p50"]
                for r in reports), 2),
            "encoder_calls_median": statistics.median(
                r["reid_workload"]["encoder_calls"] for r in reports),
            "encoded_crops_median": statistics.median(
                r["reid_workload"]["encoded_crops"] for r in reports),
        }
    base = results["per_camera"]["aggregate_fps_median"]
    batched = results["batched"]["aggregate_fps_median"]
    results["throughput_ratio_batched_vs_per_camera"] = (
        round(batched / base, 3) if base > 0 else None
    )
    return results


def local_identity_map(path):
    """Get each camera-local track's most common assigned global ID."""
    votes = defaultdict(Counter)
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            for person in row["tracks"]:
                if person.get("global_id") is not None:
                    votes[(row["camera"], person["track_id"])][person["global_id"]] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in votes.items()}


def compare_identity_partitions(reference, candidate):
    """Compare association decisions; agreement is NOT ground-truth accuracy.

    Global IDs may be renumbered between runs, so compare same/different ID
    relationships across *camera-local tracks* instead of comparing raw IDs.
    """
    before = local_identity_map(reference)
    after = local_identity_map(candidate)
    shared = sorted(before.keys() & after.keys())
    reference_pairs = candidate_pairs = false_splits = false_merges = 0
    compared = 0
    for first, second in combinations(shared, 2):
        if first[0] == second[0]:
            continue
        compared += 1
        same_before = before[first] == before[second]
        same_after = after[first] == after[second]
        reference_pairs += int(same_before)
        candidate_pairs += int(same_after)
        false_splits += int(same_before and not same_after)
        false_merges += int(not same_before and same_after)
    return {
        "shared_local_tracks": len(shared),
        "reference_local_tracks": len(before),
        "candidate_local_tracks": len(after),
        "cross_camera_pairs_compared": compared,
        "reference_same_identity_pairs": reference_pairs,
        "candidate_same_identity_pairs": candidate_pairs,
        "reference_matches_missing_in_candidate": false_splits,
        "new_candidate_matches_vs_reference": false_merges,
        "disagreement_pairs": false_splits + false_merges,
        "note": "Association agreement only; neither run has ground-truth IDs.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("outputs/benchmarks/step10"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reference-tracks", type=Path,
                        help="Optional per-camera reference global_tracks.jsonl")
    parser.add_argument("--batched-tracks", type=Path,
                        help="Optional batched global_tracks.jsonl to compare")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if bool(args.reference_tracks) != bool(args.batched_tracks):
        parser.error("Supply both --reference-tracks and --batched-tracks")
    results = summarise(args.root, args.runs)
    if args.reference_tracks:
        results["identity_agreement"] = compare_identity_partitions(
            args.reference_tracks, args.batched_tracks)
    path = args.root / "comparison.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {path}")
    print("Speed comparisons do not establish ReID accuracy.")


if __name__ == "__main__":
    main()
