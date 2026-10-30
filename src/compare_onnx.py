"""Compare 3 paired OSNet CPU-backend benchmarks, excluding export/model setup.

Run: python -m src.compare_onnx
"""

import argparse
import json
import statistics
from pathlib import Path


def compare(root, repeats=3):
    output = {}
    workloads = {}
    for backend in ("pytorch", "onnx"):
        rows = []
        for index in range(1, repeats + 1):
            path = root / f"{backend}_{index}" / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            config = report["configuration"]
            if config.get("reid_backend") != backend:
                raise ValueError(f"Wrong ReID backend reported in {path}")
            if not config.get("reid_batch_across_cameras") or config.get("reid_interval") != 5:
                raise ValueError(f"Benchmark configuration changed: {path}")
            workloads[(backend, index)] = (report["measured_batches"],
                                           report["reid_workload"]["eligible_crops"],
                                           report["reid_workload"]["encoded_crops"])
            rows.append(report)
        output[backend] = {
            "aggregate_fps_median": round(statistics.median(
                r["aggregate_fps"] for r in rows), 3),
            "aggregate_fps_range": [min(r["aggregate_fps"] for r in rows),
                                    max(r["aggregate_fps"] for r in rows)],
            "reid_p50_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["reid"]["p50"]
                for r in rows), 2),
            "total_p95_ms_median": round(statistics.median(
                r["latency_ms_per_four_camera_batch"]["total"]["p95"]
                for r in rows), 2),
        }
    for index in range(1, repeats + 1):
        if workloads[("pytorch", index)] != workloads[("onnx", index)]:
            raise ValueError(f"Different crop workloads in paired run {index}; results not comparable")
    a = output["pytorch"]["aggregate_fps_median"]
    b = output["onnx"]["aggregate_fps_median"]
    output["onnx_vs_pytorch_fps_ratio"] = round(b / a, 3) if a > 0 else None
    output["interpretation"] = "Offline CPU throughput only; numerical parity is checked separately."
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmarks/step11"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    result = compare(args.root, args.repeats)
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / "comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
