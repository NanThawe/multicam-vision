"""Step 14.26 — independent Market-1501 CPU ReID encoder comparison.

Research only. Never modifies src.reid or the live RTSP/MQTT pipeline.
Market-1501 query/gallery is a *retrieval* benchmark, NOT identity-association
or TrackEval IDF1/HOTA. All thresholds fixed: no tuning on the test labels.

python -m src.encoder_benchmark prepare --dataset-dir data/market1501/Market-1501-v15.09.15
python -m src.encoder_benchmark run --threads 4 --affinity 0,2,4,6,8,10
python -m src.encoder_benchmark report
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluation" / "identity_evaluation" / "encoder_benchmark_market1501"
DATASET = ROOT / "data" / "market1501" / "Market-1501-v15.09.15"
BASE_WEIGHTS = ROOT / "models" / "osnet_x0_25_msmt17.pt"
ALT_WEIGHTS = ROOT / "models" / "osnet_x1_0_msmt17.pt"
BASE_SHA256 = "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18"
FILE_PATTERN = re.compile(r"^(?P<pid>-1|\d{4})_c(?P<cam>[1-6])(?:s\d+)?_.*\.(?:jpg|jpeg|png)$", re.I)
PROTOCOL = "market1501_single_query_no_reranking; exclude gallery PID=-1 and query-identical PID+camera"


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_hash(value, seed=1426):
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def parse_name(filename):
    match = FILE_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Not a Market-1501 image filename: {filename}")
    return int(match["pid"]), int(match["cam"])


def list_market(dataset_dir):
    root = Path(dataset_dir)
    result = {}
    for group, folder in (("query", "query"), ("gallery", "bounding_box_test")):
        p = root / folder
        if not p.is_dir():
            raise FileNotFoundError(f"Missing Market-1501 {group} folder: {p}. Extract the dataset first.")
        imgs = []
        for entry in sorted(p.iterdir()):
            if not entry.is_file() or entry.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            pid, cam = parse_name(entry.name)
            if group == "query" and pid <= 0:
                continue
            if group == "gallery" and pid == -1:
                continue  # Official protocol always filters junk=-1.
            imgs.append({"rel": f"{folder}/{entry.name}", "pid": pid, "cam": cam})
        if not imgs:
            raise ValueError(f"No Market-1501 images in {p}")
        result[group] = imgs
    return result


def _possible_queries(query, gallery):
    available = {(g["pid"], g["cam"]) for g in gallery if g["pid"] > 0}
    return [q for q in query if any((q["pid"], other) in available for other in range(1, 7) if other != q["cam"])]


def select_samples(pool, mode="pilot", queries=150, gallery_per_pid_camera=2, seed=1426):
    if mode not in ("pilot", "full"):
        raise ValueError("mode must be pilot or full")
    if queries < 1 or gallery_per_pid_camera < 1:
        raise ValueError("Pilot sizes must be positive")
    all_q, all_g = pool["query"], pool["gallery"]
    if mode == "full":
        selected_g = all_g
        selected_q = _possible_queries(all_q, selected_g)
    else:
        buckets = defaultdict(list)
        for g in all_g:
            buckets[(g["pid"], g["cam"])].append(g)
        selected_g = []
        for group in buckets.values():
            selected_g.extend(sorted(group, key=lambda r: (stable_hash(r["rel"], seed), r["rel"]))[:gallery_per_pid_camera])
        # Sample queries only AFTER fixing gallery; otherwise a pilot can unknowingly
        # include queries with zero correct cross-camera gallery images.
        candidates = _possible_queries(all_q, selected_g)
        # Spread selections over identities, deterministic, then fill remaining slots.
        by_pid = defaultdict(list)
        for q in candidates:
            by_pid[q["pid"]].append(q)
        people = sorted(by_pid, key=lambda pid: (stable_hash(f"person:{pid}", seed), pid))
        ordered = []
        while people and len(ordered) < queries:
            rest = []
            for pid in people:
                row = min(by_pid[pid], key=lambda q: (stable_hash(q["rel"], seed), q["rel"]))
                ordered.append(row)
                by_pid[pid].remove(row)
                if by_pid[pid]:
                    rest.append(pid)
                if len(ordered) >= queries:
                    break
            people = rest
        selected_q = ordered
    selected_q = sorted(selected_q, key=lambda x: x["rel"])
    selected_g = sorted(selected_g, key=lambda x: x["rel"])
    if not selected_q or not selected_g:
        raise ValueError("Selection produced no valid query/gallery images")
    return selected_q, selected_g


def prepare(dataset_dir=DATASET, output_dir=OUT, mode="pilot", max_queries=150, gallery_per_pid_camera=2, seed=1426):
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)
    pool = list_market(dataset_dir)
    query, gallery = select_samples(pool, mode, max_queries, gallery_per_pid_camera, seed)
    existing = output_dir / "selection.json"
    result = {
        "version": 1,
        "dataset_name": "Market-1501-v15.09.15",
        "dataset_root": str(dataset_dir),
        "protocol": PROTOCOL,
        "mode": mode,
        "official_protocol": mode == "full",
        "pilot_is_not_official_market1501_metric": mode != "full",
        "selection_seed": seed,
        "max_queries": max_queries if mode == "pilot" else None,
        "gallery_per_pid_camera": gallery_per_pid_camera if mode == "pilot" else None,
        "query": query,
        "gallery": gallery,
        "counts": {
            "query": len(query), "gallery": len(gallery),
            "query_identities": len({q["pid"] for q in query}),
            "gallery_identities": len({g["pid"] for g in gallery if g["pid"] > 0}),
            "query_by_camera": {str(i): sum(q["cam"] == i for q in query) for i in range(1, 7)},
        },
        "limits": ["Query/gallery images share person identities by design, as required for retrieval.",
                   "Market-1501 is separate from previously inspected WILDTRACK, but model training provenance must be independently checked.",
                   "Pilot gallery/query sampling changes benchmark difficulty and is NOT an official Market-1501 score.",
                   "Benchmarking image crops does not measure live tracker or cross-camera association accuracy."],
    }
    canonical = json.dumps({"query": query, "gallery": gallery, "seed": seed, "mode": mode}, sort_keys=True, separators=(",", ":"))
    result["selection_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous["selection_sha256"] != result["selection_sha256"] or previous["dataset_root"] != result["dataset_root"]:
            raise ValueError("Existing output directory uses a different dataset/selection. Choose --output-dir instead of overwriting.")
    dump(existing, result)
    return result


def _load_selection(output_dir, dataset_dir=None):
    result = json.loads((Path(output_dir) / "selection.json").read_text(encoding="utf-8"))
    root = Path(dataset_dir or result["dataset_root"]).resolve()
    if root != Path(result["dataset_root"]).resolve():
        raise ValueError("Prepared dataset root differs; rerun prepare in a new --output-dir")
    for sample in result["query"] + result["gallery"]:
        if Path(sample["rel"]).is_absolute() or ".." in Path(sample["rel"]).parts:
            raise ValueError("Invalid sample path")
        if not (root / sample["rel"]).is_file():
            raise FileNotFoundError(f"Selected image missing: {root / sample['rel']}")
    return result, root


def create_encoder(weights):
    """Use identical BoxMOT v25 interface, preprocessing, CPU and FP32 for BOTH OSNets."""
    from boxmot.reid import ReIDEncoderSpec, create_reid_encoder
    weights = Path(weights).resolve()
    return create_reid_encoder(ReIDEncoderSpec(
        backend="pytorch", artifact=str(weights), artifact_sha256=hash_file(weights),
        device="cpu", precision="fp32"))


def _load_batch(root, samples):
    import cv2
    import torch
    from boxmot.structures import Boxes, Detections, Frame
    frames, dets = [], []
    for image_index, item in enumerate(samples):
        path = root / item["rel"]
        image = cv2.imread(str(path))
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Failed to decode image: {path}")
        height, width = image.shape[:2]
        if width < 16 or height < 32:
            raise ValueError(f"Image too small for encoder: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample_id = item["rel"]
        frames.append(Frame(image=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                            sample_id=sample_id, sequence_id="Market-1501",
                            frame_index=image_index, timestamp_s=None, source_uri=str(path)))
        dets.append(Detections(
            geometry=Boxes(torch.tensor([[0., 0., float(width), float(height)]], dtype=torch.float32)),
            scores=torch.ones(1, dtype=torch.float32),
            class_ids=torch.zeros(1, dtype=torch.int64), sample_id=sample_id))
    return frames, dets


def _model_embeddings(model, frames, dets):
    import torch
    raw = model.encode(frames, dets)
    if len(raw) != len(frames):
        raise ValueError("Encoder returned wrong number of images")
    feats = []
    for out in raw:
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        if out.ndim != 2 or out.shape[0] != 1 or not out.shape[1]:
            raise ValueError("Expected exactly one ReID vector per Market-1501 crop")
        v = out[0]
        n = float(np.linalg.norm(v))
        if not np.isfinite(v).all() or not np.isfinite(n) or n <= 1e-12:
            raise ValueError("Encoder returned invalid vector")
        feats.append(v/n)
    return np.stack(feats)


def extract(model, root, samples, batch_size, progress_name="encoder"):
    output = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start+batch_size]
        frames, dets = _load_batch(root, chunk)
        output.append(_model_embeddings(model, frames, dets))
        if (start // batch_size + 1) % 20 == 0 or start + len(chunk) == len(samples):
            print(f"{progress_name}: {min(start+batch_size,len(samples))}/{len(samples)} crops", flush=True)
    return np.concatenate(output, axis=0), round(time.perf_counter()-started, 3)


def benchmark_cpu(model, root, samples, batch_size=16, repeats=3, sample_size=64):
    """Decode once, warm up once, then time ONLY encode+output conversion on fixed crops."""
    group = sorted(samples, key=lambda item: (stable_hash(item["rel"]), item["rel"]))[:min(sample_size, len(samples))]
    if not group:
        raise ValueError("No latency samples")
    chunks = [_load_batch(root, group[i:i+batch_size]) for i in range(0, len(group), batch_size)]
    for frames, dets in chunks:
        _model_embeddings(model, frames, dets)
    trials = []
    for _ in range(repeats):
        start = time.perf_counter()
        for frames, dets in chunks:
            _model_embeddings(model, frames, dets)
        trials.append((time.perf_counter()-start)*1000 / len(group))
    return {"number_crops": len(group), "batch_size": batch_size, "repeats": repeats,
            "ms_per_crop_trials": [round(x, 4) for x in trials],
            "median_ms_per_crop": round(float(statistics.median(trials)), 4),
            "median_crops_per_second": round(1000/float(statistics.median(trials)), 3),
            "scope": "In-memory fixed Market-1501 crops; BoxMOT encode + output conversion only; no camera/decode/detector/tracker",
            "not_live_fps": True}


def _sha_inputs(root, samples):
    h = hashlib.sha256()
    for item in samples:
        h.update(item["rel"].encode("utf-8"))
        h.update(bytes.fromhex(hash_file(root / item["rel"])))
    return h.hexdigest()


def save_cache(output_dir, model_name, features, selection, weight_sha, image_sha, timing, cpu):
    folder = Path(output_dir)
    features = np.asarray(features, dtype=np.float32)
    expected = len(selection["query"]) + len(selection["gallery"])
    if features.ndim != 2 or features.shape[0] != expected or features.shape[1] < 1:
        raise ValueError("Invalid encoder feature shape")
    if not np.isfinite(features).all() or np.max(abs(np.linalg.norm(features,axis=1)-1)) > 1e-3:
        raise ValueError("Saved embeddings must be finite unit vectors")
    path = folder / f"{model_name}_features.npz"
    tmp = folder / f"{model_name}_features.part"
    with tmp.open("wb") as f:
        np.savez_compressed(f, features=features)
    tmp.replace(path)
    info = {"model": model_name, "checkpoint_sha256": weight_sha,
            "selection_sha256": selection["selection_sha256"], "images_sha256": image_sha,
            "features_sha256": hash_file(path), "feature_dimension": int(features.shape[1]),
            "crop_count": int(len(features)), "extract_wall_seconds_including_io": timing,
            "cpu_benchmark": cpu}
    dump(folder / f"{model_name}_manifest.json", info)
    return info


def load_cache(output_dir, model_name, selection, root, weights):
    folder = Path(output_dir)
    manifest = json.loads((folder / f"{model_name}_manifest.json").read_text(encoding="utf-8"))
    if manifest["checkpoint_sha256"] != hash_file(weights) or manifest["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("Checkpoint or selection differs from cached extraction")
    samples = selection["query"] + selection["gallery"]
    if manifest["images_sha256"] != _sha_inputs(root, samples):
        raise ValueError("Dataset image bytes changed; cache invalid")
    path = folder / f"{model_name}_features.npz"
    if hash_file(path) != manifest["features_sha256"]:
        raise ValueError("Embedding file corruption detected")
    with np.load(path, allow_pickle=False) as f:
        feats = np.asarray(f["features"],dtype=np.float32)
    if feats.shape != (len(samples), manifest["feature_dimension"]):
        raise ValueError("Cached feature shape changed")
    if not np.isfinite(feats).all() or np.max(abs(np.linalg.norm(feats,axis=1)-1)) > 1e-3:
        raise ValueError("Cached feature normalisation failed")
    return feats, manifest


def calculate_metrics(queries, gallery, qfeatures, gfeatures, gallery_camera=None):
    """Standard single-query rank/mAP with same-person/same-camera exclusion.

    rank is over a complete query-specific gallery of surviving detections;
    queries with no valid cross-camera positive are excluded and counted.
    """
    if len(queries) != len(qfeatures) or len(gallery) != len(gfeatures):
        raise ValueError("Query/gallery features not aligned")
    if not len(queries) or not len(gallery):
        raise ValueError("Empty query/gallery")
    g_pid = np.asarray([int(g["pid"]) for g in gallery], dtype=np.int32)
    g_cam = np.asarray([int(g["cam"]) for g in gallery], dtype=np.int32)
    if not np.isfinite(qfeatures).all() or not np.isfinite(gfeatures).all():
        raise ValueError("Non-finite embeddings")
    total = 0; skipped = 0; top1 = 0; top5 = 0; aps = []
    for query, feature in zip(queries, qfeatures):
        qpid = int(query["pid"]); qcam = int(query["cam"])
        # Gallery PID=0 remains a distractor, but PID=-1 is always junk.
        valid = (g_pid != -1) & ~((g_pid == qpid) & (g_cam == qcam))
        if gallery_camera is not None:
            valid &= (g_cam == gallery_camera)
        good = valid & (g_pid == qpid) & (g_cam != qcam)
        if not np.any(good):
            skipped += 1
            continue
        scores = gfeatures[valid] @ feature
        order = np.argsort(-scores, kind="stable")
        hit = good[valid][order]
        ranks = np.flatnonzero(hit)
        if not len(ranks):
            raise AssertionError("Valid query has no positive ranking")
        total += 1
        top1 += int(ranks[0] == 0)
        top5 += int(ranks[0] < 5)
        aps.append(float(np.mean(np.arange(1, len(ranks)+1)/(ranks+1))))
    return {"eligible_queries": total, "skipped_no_cross_camera_positive": skipped,
            "rank1": round(top1/total, 6) if total else None,
            "rank5": round(top5/total, 6) if total else None,
            "mAP": round(float(np.mean(aps)), 6) if total else None,
            "top1_correct": top1, "top5_correct": top5,
            "protocol": PROTOCOL}


def metrics_by_camera(queries, gallery, qfeatures, gfeatures):
    results = []
    for qcam in range(1,7):
        indices = [idx for idx,q in enumerate(queries) if q["cam"]==qcam]
        if not indices:
            continue
        for gcam in range(1,7):
            if qcam == gcam:
                continue
            d = calculate_metrics([queries[i] for i in indices],gallery,qfeatures[indices],gfeatures,gcam)
            results.append({"query_camera": qcam, "gallery_camera": gcam, **d})
    return results


def run(output_dir=OUT, base_weights=BASE_WEIGHTS, alt_weights=ALT_WEIGHTS,
        threads=4, affinity="0,2,4,6,8,10", batch_size=16, model_choice="both"):
    from .identity_evaluation import _configure_execution
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    selection, root = _load_selection(output_dir)
    actual_affinity, actual_threads = _configure_execution(threads, affinity)
    all_samples = selection["query"]+selection["gallery"]
    weights = {"osnet_x0_25": Path(base_weights), "osnet_x1_0": Path(alt_weights)}
    selected = list(weights) if model_choice == "both" else [model_choice]
    if model_choice not in ("both", *weights):
        raise ValueError("Unknown model choice")
    for name in selected:
        path = weights[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing pretrained model: {path}. See STEP14_26.md")
        if name == "osnet_x0_25" and hash_file(path) != BASE_SHA256:
            raise ValueError("Production OSNet x0.25 checkpoint SHA-256 differs; stop")
    hashes = {name: hash_file(weights[name]) for name in selected}
    input_hash = _sha_inputs(root, all_samples)
    print(f"Sample set: {len(selection['query'])} queries / {len(selection['gallery'])} gallery images; mode={selection['mode']}", flush=True)
    for name in selected:
        print(f"Loading {name}, weights sha256={hashes[name]}", flush=True)
        encoder = create_encoder(weights[name])
        # CPU-only fixed in-memory crops; both models use exactly same benchmark inputs.
        cpu = benchmark_cpu(encoder, root, all_samples, batch_size=batch_size)
        features, duration = extract(encoder, root, all_samples, batch_size, name)
        import platform
        import torch
        try:
            from importlib.metadata import version
            boxmot_version = version("boxmot")
        except Exception:
            boxmot_version = "unknown"
        cpu.update({"threads": actual_threads, "affinity": actual_affinity,
                    "boxmot_version": boxmot_version,
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "platform": platform.platform()})
        info = save_cache(output_dir, name, features, selection, hashes[name], input_hash, duration, cpu)
        print(f"{name}: {info['crop_count']} images, {duration}s wall, {cpu['median_ms_per_crop']} ms/crop (CPU encode only)", flush=True)
    return {"completed_models": selected, "selection_sha256": selection["selection_sha256"]}


def report(output_dir=OUT, base_weights=BASE_WEIGHTS, alt_weights=ALT_WEIGHTS):
    selection, root = _load_selection(output_dir)
    samples = selection["query"]+selection["gallery"]
    qn = len(selection["query"])
    models = {}
    rows = []
    for name, weight in (("osnet_x0_25",base_weights),("osnet_x1_0",alt_weights)):
        feat, manifest = load_cache(output_dir, name, selection, root, weight)
        scores = calculate_metrics(selection["query"], selection["gallery"], feat[:qn], feat[qn:])
        cameras = metrics_by_camera(selection["query"],selection["gallery"], feat[:qn], feat[qn:])
        models[name] = {"checkpoint_sha256":manifest["checkpoint_sha256"],
                        "feature_dim":manifest["feature_dimension"],
                        "retrieval": scores, "cpu":manifest["cpu_benchmark"],
                        "extract_wall_seconds_including_io":manifest["extract_wall_seconds_including_io"]}
        for line in cameras:
            rows.append({"model":name, **{k:v for k,v in line.items() if k != "protocol"}})
    if models["osnet_x1_0"]["retrieval"]["rank1"] is None:
        raise ValueError("No eligible queries for comparison")
    base, alt = models["osnet_x0_25"],models["osnet_x1_0"]
    comparison = {
        "experiment": "Step 14.26 Market-1501 pretrained OSNet encoder comparison; identity retrieval only",
        "dataset": {k: selection[k] for k in ("dataset_name","mode","official_protocol","protocol","selection_seed","selection_sha256","counts")},
        "models":models,
        "delta_x1_minus_x025": {m: round(alt["retrieval"][m]-base["retrieval"][m],6) for m in ("rank1","rank5","mAP")},
        "relative_cpu_cost_x1_over_x025": round(alt["cpu"]["median_ms_per_crop"]/base["cpu"]["median_ms_per_crop"], 4),
        "recommendation": "No model selected automatically. Quality and CPU budgets require new independent video+stateful identity validation.",
        "limitations": ["Pilot is a deterministic sampled study, NOT an official Market-1501 result.",
                        "Market-1501 images are separate from WILDTRACK and use a query/gallery protocol; the latter necessarily share ground-truth person IDs.",
                        "Models labelled MSMT17 may nonetheless have unknown provenance; independently verify training data and licence.",
                        "No BoxMOT association, no YOLO/ByteTrack, no live latency or 20FPS demonstration, no IDF1/HOTA.",
                        "CPU encode ms/crop is not aggregate four-camera pipeline FPS.",
                        "Both models use same published dataset test split; do not tune or select based on test then call it blind validation."]}
    dump(Path(output_dir)/"comparison.json", comparison)
    with (Path(output_dir)/"camera_pair_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        fields = ["model","query_camera","gallery_camera","eligible_queries","skipped_no_cross_camera_positive","rank1","rank5","mAP","top1_correct","top5_correct"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    dump(Path(output_dir)/"latency.json", {name: models[name]["cpu"] for name in models})
    return comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare",help="Index an extracted Market-1501 query/gallery dataset")
    a.add_argument("--dataset-dir",type=Path,default=DATASET)
    a.add_argument("--output-dir",type=Path,default=OUT)
    a.add_argument("--mode",choices=["pilot","full"],default="pilot")
    a.add_argument("--max-queries",type=int,default=150)
    a.add_argument("--gallery-per-pid-camera",type=int,default=2)
    a.add_argument("--seed",type=int,default=1426)
    b=sub.add_parser("run",help="Run both pretrained encoders on CPU; caches separately")
    b.add_argument("--output-dir",type=Path,default=OUT)
    b.add_argument("--base-weights",type=Path,default=BASE_WEIGHTS)
    b.add_argument("--alt-weights",type=Path,default=ALT_WEIGHTS)
    b.add_argument("--threads",type=int,default=4)
    b.add_argument("--affinity",default="0,2,4,6,8,10")
    b.add_argument("--batch-size",type=int,default=16)
    b.add_argument("--model",choices=["both","osnet_x0_25","osnet_x1_0"],default="both")
    c=sub.add_parser("report",help="Verify caches and score identically")
    c.add_argument("--output-dir",type=Path,default=OUT)
    c.add_argument("--base-weights",type=Path,default=BASE_WEIGHTS)
    c.add_argument("--alt-weights",type=Path,default=ALT_WEIGHTS)
    args=p.parse_args()
    if args.command=="prepare":
        value=prepare(args.dataset_dir,args.output_dir,args.mode,args.max_queries,args.gallery_per_pid_camera,args.seed)
        print(f"Prepared {value['counts']['query']} queries and {value['counts']['gallery']} gallery crops")
    elif args.command=="run":
        print(run(args.output_dir,args.base_weights,args.alt_weights,args.threads,args.affinity,args.batch_size,args.model))
    else:
        value=report(args.output_dir,args.base_weights,args.alt_weights)
        for name,v in value["models"].items():
            print(f"{name}: Rank-1={v['retrieval']['rank1']}, mAP={v['retrieval']['mAP']}, CPU ms/crop={v['cpu']['median_ms_per_crop']}")


if __name__=="__main__":
    main()
