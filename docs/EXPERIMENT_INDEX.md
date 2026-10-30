# Where to find the code and evidence

This repository retains the **entire research trail**, rather than deleting experiments or pretending all experimental changes shipped in production. The primary entry points and their purpose are:

| Layer | Main modules | What to show |
|---|---|---|
| Core detection/tracking | `src/detector.py`, `src/tracker.py`, `src/reid.py`, `src/live_rtsp.py` | Four RTSP streams, confirmed local tracks, tentative global IDs |
| Events and transport | `src/events.py`, `src/mqtt_events.py` | Zone and line events, JSONL, broker/subscriber verification |
| CPU optimisation | `src/fixed_inference.py`, `src/affinity_verify.py`, `src/profile_cpu.py`, `src/cpu_contention.py` | Controlled profiling and scheduling experiments |
| Identity experiments | `src/association_ablation.py`, `src/source_consistency.py`, `src/identity_validation.py` | Production vs undeployed association on same annotated inputs |
| Encoder experiments | `src/encoder_benchmark.py`, `src/crop_ablation.py` | Market-1501 pilot and WILDTRACK crop investigations |
| Reliability | `src/reliability_validation.py`, `src/end_to_end_validation.py` | Three 180-second trials and MQTT local delivery |
| Presentation | `src/demo_release.py`, `src/demo_assets/`, `scripts/build_showcase_video.py` | Recorded four-camera dashboard, optional side-by-side video |
| Repository hygiene | `scripts/project_cleanup.py`, `scripts/preflight_release.py` | Local file inventory and safe source staging |

Source-code experiments are not equivalent to production deployments; some have intentionally lower precision or higher CPU cost. Consult [SHOWCASE.md](SHOWCASE.md) for the measured narratives and [RESULTS.md](RESULTS.md) for limitations.

Original `outputs/` trial artifacts should remain on the author's computer; public `docs/evidence/curated` reports contain aggregate-only summaries.
