# Development showcase: what changed and what was measured

The project is designed as an engineering portfolio. **Do not present every experiment as an upgrade deployed to the final RTSP pipeline.** Use two categories of demonstration:

1. **Visual demo** (recorded): four cameras, confirmed local tracks, provisional display boxes, illustrative event zones/lines, MQTT timeline. The Step 14.29 dashboard can be regenerated locally from `outputs/rtsp/step14_28/overlay_mqtt_02` when its video and logs are available.
2. **Controlled evaluation before/after**: charts and tables comparing configurations on the same annotated inputs. These provide stronger evidence for technical changes than visual overlays alone.

## Three useful before/after narratives

| Story | Before | After | Scope and caveat |
|---|---|---|---|
| Cross-camera association | Production EMA: 29 true cross-camera pairs / 3,694 (0.79% recall) | **Undeployed** camera-balanced Hungarian: 167 / 3,694 (4.52% recall) | Same 60 later WILDTRACK annotated frame sets, oracle boxes/perfect local tracks, 0 false pairs observed in both; **not** live production association |
| Independent appearance encoder pilot | OSNet x0.25: Rank-1 54.00%, 5.92 ms/crop | OSNet x1.0: Rank-1 47.33%, 65.85 ms/crop | Pilot with 150 sampled queries and reduced gallery; the larger model **regressed**, so x0.25 was retained |
| Sustained end-to-end validation | Target 20 aggregate FPS | Trials: 15.196, 16.683, 16.839 FPS | Three 180-second RTSP runs; **target failed**. MQTT subscriber independently received 927/927 generated events |

The first comparison is a genuine, fixed-input **experimental** before/after with an identity-quality improvement (although absolute recall remains low). The second is an ablation where the proposed bigger model loses. The third is a system-level reliability result rather than proof of optimisation.

## Recommended 90-second demonstration sequence

- **0–15 seconds:** four-camera recorded preview; introduce local multi-camera CPU pipeline and system architecture.
- **15–35 seconds:** confirmed versus provisional boxes; show event trigger and MQTT delivery in the local recorded dashboard.
- **35–50 seconds:** controlled cross-camera association results (29 → 167 correct pair observations), clearly labelled oracle-only and undeployed.
- **50–70 seconds:** CPU benchmarks (16.68 FPS median of three trial averages, target not met; 927/927 local MQTT delivery).
- **70–90 seconds:** discuss engineering trade-offs, reproducibility and failure analysis. Show why x0.25 was selected over x1.0 on the pilot.

## Optional side-by-side footage

If you have two **authorized, comparable** preview recordings from local experiment runs, create a side-by-side video:

```bat
python scripts\build_showcase_video.py --baseline "outputs\YOUR_BASELINE_RUN\provisional_preview.mp4" --experimental "outputs\YOUR_EXPERIMENTAL_RUN\provisional_preview.mp4" --output "outputs\showcase\side_by_side.mp4" --seconds 30
```

Use the same source clip, camera mapping, approximate playback duration and equal annotation/display conditions wherever possible. **Do not claim frame-perfect synchronization** between independent RTSP captures or infer accuracy from a visual side-by-side. If you only have the Step 14.28 final preview, create a one-video walkthrough first; do not fabricate a baseline.

The side-by-side generator writes only under the specified local output path and refuses to overwrite. The generated MP4 stays gitignored. To publish it, verify distribution rights first and use an authorised release/video-host link rather than committing third-party footage to the source repository.

## Public evidence versus local research outputs

- Publish: curated aggregate metrics, clearly qualified experimental tables, architecture diagram, demo walkthrough documentation, non-identifying synthetic or licensed screenshots.
- Keep local: WILDTRACK/Market-1501 images, full videos, raw per-person feature traces, embeddings, weights, verbose CCTV logs.
- Publish *links* to external weights/datasets where licenses allow, along with required filenames and checkpoint hashes.

`docs/evidence/curated/` contains compact data records suitable for plotting and sharing with an interviewer. Every card states the methodological limits.
