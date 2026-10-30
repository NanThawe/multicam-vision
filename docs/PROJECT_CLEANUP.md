# Clean the working project without losing your evidence

**Do not delete `models/`, `data/` or `outputs/` outright.** They are local runtime and research assets; excluding them from GitHub is not the same as removing them from your computer. In particular, the Step 14.28 source videos and JSONL logs are required to rebuild Step 14.29's recorded demo.

## Safest workflow (Windows CMD)

From your actual existing project root:

```bat
python scripts\project_cleanup.py audit
```

This writes an inventory to `outputs/release_audit.json`. Review which files are `SOURCE`, `OUTPUT_LOCAL`, `MODEL_LOCAL`, `DATA_LOCAL`, `MEDIA_OR_BINARY_LOCAL`, or `REVIEW`. It does **not** move or delete anything.

Then create a **separate publishable folder**:

```bat
python scripts\project_cleanup.py stage --destination outputs/release_candidate
cd outputs\release_candidate
python -m unittest discover -s tests -p "test_*.py" -q
python scripts\preflight_release.py
```

The `stage` command copies whitelisted source, tests, documentation and Compose files; it never copies checkpoints, footage, evaluation caches or generated outputs. It refuses to overwrite an existing staged directory. It writes `release_manifest.json` with SHA-256 hashes of copied files.

If code imports local files that are not on the allowlist, inspect the inventory and add them intentionally rather than copying all outputs. Compare local `src/` to the reconstructed release snapshot before publishing; your Windows checkout is the authoritative working copy.

## Recommended local organisation

```text
multicam-analytics/
  src/                         # maintained pipeline, experiment scripts
  tests/                       # automated tests
  scripts/                     # release audit, demo creation
  config/                      # local configuration templates
  docs/                        # architecture, experiments, evaluation
  models/                      # local pretrained weights; gitignored
  data/                        # local licensed input recordings; gitignored
  outputs/
    rtsp/                      # raw/per-run captured telemetry and preview
    evaluation/                # raw experiment runs, crops, embeddings
    demo/                      # recorded dashboards, exported data
    showcase/                  # locally generated comparison videos
    archive/                   # optional manually reviewed old outputs
    release_candidate/         # publishable source snapshot, not pushed directly
```

You can move old output runs into `outputs/archive/` **only after** saving a backup and ensuring no script uses their original paths. Some evaluation modules expect fixed paths. This tool deliberately performs no destructive cleanup.

## Model reproduction

The code references OSNet x0.25 at `models/osnet_x0_25_msmt17.pt` and YOLO at `models/yolo26n.pt`. OSNet's adapter records a SHA-256 and may resolve the checkpoint on first use. Weight files remain local because they are large and may have separate distribution terms. Link to their authorized sources and checksums in documentation; don't commit the actual files unless you verify redistribution rights. Git LFS is a storage mechanism, **not a license to redistribute weights**.

## Evidence publication

`docs/evidence/curated/` contains **aggregate-only** JSON copied from verified experiments. Preserve your detailed raw run JSON, frame-by-frame logs, and cached embeddings under local `outputs/` for reproducibility. Do not publish identifiable person images, raw CCTV footage, sensitive log fields, or third-party dataset crops without reviewing permissions.
