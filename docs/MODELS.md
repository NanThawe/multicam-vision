# Local pretrained checkpoints (not committed)

Place the following files here before inference:

- `yolo26n.pt` — YOLO detector checkpoint used in the existing pipeline.
- `osnet_x0_25_msmt17.pt` — default pretrained OSNet appearance encoder; expected SHA-256 `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- `osnet_x1_0_msmt17.pt` — **optional**, used only for the encoder pilot benchmark; not required for deployment.

The code in `src/reid.py` documents the x0.25 checkpoint source and verifies the checksum; Ultralytics may download its configured detector weights when allowed.

Do not commit third-party weights without verifying their redistributability and license. Model files are intentionally excluded by `.gitignore`; they are **not** meant to be deleted from your working Windows project.
