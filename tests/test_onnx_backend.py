"""Small isolated tests that require no real model download or ONNX Runtime installation."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import reid
from src.compare_onnx import compare
from src.validate_onnx import embedding_agreement


class DummySpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Step11Tests(unittest.TestCase):
    def test_numeric_parity_identical(self):
        vectors = np.array([[1, 0, 0], [0, 0.5, 0.5]], dtype=np.float32)
        result = embedding_agreement(vectors, vectors)
        self.assertTrue(result["passed"])
        self.assertEqual(result["crops"], 2)

    def test_numeric_parity_detects_incorrect_embedding(self):
        x = np.array([[1, 0]], dtype=np.float32)
        y = np.array([[0, 1]], dtype=np.float32)
        self.assertFalse(embedding_agreement(x, y)["passed"])

    def test_numeric_parity_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            embedding_agreement(np.ones((2, 3)), np.ones((3, 3)))

    def test_encoder_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            reid.AppearanceEncoder(backend="tensorrt")

    def test_missing_onnx_message(self):
        fake_boxmot = types.ModuleType("boxmot")
        fake_components = types.ModuleType("boxmot.components")
        fake_artifacts = types.ModuleType("boxmot.components.artifacts")
        fake_artifacts.resolve_artifact = lambda *a, **kw: None
        fake_reid = types.ModuleType("boxmot.reid")
        fake_reid.ReIDEncoderSpec = DummySpec
        fake_reid.create_reid_encoder = lambda spec: spec
        modules = {
            "boxmot": fake_boxmot,
            "boxmot.components": fake_components,
            "boxmot.components.artifacts": fake_artifacts,
            "boxmot.reid": fake_reid,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules), \
                patch.object(reid, "ONNX_WEIGHTS", Path(tmp) / "missing.onnx"):
            with self.assertRaisesRegex(FileNotFoundError, "Export it first"):
                reid.AppearanceEncoder(backend="onnx")

    def test_onnx_model_spec_and_hash(self):
        fake_boxmot = types.ModuleType("boxmot")
        fake_components = types.ModuleType("boxmot.components")
        fake_artifacts = types.ModuleType("boxmot.components.artifacts")
        fake_artifacts.resolve_artifact = lambda *a, **kw: None
        fake_artifacts.sha256_artifact = lambda p: "fakedigest"
        fake_reid = types.ModuleType("boxmot.reid")
        fake_reid.ReIDEncoderSpec = DummySpec
        fake_reid.create_reid_encoder = lambda spec: spec
        fake_ort = types.ModuleType("onnxruntime")
        fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
        modules = {
            "boxmot": fake_boxmot, "boxmot.components": fake_components,
            "boxmot.components.artifacts": fake_artifacts,
            "boxmot.reid": fake_reid, "onnxruntime": fake_ort,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", modules):
            path = Path(tmp) / "osnet.onnx"
            path.write_bytes(b"dummy")
            with patch.object(reid, "ONNX_WEIGHTS", path):
                encoder = reid.AppearanceEncoder(backend="onnx")
            self.assertEqual(encoder.model.backend, "onnx")
            self.assertEqual(encoder.model.precision, "fp32")
            self.assertEqual(encoder.model.artifact_sha256, "fakedigest")
            self.assertEqual(encoder.model.artifact, str(path))

    def test_comparison_fps_and_workloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backend, fps in (("pytorch", 8), ("onnx", 12)):
                folder = root / f"{backend}_1"
                folder.mkdir()
                obj = {
                    "configuration": {"reid_backend": backend,
                                      "reid_batch_across_cameras": True,
                                      "reid_interval": 5},
                    "measured_batches": 100,
                    "reid_workload": {"eligible_crops": 4081, "encoded_crops": 1188},
                    "aggregate_fps": fps,
                    "latency_ms_per_four_camera_batch": {
                        "reid": {"p50": 200}, "total": {"p95": 500}
                    },
                }
                (folder / "report.json").write_text(json.dumps(obj), encoding="utf-8")
            result = compare(root, 1)
            self.assertEqual(result["onnx_vs_pytorch_fps_ratio"], 1.5)


if __name__ == "__main__":
    unittest.main()
