from __future__ import annotations

import tomllib

import numpy as np

from geoinsight.config import load_config
from geoinsight.indexing import embedder


class FakeSentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device


def test_pyproject_declares_optional_edge_dependencies():
    with open("pyproject.toml", "rb") as handle:
        pyproject = tomllib.load(handle)

    edge = pyproject["project"]["optional-dependencies"]["edge"]

    assert "onnxruntime" in edge
    assert "optimum[onnxruntime]" in edge


def test_create_embedder_defaults_to_sentence_transformer(monkeypatch):
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", FakeSentenceTransformerEmbedder)

    created = embedder.create_embedder("demo-model", device="cpu")

    assert isinstance(created, FakeSentenceTransformerEmbedder)
    assert created.model_name == "demo-model"


def test_create_embedder_falls_back_when_onnx_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", FakeSentenceTransformerEmbedder)

    created = embedder.create_embedder(
        "demo-model",
        backend="onnx",
        onnx_model_dir=tmp_path / "missing-onnx-model",
    )

    assert isinstance(created, FakeSentenceTransformerEmbedder)


def test_onnx_pooling_normalises_masked_tokens():
    token_embeddings = np.array(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
        ],
        dtype=np.float32,
    )
    attention_mask = np.array([[1, 1, 0]], dtype=np.int64)

    pooled = embedder._pool_onnx_outputs(token_embeddings, attention_mask)

    assert np.allclose(pooled, np.array([[2.0, 2.0]], dtype=np.float32))


def test_config_reads_optional_embedding_backend_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "onnx")
    monkeypatch.setenv("ONNX_EMBEDDING_MODEL_DIR", "data/index/test_onnx")

    config = load_config()

    assert config.legacy.embedding_backend == "onnx"
    assert str(config.legacy.onnx_embedding_model_dir).endswith("data\\index\\test_onnx") or str(
        config.legacy.onnx_embedding_model_dir
    ).endswith("data/index/test_onnx")