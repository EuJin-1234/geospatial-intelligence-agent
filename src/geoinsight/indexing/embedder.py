from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("sentence-transformers is required for production embeddings") from exc
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except RuntimeError as exc:
            if self.device != "cpu" and "CUDA" in str(exc):
                LOGGER.warning("CUDA embedding failed; retrying on CPU: %s", exc)
                from sentence_transformers import SentenceTransformer

                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device="cpu")
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            else:
                raise
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class ONNXRuntimeEmbedder:
    """CPU ONNX Runtime embedder for edge-oriented retrieval demos.

    This is intentionally limited to embedding inference. LLM ONNX deployment is
    left as future work because autoregressive generation, KV cache handling,
    quantisation, and memory constraints are substantially more complex.
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"ONNX embedding model directory not found: {self.model_dir}")
        onnx_path = self._find_onnx_model(self.model_dir)
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "ONNX embedding inference requires `pip install -e .[edge]`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
            inputs = {name: encoded[name] for name in self.input_names if name in encoded}
            outputs = self.session.run(None, inputs)
            batches.append(_pool_onnx_outputs(outputs[0], encoded.get("attention_mask")))
        return _normalise_rows(np.vstack(batches).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @staticmethod
    def _find_onnx_model(model_dir: Path) -> Path:
        preferred = model_dir / "model.onnx"
        if preferred.exists():
            return preferred
        candidates = sorted(model_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx model file found in {model_dir}")
        return candidates[0]


class HashingEmbedder:
    """Deterministic lightweight fallback for tests and dependency-constrained demos."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)


def create_embedder(
    model_name: str,
    device: str = "cpu",
    backend: str = "sentence-transformer",
    onnx_model_dir: str | Path | None = None,
    allow_fallback: bool = True,
):
    backend = backend.lower().replace("_", "-")
    if backend in {"onnx", "onnxruntime", "ort"}:
        try:
            if onnx_model_dir is None:
                raise ValueError("onnx_model_dir is required for ONNX embedding inference")
            LOGGER.info("Using ONNX Runtime embedding backend from %s", onnx_model_dir)
            return ONNXRuntimeEmbedder(onnx_model_dir)
        except Exception as exc:
            if not allow_fallback:
                raise
            LOGGER.warning(
                "ONNX embedding backend unavailable; falling back to SentenceTransformer: %s",
                exc,
            )
    return SentenceTransformerEmbedder(model_name, device=device)


def export_sentence_transformer_to_onnx(model_name: str, output_dir: str | Path) -> Path:
    """Export the embedding transformer to ONNX for CPU ONNX Runtime inference."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Export requires `pip install -e .[edge]`.") from exc

    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def _pool_onnx_outputs(token_embeddings: np.ndarray, attention_mask: np.ndarray | None) -> np.ndarray:
    if token_embeddings.ndim == 2:
        return token_embeddings
    if attention_mask is None:
        return token_embeddings.mean(axis=1)
    mask = np.expand_dims(attention_mask.astype(np.float32), axis=-1)
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (values / norms).astype(np.float32)