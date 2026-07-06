from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from geoinsight.config import LegacyAppConfig, ensure_legacy_data_dirs, load_legacy_config
from geoinsight.schemas import PlaceRecord

LOGGER = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, embedder=None, config: LegacyAppConfig | None = None, index_path: Path | None = None, metadata_path: Path | None = None):
        self.config = config or load_legacy_config()
        self.embedder = embedder
        self.index_path = index_path or self.config.faiss_index_path
        self.metadata_path = metadata_path or self.config.metadata_path
        self.index = None
        self.records: list[PlaceRecord] = []
        self._fallback_embeddings: np.ndarray | None = None

    def build_index(self, records: list[PlaceRecord]) -> None:
        if self.embedder is None: raise ValueError("An embedder is required to build the vector index")
        self.records = records
        embeddings = _normalise_rows(np.asarray(self.embedder.embed_texts([r.llm_description for r in records]), dtype=np.float32))
        try:
            import faiss
            self.index = faiss.IndexFlatIP(embeddings.shape[1]); self.index.add(embeddings); self._fallback_embeddings = None
        except ImportError:
            LOGGER.warning("faiss-cpu is unavailable; using NumPy fallback search")
            self.index = None; self._fallback_embeddings = embeddings

    def save(self) -> None:
        ensure_legacy_data_dirs(self.config)
        self.index_path.parent.mkdir(parents=True, exist_ok=True); self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(self.index_path))
        elif self._fallback_embeddings is not None:
            np.savez_compressed(str(self.index_path) + ".npz", embeddings=self._fallback_embeddings)
        self.metadata_path.write_text(json.dumps([r.model_dump(mode="json") for r in self.records], indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.metadata_path.exists(): raise FileNotFoundError(f"Missing vector metadata: {self.metadata_path}")
        self.records = [PlaceRecord.model_validate(item) for item in json.loads(self.metadata_path.read_text(encoding="utf-8"))]
        try:
            import faiss
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path)); self._fallback_embeddings = None; return
        except ImportError:
            pass
        fallback_path = Path(str(self.index_path) + ".npz")
        if fallback_path.exists():
            self._fallback_embeddings = np.load(fallback_path)["embeddings"].astype(np.float32); self.index = None; return
        raise FileNotFoundError(f"Missing vector index: {self.index_path}")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[PlaceRecord, float]]:
        if not self.records: return []
        query = _normalise_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        top_k = min(max(top_k, 1), len(self.records))
        if self.index is not None:
            scores, indices = self.index.search(query, top_k)
            return [(self.records[int(idx)], float(score)) for idx, score in zip(indices[0], scores[0], strict=False) if idx >= 0]
        if self._fallback_embeddings is None: raise ValueError("Vector store has not been built or loaded")
        scores = self._fallback_embeddings @ query[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.records[int(idx)], float(scores[int(idx)])) for idx in indices]


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True); norms[norms == 0] = 1
    return (values / norms).astype(np.float32)