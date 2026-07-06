from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from geoinsight.schemas import PlaceRecord

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_record(
    name: str = "Hartley Library",
    category: str = "library",
    distance: float = 300,
    themes: list[str] | None = None,
    place_id: str = "place-1",
    latitude: float = 50.935,
    longitude: float = -1.396,
    spatial_relations: list[dict] | None = None,
) -> PlaceRecord:
    return PlaceRecord(
        place_id=place_id,
        name=name,
        category=category,
        subcategory=None,
        latitude=latitude,
        longitude=longitude,
        distance_to_origin_m=distance,
        walking_distance_estimate_m=distance * 1.25,
        nearby_feature_types=["cafe", "public_transport"],
        themes=themes or ["campus", "quiet", "study"],
        spatial_relations=spatial_relations or [],
        spatial_context="within walking distance of campus; near public transport",
        llm_description=f"{name} is a {category} near campus.",
        metadata={"amenity": category},
    )


class FakeEmbedder:
    def embed_query(self, text: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_texts(self, texts: list[str], batch_size: int = 32):
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def search(self, query_embedding, top_k: int):
        return self.results[:top_k]


@pytest.fixture
def make_record():
    return _make_record


@pytest.fixture(name="FakeEmbedder")
def fake_embedder_fixture():
    return FakeEmbedder


@pytest.fixture(name="FakeVectorStore")
def fake_vector_store_fixture():
    return FakeVectorStore
