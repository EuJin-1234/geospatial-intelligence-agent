from __future__ import annotations

from typing import TypedDict


class GeoQueryState(TypedDict):
    query: str
    lat: float
    lon: float
    top_k: int
    max_distance_m: int
    requested_themes: list[str]
    map_requested: bool
    context: dict
    query_embedding: list[float] | None
    retrieved_places: list[dict]
    retrieval_quality: str
    answer: str | None
    map_path: str | None
    latency_ms: float | None
    fallback_used: bool
    errors: list[str]