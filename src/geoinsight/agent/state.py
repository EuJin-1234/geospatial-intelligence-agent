from __future__ import annotations

from typing import Any, TypedDict


class GeoInsightAgentState(TypedDict, total=False):
    query: str
    parsed_intent: dict[str, Any]
    task_type: str
    context: dict[str, Any]
    retrieved_places: list[dict[str, Any]]
    area_stats: dict[str, Any]
    comparison_result: dict[str, Any]
    eda_summary: dict[str, Any]
    answer: str
    map_path: str | None
    errors: list[str]
    latency_ms: float | None
