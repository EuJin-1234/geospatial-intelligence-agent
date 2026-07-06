from __future__ import annotations

from geoinsight.schemas import QueryResponse


def evaluate_response(response: QueryResponse, latency_ms: float | None = None) -> dict:
    retrieved = response.retrieved_places
    strong = [item for item in retrieved if item.is_strong_match]
    weak = [item for item in retrieved if not item.is_strong_match]
    return {
        "latency_ms": latency_ms if latency_ms is not None else response.latency_ms,
        "strong_matches": len(strong),
        "answer_generated": bool(response.answer),
        "retrieval_coverage": len(retrieved),
        "weak_result_rate": round(len(weak) / len(retrieved), 3) if retrieved else 1.0,
        "grounded_in_retrieved_records": bool(retrieved and any(item.record.name in response.answer for item in retrieved)),
    }
