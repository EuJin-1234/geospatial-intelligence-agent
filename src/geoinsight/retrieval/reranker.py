from __future__ import annotations

from geoinsight.schemas import RankedPlace


def rerank_places(results: list[RankedPlace], intent: str | None = None, top_k: int = 5) -> list[RankedPlace]:
    seen_names: set[str] = set()
    reranked = []
    for result in results:
        score = result.combined_score
        reasons = list(result.reasons)
        name_key = result.record.name.strip().lower()
        if result.record.name.lower().startswith("unnamed"):
            score *= 0.88
            reasons.append("unnamed place is slightly penalised")
        if result.record.category in {"parking", "bicycle_parking"} and intent != "transport":
            score *= 0.55
            reasons.append("parking is only preferred for transport intent")
        if not result.record.themes:
            score *= 0.8
            reasons.append("limited thematic evidence")
        if name_key in seen_names:
            score *= 0.7
            reasons.append("duplicate name reduced for result diversity")
        seen_names.add(name_key)
        reranked.append(result.model_copy(update={"combined_score": round(score, 4), "reasons": reasons}))
    return sorted(reranked, key=lambda item: item.combined_score, reverse=True)[:top_k]
