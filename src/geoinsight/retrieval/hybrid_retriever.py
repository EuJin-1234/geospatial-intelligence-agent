from __future__ import annotations

from geoinsight.processing.contextual_thematic import score_contextual_theme_match
from geoinsight.schemas import PlaceRecord, QueryContext, RetrievedPlace

from geoinsight.config import load_config, load_json_config
from geoinsight.retrieval.intent_parser import parse_intent
from geoinsight.retrieval.reranker import rerank_places
from geoinsight.schemas import RankedPlace


DEFAULT_WEIGHTS = {
    "semantic": 0.4,
    "spatial": 0.2,
    "theme": 0.15,
    "accessibility": 0.1,
    "density": 0.05,
    "context": 0.1,
}


class HybridGeoRetriever:
    def __init__(self, vector_store, embedder=None, weights: dict[str, float] | None = None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.weights = weights or load_json_config(load_config().retrieval_weights_path, DEFAULT_WEIGHTS)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance_m: int = 1500,
        requested_themes: list[str] | None = None,
        context: QueryContext | None = None,
    ) -> list[RetrievedPlace]:
        parsed = parse_intent(query)
        themes = sorted(set(requested_themes or parsed.requested_themes))
        if parsed.distance_constraint_m:
            max_distance_m = min(max_distance_m, parsed.distance_constraint_m)
        if self.embedder is None:
            raw_results = [(record, 0.0) for record in getattr(self.vector_store, "records", [])]
        else:
            raw_results = self.vector_store.search(self.embedder.embed_query(query), max(top_k * 6, top_k))
        ranked = self.score_candidates(raw_results, max_distance_m, themes, context, parsed.intent)
        return [_to_retrieved(item) for item in rerank_places(ranked, parsed.intent, top_k)]

    def score_candidates(
        self,
        raw_results: list[tuple[PlaceRecord, float]],
        max_distance_m: int,
        requested_themes: list[str],
        context: QueryContext | None,
        intent: str | None,
    ) -> list[RankedPlace]:
        candidates = []
        requested = set(requested_themes)
        for record, semantic_score in raw_results:
            if record.distance_to_origin_m > max_distance_m:
                continue
            spatial_score = 1 - min(record.distance_to_origin_m / max(max_distance_m, 1), 1)
            theme_score = (
                len(requested.intersection(record.themes)) / len(requested) if requested else 0.0
            )
            context_score, evidence = score_contextual_theme_match(record, context) if context else (0.0, [])
            accessibility_score = _accessibility_score(record)
            density_score = min(len(record.nearby_feature_types) / 6, 1.0)
            combined = (
                self.weights.get("semantic", 0) * float(semantic_score)
                + self.weights.get("spatial", 0) * spatial_score
                + self.weights.get("theme", 0) * theme_score
                + self.weights.get("accessibility", 0) * accessibility_score
                + self.weights.get("density", 0) * density_score
                + self.weights.get("context", 0) * context_score
            )
            reasons = _reasons(record, theme_score, spatial_score, context_score, evidence, intent)
            candidates.append(
                RankedPlace(
                    record=record,
                    semantic_score=round(float(semantic_score), 4),
                    spatial_score=round(spatial_score, 4),
                    theme_score=round(theme_score, 4),
                    accessibility_score=round(accessibility_score, 4),
                    density_score=round(density_score, 4),
                    context_score=round(context_score, 4),
                    combined_score=round(combined, 4),
                    reasons=reasons,
                )
            )
        return candidates


def _accessibility_score(record: PlaceRecord) -> float:
    themes = set(record.themes) | set(record.nearby_feature_types)
    return min(
        1.0,
        0.25 * ("transport" in themes or "public_transport" in themes)
        + 0.25 * ("food" in themes or "cafe" in themes)
        + 0.25 * ("outdoor" in themes or "park" in themes)
        + 0.25 * ("study" in themes or "library" in themes),
    )


def _reasons(
    record: PlaceRecord,
    theme_score: float,
    spatial_score: float,
    context_score: float,
    evidence: list[str],
    intent: str | None,
) -> list[str]:
    reasons = []
    if theme_score:
        reasons.append("matches requested themes")
    if spatial_score >= 0.7:
        reasons.append("close to the requested origin")
    if context_score:
        reasons.extend(evidence)
    if intent and intent in record.themes:
        reasons.append(f"supports {intent} intent")
    if not reasons:
        reasons.append("candidate retained by semantic and spatial retrieval")
    return reasons


def _to_retrieved(item: RankedPlace) -> RetrievedPlace:
    return RetrievedPlace(
        record=item.record,
        semantic_score=item.semantic_score,
        spatial_score=item.spatial_score,
        theme_score=item.theme_score,
        contextual_score=item.context_score,
        contextual_evidence=item.reasons,
        combined_score=item.combined_score,
        is_strong_match=item.combined_score >= 0.2,
        match_reason="; ".join(item.reasons),
    )
