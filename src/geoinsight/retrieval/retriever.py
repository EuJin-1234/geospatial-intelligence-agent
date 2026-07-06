from __future__ import annotations

import logging
from dataclasses import dataclass

from geoinsight.processing.contextual_thematic import STRONG_INTENTS, infer_query_context, score_contextual_theme_match
from geoinsight.schemas import QueryContext, RetrievedPlace

LOGGER = logging.getLogger(__name__)
MIN_STRONG_RESULT_SCORE = 0.20


@dataclass(frozen=True)
class RetrievalWeights:
    semantic: float = 0.50
    spatial: float = 0.20
    theme: float = 0.15
    contextual: float = 0.15


class HybridRetriever:
    def __init__(self, vector_store, embedder, weights: RetrievalWeights | None = None):
        self.vector_store = vector_store; self.embedder = embedder; self.weights = weights or RetrievalWeights()

    def retrieve(self, query: str, top_k: int = 5, max_distance_m: int = 1500, requested_themes: list[str] | None = None, context: QueryContext | None = None) -> list[RetrievedPlace]:
        context = infer_query_context(query, context)
        return self.retrieve_with_embedding(self.embedder.embed_query(query), top_k, max_distance_m, requested_themes, context)

    def retrieve_with_embedding(self, query_embedding, top_k: int = 5, max_distance_m: int = 1500, requested_themes: list[str] | None = None, context: QueryContext | None = None) -> list[RetrievedPlace]:
        requested_themes = sorted({theme.lower() for theme in (requested_themes or [])})
        raw_results = self.vector_store.search(query_embedding, max(top_k * 5, top_k))
        named_alternatives = sum(1 for record, _ in raw_results if record.distance_to_origin_m <= max_distance_m and not _is_unnamed(record.name))
        scored: list[RetrievedPlace] = []
        for record, semantic_score in raw_results:
            if record.distance_to_origin_m > max_distance_m: continue
            spatial_score = 1 - min(record.distance_to_origin_m / max(max_distance_m, 1), 1)
            theme_score = len(set(requested_themes).intersection(record.themes)) / len(requested_themes) if requested_themes else 0.0
            contextual_score, contextual_evidence = score_contextual_theme_match(record, context) if context else (0.0, [])
            combined = self.weights.semantic * float(semantic_score) + self.weights.spatial * spatial_score + self.weights.theme * theme_score + self.weights.contextual * contextual_score
            combined = _apply_penalties(combined, record.category, record.name, theme_score, contextual_score, context.intent if context else None, named_alternatives, top_k)
            scored.append(RetrievedPlace(record=record, semantic_score=round(float(semantic_score), 4), spatial_score=round(spatial_score, 4), theme_score=round(theme_score, 4), contextual_score=round(contextual_score, 4), contextual_evidence=contextual_evidence, combined_score=round(combined, 4)))
        return label_match_strength(sorted(scored, key=lambda item: item.combined_score, reverse=True)[:top_k], context)


def label_match_strength(results: list[RetrievedPlace], context: QueryContext | None = None, min_strong_result_score: float = MIN_STRONG_RESULT_SCORE) -> list[RetrievedPlace]:
    intent = context.intent if context else None
    return [item.model_copy(update={"is_strong_match": _match_strength(item, intent, min_strong_result_score)[0], "match_reason": _match_strength(item, intent, min_strong_result_score)[1]}) for item in results]


def _match_strength(item: RetrievedPlace, intent: str | None, min_strong_result_score: float) -> tuple[bool, str]:
    if item.record.category in {"bicycle_parking", "parking"} and intent != "transport": return False, "Weak match: parking-related category not relevant unless transport intent is requested."
    if item.combined_score < min_strong_result_score: return False, "Weak match: combined retrieval score is below the strong-match threshold."
    if intent in STRONG_INTENTS and item.contextual_score < 0.2: return False, f"Weak match: low contextual relevance for {intent} intent."
    return True, f"Strong match: matches {intent} intent with relevant themes or context." if intent else "Strong match: high combined semantic and spatial retrieval score."


def _apply_penalties(combined_score: float, category: str, name: str, theme_score: float, contextual_score: float, intent: str | None, named_alternatives: int, top_k: int) -> float:
    if intent in STRONG_INTENTS and theme_score == 0 and contextual_score < 0.2: combined_score *= 0.45
    if category in {"bicycle_parking", "parking"} and intent != "transport": combined_score *= 0.35
    if _is_unnamed(name) and named_alternatives >= top_k: combined_score *= 0.75
    return combined_score


def _is_unnamed(name: str) -> bool:
    return name.lower().startswith("unnamed")