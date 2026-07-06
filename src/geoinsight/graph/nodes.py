from __future__ import annotations

import time
from dataclasses import dataclass

from geoinsight.llm.ollama_client import OllamaClient, build_fallback_answer, select_places_for_llm
from geoinsight.processing.contextual_thematic import STRONG_INTENTS, infer_query_context
from geoinsight.schemas import QueryContext, RetrievedPlace

THEME_KEYWORDS = {"study": {"study", "work", "library"}, "quiet": {"quiet", "calm"}, "coffee": {"coffee", "cafe"}, "food": {"food", "eat", "lunch", "restaurant"}, "relax": {"relax", "rest"}, "transport": {"transport", "bus"}, "social": {"friend", "friends", "meet", "social"}, "outdoor": {"outdoor", "outside", "park"}, "shopping": {"shop", "shopping"}, "fitness": {"fitness", "sport", "gym"}, "campus": {"campus", "university"}}


@dataclass
class GeoQueryNodes:
    embedder: object | None = None
    retriever: object | None = None
    llm_client: OllamaClient | None = None
    map_builder: object | None = None

    def infer_context(self, state: dict) -> dict:
        context = infer_query_context(state["query"], _context_from_state(state))
        state["context"] = context.model_dump(mode="json", exclude_none=True)
        if not state.get("requested_themes"):
            terms = {term.strip(".,?!").lower() for term in state["query"].split()}
            inferred = [theme for theme, keywords in THEME_KEYWORDS.items() if terms.intersection(keywords)]
            if context.intent: inferred.append(context.intent)
            state["requested_themes"] = sorted(set(inferred))
        return state

    def parse_query(self, state: dict) -> dict:
        return self.infer_context(state)

    def embed_query(self, state: dict) -> dict:
        if self.embedder is None:
            state.setdefault("errors", []).append("No embedder configured"); state["query_embedding"] = None; return state
        state["query_embedding"] = self.embedder.embed_query(state["query"]).tolist()
        return state

    def retrieve_places(self, state: dict) -> dict:
        if self.retriever is None:
            state.setdefault("errors", []).append("No retriever configured")
            state["retrieved_places"] = []
            return state
        context = _context_from_state(state)
        if state.get("query_embedding") is not None and hasattr(self.retriever, "retrieve_with_embedding"):
            results = self.retriever.retrieve_with_embedding(
                query_embedding=state["query_embedding"],
                top_k=state["top_k"],
                max_distance_m=state["max_distance_m"],
                requested_themes=state.get("requested_themes", []),
                context=context,
            )
        else:
            results = self.retriever.retrieve(
                query=state["query"],
                top_k=state["top_k"],
                max_distance_m=state["max_distance_m"],
                requested_themes=state.get("requested_themes", []),
                context=context,
            )
        state["retrieved_places"] = [item.model_dump(mode="json") for item in results]
        return state

    def evaluate_retrieval(self, state: dict) -> dict:
        retrieved = _retrieved_from_state(state)
        context = _context_from_state(state)
        strong_intent = context is not None and context.intent in STRONG_INTENTS
        state["retrieval_quality"] = "weak" if (not retrieved or retrieved[0].combined_score < 0.35 or (strong_intent and retrieved[0].contextual_score < 0.2)) else "good"
        return state

    def generate_answer(self, state: dict) -> dict:
        started = time.perf_counter()
        result = (self.llm_client or OllamaClient()).generate(state["query"], _retrieved_from_state(state), _context_from_state(state))
        state["answer"] = result.answer; state["latency_ms"] = result.latency_ms or ((time.perf_counter() - started) * 1000); state["fallback_used"] = result.fallback_used
        return state

    def fallback_answer(self, state: dict) -> dict:
        places, no_strong = select_places_for_llm(_retrieved_from_state(state))
        state["answer"] = build_fallback_answer(state["query"], places, _context_from_state(state), no_strong)
        state["fallback_used"] = True
        return state

    def build_map(self, state: dict) -> dict:
        if state.get("map_requested") and self.map_builder is not None:
            state["map_path"] = str(self.map_builder.build_map(state["lat"], state["lon"], _retrieved_from_state(state)))
        return state


def _retrieved_from_state(state: dict) -> list[RetrievedPlace]:
    return [RetrievedPlace.model_validate(item) for item in state.get("retrieved_places", [])]


def _context_from_state(state: dict) -> QueryContext | None:
    raw = state.get("context")
    if isinstance(raw, QueryContext): return raw
    if isinstance(raw, dict) and raw: return QueryContext.model_validate(raw)
    return None