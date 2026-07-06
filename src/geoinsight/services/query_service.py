from __future__ import annotations

import json
import time

from geoinsight.config import DEFAULT_LAT, DEFAULT_LON
from geoinsight.pipeline.query_pipeline import make_request, query_with_graph, query_without_graph
from geoinsight.processing.contextual_thematic import infer_query_context
from geoinsight.schemas import PlaceRecord, QueryContext, QueryResponse, RetrievedPlace
from geoinsight.visualisation.map_builder import FoliumMapBuilder

from geoinsight.config import load_config
from geoinsight.llm.ollama_provider import get_llm_provider
from geoinsight.retrieval.intent_parser import parse_intent
from geoinsight.schemas import QueryOptions


VALID_INTENTS = {"study", "relax", "food", "transport", "social", "coffee", "outdoor"}
VALID_WEATHER = {"clear", "rainy", "cold", "hot"}
VALID_TIMES = {"morning", "afternoon", "evening", "night"}


def run_agent_query(
    query: str,
    options: QueryOptions | None = None,
    intent: str | None = None,
    weather: str | None = None,
    time_of_day: str | None = None,
    top_k: int = 5,
    max_distance_m: int = 1500,
    generate_map: bool = False,
    use_agent: bool = True,
) -> QueryResponse:
    if options is None:
        options = QueryOptions(
            top_k=top_k,
            max_distance_m=max_distance_m,
            map_requested=generate_map,
            use_graph=use_agent,
            context=_query_context(intent, weather, time_of_day),
        )
    config = load_config()
    if config.llm_provider == "template":
        return _run_template_query(query, options)
    if config.llm_provider == "azure":
        provider = get_llm_provider()
        raise NotImplementedError(provider.generate(query))
    context = infer_query_context(query, options.context)
    request = make_request(
        query=query,
        top_k=options.top_k,
        max_distance_m=options.max_distance_m,
        themes=options.requested_themes,
        map_requested=options.map_requested,
        context=context,
    )
    return query_with_graph(request) if options.use_graph else query_without_graph(request)


def _run_template_query(query: str, options: QueryOptions) -> QueryResponse:
    config = load_config()
    started = time.perf_counter()
    context = infer_query_context(query, options.context)
    records = _load_index_records()
    retrieved = _template_retrieve(
        query=query,
        records=records,
        top_k=options.top_k,
        max_distance_m=options.max_distance_m,
        requested_themes=options.requested_themes,
        context=context,
    )
    provider = get_llm_provider()
    answer = provider.generate(query, retrieved_places=retrieved, context=context)
    map_path = None
    if options.map_requested and retrieved:
        map_path = str(FoliumMapBuilder(config.legacy).build_map(DEFAULT_LAT, DEFAULT_LON, retrieved))
    return QueryResponse(
        query=query,
        answer=answer,
        retrieved_places=retrieved,
        context=context,
        map_path=map_path,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        fallback_used=True,
    )


def _load_index_records() -> list[PlaceRecord]:
    config = load_config()
    if not config.legacy.metadata_path.exists():
        raise FileNotFoundError(
            "Vector metadata not found. Run `python -m geoinsight.cli build-index` first."
        )
    return [
        PlaceRecord.model_validate(item)
        for item in json.loads(config.legacy.metadata_path.read_text(encoding="utf-8"))
    ]


def _template_retrieve(
    query: str,
    records: list[PlaceRecord],
    top_k: int,
    max_distance_m: int,
    requested_themes: list[str],
    context: QueryContext | None,
) -> list[RetrievedPlace]:
    parsed = parse_intent(query)
    themes = set(requested_themes or parsed.requested_themes)
    if context and context.intent:
        themes.add(context.intent)
    terms = {term.strip(".,?!:;()[]").lower() for term in query.split()}
    scored: list[RetrievedPlace] = []
    for record in records:
        if record.distance_to_origin_m > max_distance_m:
            continue
        record_terms = {
            record.name.lower(),
            record.category.lower(),
            *(theme.lower() for theme in record.themes),
        }
        record_text = " ".join(
            [record.name, record.category, record.subcategory or "", record.llm_description]
        ).lower()
        theme_hits = themes.intersection(set(record.themes))
        lexical_hits = sum(1 for term in terms if len(term) > 2 and term in record_text)
        semantic_score = min(1.0, 0.15 * lexical_hits + 0.2 * len(theme_hits))
        spatial_score = 1 - min(record.distance_to_origin_m / max(max_distance_m, 1), 1)
        theme_score = len(theme_hits) / len(themes) if themes else 0.0
        contextual_score = 0.25 if context and context.intent in record.themes else 0.0
        combined_score = (
            0.35 * semantic_score
            + 0.3 * spatial_score
            + 0.25 * theme_score
            + 0.1 * contextual_score
        )
        if not lexical_hits and not theme_hits and record.category not in terms and record.name.lower() not in record_terms:
            combined_score *= 0.75
        evidence = []
        if theme_hits:
            evidence.append(f"matches themes: {', '.join(sorted(theme_hits))}")
        if lexical_hits:
            evidence.append(f"matches {lexical_hits} query terms")
        if spatial_score > 0.7:
            evidence.append("close to the origin")
        scored.append(
            RetrievedPlace(
                record=record,
                semantic_score=round(semantic_score, 4),
                spatial_score=round(spatial_score, 4),
                theme_score=round(theme_score, 4),
                contextual_score=round(contextual_score, 4),
                contextual_evidence=evidence,
                combined_score=round(combined_score, 4),
                is_strong_match=combined_score >= 0.2,
                match_reason="; ".join(evidence) or "Template retrieval candidate from saved metadata.",
            )
        )
    return sorted(scored, key=lambda item: item.combined_score, reverse=True)[:top_k]


def _query_context(
    intent: str | None = None,
    weather: str | None = None,
    time_of_day: str | None = None,
) -> QueryContext:
    return QueryContext(
        intent=_normalise_optional(intent, VALID_INTENTS),
        weather=_normalise_optional(weather, VALID_WEATHER),
        time_of_day=_normalise_optional(time_of_day, VALID_TIMES),
    )


def _normalise_optional(value: str | None, allowed: set[str]) -> str | None:
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised in {"", "none", "auto"}:
        return None
    return normalised if normalised in allowed else None
