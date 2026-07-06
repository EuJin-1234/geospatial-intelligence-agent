from __future__ import annotations

from geoinsight.processing.spatial_relations import relation_summary
from geoinsight.schemas import QueryContext, RetrievedPlace

SYSTEM_RULES = """You are a spatial intelligence assistant.
Use only the retrieved geospatial context.
Do not invent facts.
Do not mention ratings, opening hours, crowd levels, or reviews unless provided.
The retrieved places are already ranked by the retrieval system.
Use the first place as the best recommendation unless it is explicitly marked as weak."""


def build_grounded_prompt(query: str, retrieved_places: list[RetrievedPlace], context: QueryContext | None = None, no_strong_match: bool = False) -> str:
    best = retrieved_places[0].record.name if retrieved_places else "none"
    lines = []
    for idx, item in enumerate(retrieved_places, start=1):
        record = item.record
        lines.append("\n".join([f"{idx}. {record.name}", f"category: {record.category}", f"distance_m: {round(record.distance_to_origin_m)}", f"themes: {', '.join(record.themes) if record.themes else 'none'}", f"contextual_evidence: {', '.join(item.contextual_evidence) if item.contextual_evidence else 'none'}", f"spatial_relations_summary: {relation_summary(record) or 'none'}", f"combined_score: {item.combined_score}", f"match: {'strong' if item.is_strong_match else 'weak'}", f"match_reason: {item.match_reason}"]))
    return f"""{SYSTEM_RULES}

User query:
{query}

Query context:
intent={context.intent if context else None}
weather={context.weather if context else None}
time_of_day={context.time_of_day if context else None}

System-selected best recommendation:
{best}

Strong match status:
{"No strong match was found. The following are the closest weak matches." if no_strong_match else "Strong matches are available. Use the first retrieved place as the best recommendation."}

Retrieved geospatial context:
{chr(10).join(lines) if lines else "No retrieved places."}
"""