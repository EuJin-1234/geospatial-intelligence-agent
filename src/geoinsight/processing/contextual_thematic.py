from __future__ import annotations

from geoinsight.schemas import PlaceRecord, QueryContext

STRONG_INTENTS = {"study", "relax", "outdoor", "food", "coffee", "social"}


def infer_query_context(query: str, explicit_context: QueryContext | None = None) -> QueryContext:
    inferred = QueryContext(intent=_infer_intent(query))
    if explicit_context is None:
        return inferred
    return QueryContext(intent=explicit_context.intent or inferred.intent, weather=explicit_context.weather, time_of_day=explicit_context.time_of_day)


def score_contextual_theme_match(record: PlaceRecord, context: QueryContext) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    intent = context.intent
    if intent == "study":
        score += _boost(record.category in {"library", "university", "school"}, evidence, "study category")
        score += _boost(record.category == "cafe" and "study" in record.themes, evidence, "study-friendly cafe")
        score += _boost("study" in record.themes or "quiet" in record.themes, evidence, "study/quiet theme")
        score += _boost(_has_relation(record, "near_theme", "study"), evidence, "near study-related places")
        if record.category in {"parking", "bicycle_parking"}:
            score -= 0.35; evidence.append("parking is weak for study intent")
        if record.category in {"pub", "bar"}:
            score -= 0.25; evidence.append("pub/bar is weak for study intent")
    elif intent in {"relax", "outdoor"}:
        score += _boost(record.category in {"park", "garden"}, evidence, "green-space category")
        score += _boost("outdoor" in record.themes or "relax" in record.themes, evidence, "outdoor/relax theme")
        score += _boost("quiet" in record.themes, evidence, "quiet theme")
        score += _boost(_has_relation(record, "near_theme", "outdoor"), evidence, "near green space")
        if context.weather in {"rainy", "cold"}:
            if record.category in {"park", "garden"} or "outdoor" in record.themes:
                score -= 0.25; evidence.append(f"{context.weather} weather reduces outdoor suitability")
            elif record.category in {"library", "cafe", "university"}:
                score += 0.15; evidence.append(f"{context.weather} weather favours indoor alternative")
    elif intent in {"food", "coffee"}:
        score += _boost(record.category in {"cafe", "restaurant", "fast_food"}, evidence, "food/coffee category")
        score += _boost("food" in record.themes or "coffee" in record.themes, evidence, "food/coffee theme")
        score += _boost(_has_relation(record, "near_theme", "food"), evidence, "near food options")
        if intent == "coffee":
            score += _boost(record.category == "cafe" or "coffee" in record.themes, evidence, "coffee-specific match")
    elif intent == "transport":
        score += _boost(record.category in {"bus_station", "public_transport", "bicycle_parking"}, evidence, "transport category")
        score += _boost("transport" in record.themes, evidence, "transport theme")
        score += _boost(_has_relation(record, "near_theme", "transport"), evidence, "near transport relation")
    elif intent == "social":
        score += _boost(record.category in {"cafe", "restaurant", "fast_food", "pub", "bar"}, evidence, "social category")
        score += _boost("social" in record.themes or "food" in record.themes, evidence, "social/food theme")
    if context.time_of_day in {"evening", "night"} and (record.category in {"restaurant", "pub", "bar"} or {"social", "food"}.intersection(record.themes)):
        score += 0.15; evidence.append(f"{context.time_of_day} favours food/social places")
    return max(0.0, min(round(score, 4), 1.0)), evidence


def _infer_intent(query: str) -> str | None:
    text = query.lower()
    if any(token in text for token in ("study", "work", "quiet")): return "study"
    if any(token in text for token in ("relax", "park", "garden")): return "relax"
    if "outdoor" in text or "outside" in text: return "outdoor"
    if any(token in text for token in ("food", "eat", "restaurant", "dinner", "lunch")): return "food"
    if "coffee" in text or "cafe" in text: return "coffee"
    if any(token in text for token in ("bus", "train", "transport")): return "transport"
    if any(token in text for token in ("meet", "friend", "social")): return "social"
    return None


def _boost(condition: bool, evidence: list[str], label: str) -> float:
    if condition:
        evidence.append(label); return 0.25
    return 0.0


def _has_relation(record: PlaceRecord, relation: str, target: str) -> bool:
    return any(item.get("relation") == relation and item.get("target") == target for item in record.spatial_relations)