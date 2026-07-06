from __future__ import annotations

import re

from geoinsight.config import load_config, load_json_config
from geoinsight.schemas import ParsedIntent


TASK_PATTERNS = {
    "area_comparison": ("compare", "better for", "versus", " vs "),
    "area_summary": ("summarise", "summarize", "local area", "around this coordinate"),
    "data_question": ("most common", "statistics", "eda", "categories", "how many"),
    "map_request": ("show", "map", "visualise", "visualize"),
}


def parse_intent(query: str) -> ParsedIntent:
    text = query.lower()
    rules = load_json_config(load_config().theme_rules_path, {})
    requested_themes = []
    for theme, rule in rules.items():
        categories = set(rule.get("categories", [])) | set(rule.get("soft_categories", []))
        keywords = set(rule.get("keywords", []))
        if any(token in text for token in keywords | categories):
            requested_themes.append(theme)

    distance_constraint = _parse_distance(text)
    category_preference = _category_preference(text)
    task_type = _task_type(text)
    intent = requested_themes[0] if requested_themes else None
    return ParsedIntent(
        intent=intent,
        requested_themes=sorted(set(requested_themes)),
        distance_constraint_m=distance_constraint,
        category_preference=category_preference,
        task_type=task_type,
    )


def _parse_distance(text: str) -> int | None:
    match = re.search(r"(?:within|under|less than|nearby|walk(?:ing)? distance).*?(\d+)\s*(m|metres|meters|km)", text)
    if not match:
        return None
    value = int(match.group(1))
    return value * 1000 if match.group(2) == "km" else value


def _category_preference(text: str) -> str | None:
    for category in ("cafe", "library", "restaurant", "park", "garden", "bus", "transport"):
        if category in text:
            return "public_transport" if category in {"bus", "transport"} else category
    return None


def _task_type(text: str) -> str:
    for task_type, patterns in TASK_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return task_type
    return "place_recommendation"
