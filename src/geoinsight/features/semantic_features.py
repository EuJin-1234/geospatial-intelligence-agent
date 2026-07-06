from __future__ import annotations

from geoinsight.schemas import PlaceRecord

from geoinsight.config import load_config, load_json_config


def build_semantic_features(records: list[PlaceRecord]) -> list[dict]:
    rules = load_json_config(load_config().theme_rules_path, {})
    return [
        {
            "place_id": record.place_id,
            "themes": record.themes,
            "theme_count": len(record.themes),
            "primary_theme": record.themes[0] if record.themes else None,
            "context_relevance_scores": _context_scores(record, rules),
            "llm_description": record.llm_description,
        }
        for record in records
    ]


def infer_themes_from_rules(category: str, text: str = "") -> list[str]:
    rules = load_json_config(load_config().theme_rules_path, {})
    text = text.lower()
    themes = []
    for theme, rule in rules.items():
        categories = set(rule.get("categories", [])) | set(rule.get("soft_categories", []))
        keywords = set(rule.get("keywords", []))
        if category in categories or any(keyword in text for keyword in keywords):
            themes.append(theme)
    return sorted(themes)


def _context_scores(record: PlaceRecord, rules: dict) -> dict[str, float]:
    values = " ".join(
        [
            record.name,
            record.category,
            record.subcategory or "",
            record.llm_description,
            " ".join(record.themes),
        ]
    ).lower()
    scores = {}
    for theme, rule in rules.items():
        score = 0.0
        if record.category in set(rule.get("categories", [])):
            score += 0.6
        if record.category in set(rule.get("soft_categories", [])):
            score += 0.3
        keyword_hits = sum(1 for keyword in rule.get("keywords", []) if keyword in values)
        score += min(0.4, keyword_hits * 0.1)
        scores[theme] = round(min(score, 1.0), 3)
    return scores
