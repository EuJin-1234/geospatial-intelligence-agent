from __future__ import annotations

from geoinsight.processing.contextual_thematic import (
    infer_query_context,
    score_contextual_theme_match,
)
from geoinsight.schemas import QueryContext


def test_infer_query_context_and_explicit_override():
    inferred = infer_query_context("Find a quiet cafe where I can study")
    assert inferred.intent == "study"

    overridden = infer_query_context(
        "Find a quiet cafe where I can study",
        QueryContext(intent="social", weather="rainy", time_of_day="evening"),
    )
    assert overridden.intent == "social"
    assert overridden.weather == "rainy"
    assert overridden.time_of_day == "evening"


def test_contextual_theme_scoring_for_study_relation(make_record):
    record = make_record(
        name="Cafe 38",
        category="cafe",
        themes=["coffee", "food", "study"],
        spatial_relations=[
            {
                "relation": "near_theme",
                "target": "study",
                "target_category": "semantic_theme",
                "distance_m": 100,
                "evidence": "Nearby library/university feature within 100m",
            }
        ],
    )

    score, evidence = score_contextual_theme_match(record, QueryContext(intent="study"))

    assert score >= 0.5
    assert "study-friendly cafe" in evidence
    assert "near study-related places" in evidence


def test_rainy_weather_reduces_outdoor_score(make_record):
    park = make_record(name="Valley Garden", category="garden", themes=["outdoor", "relax"])

    clear_score, _ = score_contextual_theme_match(park, QueryContext(intent="relax", weather="clear"))
    rainy_score, evidence = score_contextual_theme_match(
        park, QueryContext(intent="relax", weather="rainy")
    )

    assert rainy_score < clear_score
    assert "rainy weather reduces outdoor suitability" in evidence
