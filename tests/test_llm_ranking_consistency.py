from __future__ import annotations

from geoinsight.llm.ollama_client import build_fallback_answer, select_places_for_llm
from geoinsight.llm.prompts import build_grounded_prompt
from geoinsight.schemas import QueryContext, RetrievedPlace


def test_prompt_uses_top_ranked_result_as_system_selected_best(make_record):
    top = RetrievedPlace(
        record=make_record(name="Top Cafe", category="cafe", themes=["coffee", "study"]),
        semantic_score=0.9,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.8,
        combined_score=0.91,
        is_strong_match=True,
    )
    second = RetrievedPlace(
        record=make_record(name="Second Library", category="library", place_id="library"),
        semantic_score=0.8,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.8,
        combined_score=0.85,
        is_strong_match=True,
    )

    prompt = build_grounded_prompt(
        "Find a quiet cafe near campus where I can study",
        [top, second],
        QueryContext(intent="study"),
    )

    assert "System-selected best recommendation:\nTop Cafe" in prompt
    assert "Use the first place as the best recommendation" in prompt


def test_select_places_for_llm_excludes_weak_when_strong_exists(make_record):
    strong = RetrievedPlace(
        record=make_record(name="Strong Cafe", category="cafe"),
        semantic_score=0.8,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.5,
        combined_score=0.8,
        is_strong_match=True,
    )
    weak = RetrievedPlace(
        record=make_record(name="Weak Parking", category="parking", place_id="parking"),
        semantic_score=0.9,
        spatial_score=0.8,
        theme_score=0.0,
        contextual_score=0.0,
        combined_score=0.5,
        is_strong_match=False,
    )

    selected, no_strong = select_places_for_llm([strong, weak])

    assert selected == [strong]
    assert no_strong is False


def test_fallback_answer_keeps_best_result_first(make_record):
    best = RetrievedPlace(
        record=make_record(name="Ranked Best Cafe", category="cafe"),
        semantic_score=0.8,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.5,
        combined_score=0.8,
        is_strong_match=True,
    )

    answer = build_fallback_answer("study query", [best], QueryContext(intent="study"))

    assert "Best recommendation:\n- Ranked Best Cafe" in answer
    assert "Only one strong match was found" in answer
