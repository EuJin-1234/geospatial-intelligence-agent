from __future__ import annotations

from fastapi.testclient import TestClient

from geoinsight.schemas import QueryResponse, RetrievedPlace
from geoinsight.api.app import create_app
from geoinsight.services import query_service


def test_ready_includes_environment_and_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "template")
    monkeypatch.setenv("ENVIRONMENT", "production")

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["environment"] == "production"
    assert payload["llm_provider"] == "template"
    assert "llm_provider_available" in payload["checks"]
    assert payload["checks"]["llm_provider_available"] is True


def test_template_query_service_uses_saved_records_without_ollama(monkeypatch, make_record):
    monkeypatch.setenv("LLM_PROVIDER", "template")
    records = [
        make_record(name="Template Cafe", category="cafe", themes=["food", "study"]),
        make_record(name="Template Park", category="park", themes=["outdoor"]),
    ]
    monkeypatch.setattr(query_service, "_load_index_records", lambda: records)

    response = query_service.run_agent_query(
        "Find study food",
        top_k=1,
        generate_map=False,
        use_agent=False,
    )

    assert response.fallback_used is True
    assert response.retrieved_places[0].record.name == "Template Cafe"
    assert "LLM provider is disabled" in response.answer


def test_api_query_can_return_template_fallback(monkeypatch, make_record):
    from geoinsight.api import routes

    monkeypatch.setenv("LLM_PROVIDER", "template")
    retrieved = RetrievedPlace(
        record=make_record(name="Template Cafe", category="cafe", themes=["food"]),
        semantic_score=0.5,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.0,
        combined_score=0.7,
        contextual_evidence=["matches food"],
    )

    def fake_run_agent_query(*args, **kwargs):
        return QueryResponse(
            query="Find food",
            answer="LLM provider is disabled in this deployment.",
            retrieved_places=[retrieved],
            fallback_used=True,
        )

    monkeypatch.setattr(routes.query_service, "run_agent_query", fake_run_agent_query)
    response = TestClient(create_app()).post("/query", json={"query": "Find food"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "template"
    assert payload["warnings"]
