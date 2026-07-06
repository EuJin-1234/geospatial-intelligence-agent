from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from geoinsight.schemas import QueryResponse, RetrievedPlace
from geoinsight.api.app import create_app
from geoinsight.api.dependencies import get_config


def _client():
    return TestClient(create_app())


def test_health_returns_ok():
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_valid_structure():
    response = _client().get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ready", "not_ready"}
    assert isinstance(payload["checks"], dict)
    assert isinstance(payload["missing"], list)


def test_query_rejects_empty_query():
    response = _client().post("/query", json={"query": ""})

    assert response.status_code == 422


def test_query_returns_expected_answer_when_service_is_mocked(monkeypatch, make_record):
    from geoinsight.api import routes

    retrieved = RetrievedPlace(
        record=make_record(name="Demo Cafe", category="cafe"),
        semantic_score=0.9,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.4,
        combined_score=0.82,
        contextual_evidence=["study-friendly cafe"],
    )

    def fake_run_agent_query(*args, **kwargs):
        return QueryResponse(
            query="Find a cafe",
            answer="Demo Cafe is a good option.",
            retrieved_places=[retrieved],
            latency_ms=12.0,
        )

    monkeypatch.setattr(routes.query_service, "run_agent_query", fake_run_agent_query)
    response = _client().post("/query", json={"query": "Find a cafe", "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Demo Cafe is a good option."
    assert payload["retrieved_places"][0]["name"] == "Demo Cafe"


def test_eda_report_missing_returns_helpful_404(tmp_path):
    app = create_app()
    missing = tmp_path / "missing_eda.json"
    config = SimpleNamespace(
        eda_summary_json_path=missing,
        data_quality_report_path=tmp_path / "quality.json",
        place_features_json_path=tmp_path / "features.json",
        place_features_parquet_path=tmp_path / "features.parquet",
        legacy=SimpleNamespace(faiss_index_path=tmp_path / "places.faiss", metadata_path=tmp_path / "meta.json"),
    )
    app.dependency_overrides[get_config] = lambda: config

    response = TestClient(app).get("/reports/eda")

    assert response.status_code == 404
    assert "run-eda" in response.json()["detail"]


def test_feature_preview_respects_limit(monkeypatch):
    from geoinsight.api import routes

    def fake_preview(limit):
        return [{"place_id": str(index)} for index in range(limit)]

    monkeypatch.setattr(routes.data_service, "get_feature_preview", fake_preview)
    response = _client().get("/features/preview?limit=3")

    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_build_endpoint_returns_mocked_success(monkeypatch):
    from geoinsight.api import routes

    monkeypatch.setattr(
        routes.data_service,
        "run_eda_service",
        lambda: {
            "status": "ok",
            "path": "data/reports/eda_summary.json",
            "data": {"cleaned_places": 1},
            "message": "done",
        },
    )
    response = _client().post("/build/eda")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
