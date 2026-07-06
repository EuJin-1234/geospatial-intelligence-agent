from __future__ import annotations

from geoinsight.pipeline import query_pipeline
from geoinsight.schemas import QueryResponse


def test_run_spatial_query_returns_query_response(monkeypatch):
    def fake_query_with_graph(request):
        return QueryResponse(
            query=request.query,
            answer="ok",
            retrieved_places=[],
            context=request.context,
            latency_ms=1.0,
        )

    monkeypatch.setattr(query_pipeline, "query_with_graph", fake_query_with_graph)

    response = query_pipeline.run_spatial_query(
        "Find transport nearby",
        intent="transport",
        weather="none",
        time_of_day="none",
        map_requested=False,
    )

    assert isinstance(response, QueryResponse)
    assert response.context.intent == "transport"
    assert response.context.weather is None
