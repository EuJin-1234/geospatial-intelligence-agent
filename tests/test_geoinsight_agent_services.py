from __future__ import annotations

from geoinsight.schemas import QueryResponse

from geoinsight.agent.workflow import create_geoinsight_workflow
from geoinsight.llm.ollama_provider import OllamaProvider
from geoinsight.schemas import QueryOptions
from geoinsight.services import query_service


def test_service_query_function_uses_options(monkeypatch):
    def fake_query_without_graph(request):
        return QueryResponse(query=request.query, answer="ok", retrieved_places=[])

    monkeypatch.setattr(query_service, "query_without_graph", fake_query_without_graph)

    response = query_service.run_agent_query("Find a cafe", QueryOptions(use_graph=False))

    assert response.answer == "ok"


def test_geoinsight_workflow_routes_data_question(monkeypatch):
    monkeypatch.setattr(
        "geoinsight.agent.nodes.get_eda_summary_tool",
        lambda: {"dataset_overview": {"cleaned_places": 3, "category_counts": {"cafe": 2}}},
    )
    workflow = create_geoinsight_workflow()

    final = workflow.invoke({"query": "What are the most common amenity categories?"})

    assert final["task_type"] == "data_question"
    assert "3" in final["answer"]


def test_ollama_provider_abstraction_accepts_mock_client():
    class MockClient:
        def generate(self, prompt, retrieved_places, context):
            class Result:
                answer = "mock answer"

            return Result()

    provider = OllamaProvider(client=MockClient())

    assert provider.generate("hello") == "mock answer"
