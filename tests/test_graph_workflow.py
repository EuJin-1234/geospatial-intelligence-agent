from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geoinsight.graph.nodes import GeoQueryNodes
from geoinsight.graph.workflow import create_geo_query_workflow
from geoinsight.llm.ollama_client import LLMResult, build_fallback_answer


def _state(query: str = "Find a quiet cafe near campus where I can study", map_requested: bool = False):
    return {
        "query": query,
        "lat": 50.935,
        "lon": -1.396,
        "top_k": 5,
        "max_distance_m": 1500,
        "requested_themes": [],
        "map_requested": map_requested,
        "context": {},
        "query_embedding": None,
        "retrieved_places": [],
        "retrieval_quality": "",
        "answer": None,
        "map_path": None,
        "latency_ms": None,
        "fallback_used": False,
        "errors": [],
    }


class GoodRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve_with_embedding(self, **kwargs):
        return [self.result]


class EmptyRetriever:
    def retrieve_with_embedding(self, **kwargs):
        return []


class FakeLLM:
    def generate(self, query, retrieved_places, context=None):
        return LLMResult("grounded answer", 12.0, False)


@dataclass
class FakeMapBuilder:
    path: Path

    def build_map(self, lat, lon, retrieved_places):
        return self.path


def test_fallback_answer_when_no_places():
    answer = build_fallback_answer("query", [])
    assert "Retrieved spatial context was limited" in answer


def test_workflow_routes_to_fallback_when_retrieval_is_weak(FakeEmbedder):
    nodes = GeoQueryNodes(embedder=FakeEmbedder(), retriever=EmptyRetriever(), llm_client=FakeLLM())
    workflow = create_geo_query_workflow(nodes)

    final = workflow.invoke(_state())

    assert final["retrieval_quality"] == "weak"
    assert final["fallback_used"] is True
    assert final["answer"]


def test_workflow_routes_to_answer_when_retrieval_is_good(make_record, FakeEmbedder):
    record = make_record(name="Good Cafe", category="cafe", themes=["coffee", "study"])
    from geoinsight.schemas import RetrievedPlace

    result = RetrievedPlace(
        record=record,
        semantic_score=0.9,
        spatial_score=0.9,
        theme_score=1.0,
        contextual_score=0.6,
        contextual_evidence=["study-friendly cafe"],
        combined_score=0.91,
    )
    nodes = GeoQueryNodes(embedder=FakeEmbedder(), retriever=GoodRetriever(result), llm_client=FakeLLM())
    workflow = create_geo_query_workflow(nodes)

    final = workflow.invoke(_state())

    assert final["retrieval_quality"] == "good"
    assert final["fallback_used"] is False
    assert final["answer"] == "grounded answer"
    assert final["retrieved_places"]
    assert final["context"]["intent"] == "study"


def test_workflow_returns_optional_map_path(make_record, FakeEmbedder, tmp_path):
    from geoinsight.schemas import RetrievedPlace

    result = RetrievedPlace(
        record=make_record(name="Mapped Cafe", category="cafe", themes=["coffee"]),
        semantic_score=0.9,
        spatial_score=0.8,
        theme_score=1.0,
        contextual_score=0.6,
        contextual_evidence=["coffee-specific match"],
        combined_score=0.88,
    )
    map_path = tmp_path / "map.html"
    nodes = GeoQueryNodes(
        embedder=FakeEmbedder(),
        retriever=GoodRetriever(result),
        llm_client=FakeLLM(),
        map_builder=FakeMapBuilder(map_path),
    )
    workflow = create_geo_query_workflow(nodes)

    final = workflow.invoke(_state(map_requested=True))

    assert final["map_path"] == str(map_path)
    assert final["retrieved_places"]
    assert final["fallback_used"] is False


def test_workflow_marks_strong_intent_result_weak_when_contextual_score_is_low(
    make_record, FakeEmbedder
):
    from geoinsight.schemas import RetrievedPlace

    result = RetrievedPlace(
        record=make_record(name="Weak Cafe", category="cafe", themes=["coffee"]),
        semantic_score=0.95,
        spatial_score=0.9,
        theme_score=1.0,
        contextual_score=0.0,
        combined_score=0.9,
    )
    nodes = GeoQueryNodes(embedder=FakeEmbedder(), retriever=GoodRetriever(result), llm_client=FakeLLM())
    workflow = create_geo_query_workflow(nodes)

    final = workflow.invoke(_state(query="Find a quiet place to study"))

    assert final["context"]["intent"] == "study"
    assert final["retrieval_quality"] == "weak"
    assert final["fallback_used"] is True
