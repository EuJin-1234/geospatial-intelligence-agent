from __future__ import annotations

from geoinsight.retrieval.hybrid_retriever import HybridGeoRetriever
from geoinsight.retrieval.intent_parser import parse_intent
from geoinsight.retrieval.reranker import rerank_places
from geoinsight.schemas import RankedPlace


def test_intent_parser_extracts_theme_and_distance():
    parsed = parse_intent("Find food within 500m near transport")

    assert "food" in parsed.requested_themes
    assert parsed.distance_constraint_m == 500


def test_reranker_penalises_parking_without_transport(make_record):
    result = RankedPlace(
        record=make_record(name="Car Park", category="parking", themes=[]),
        combined_score=1.0,
    )

    reranked = rerank_places([result], intent="study")

    assert reranked[0].combined_score < 1.0


def test_hybrid_retriever_scores_candidates(make_record, FakeVectorStore, FakeEmbedder):
    record = make_record(name="Cafe", category="cafe", themes=["food", "study"])
    store = FakeVectorStore([(record, 0.9)])
    retriever = HybridGeoRetriever(store, FakeEmbedder())

    results = retriever.retrieve("Find study food", requested_themes=["study"], top_k=1)

    assert results
    assert results[0].combined_score > 0
