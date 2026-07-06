from __future__ import annotations

from geoinsight.retrieval.retriever import HybridRetriever
from geoinsight.schemas import QueryContext


def test_hybrid_retrieval_scores_and_sorts(make_record, FakeEmbedder, FakeVectorStore):
    close_cafe = make_record(
        name="Close Cafe",
        category="cafe",
        distance=100,
        themes=["coffee", "food", "study"],
        place_id="close",
    )
    far_library = make_record(
        name="Far Library",
        category="library",
        distance=1400,
        themes=["quiet", "study"],
        place_id="far",
    )
    store = FakeVectorStore([(far_library, 0.8), (close_cafe, 0.7)])
    retriever = HybridRetriever(store, FakeEmbedder())

    results = retriever.retrieve(
        "quiet coffee study",
        top_k=2,
        max_distance_m=1500,
        requested_themes=["coffee", "study"],
    )

    assert results[0].record.name == "Close Cafe"
    assert results[0].theme_score == 1.0
    assert results[0].spatial_score > results[1].spatial_score


def test_retriever_filters_by_max_distance(make_record, FakeEmbedder, FakeVectorStore):
    far = make_record(name="Far Place", distance=2000)
    retriever = HybridRetriever(FakeVectorStore([(far, 0.9)]), FakeEmbedder())

    assert retriever.retrieve("anything", top_k=5, max_distance_m=1500) == []


def test_parking_penalised_for_non_transport_query(make_record, FakeEmbedder, FakeVectorStore):
    park = make_record(
        name="Valley Garden",
        category="garden",
        themes=["outdoor", "relax", "quiet"],
        place_id="garden",
    )
    bicycle_parking = make_record(
        name="Bicycle Parking",
        category="bicycle_parking",
        themes=[],
        place_id="bike",
    )
    retriever = HybridRetriever(
        FakeVectorStore([(bicycle_parking, 0.95), (park, 0.75)]),
        FakeEmbedder(),
    )

    results = retriever.retrieve(
        "Find places to relax outdoors",
        top_k=2,
        requested_themes=["relax", "outdoor"],
    )

    assert results[0].record.name == "Valley Garden"
    assert results[1].combined_score < results[0].combined_score
    assert results[0].is_strong_match is True
    assert results[1].is_strong_match is False
    assert "parking-related category" in results[1].match_reason


def test_bicycle_parking_allowed_for_transport_query(make_record, FakeEmbedder, FakeVectorStore):
    bicycle_parking = make_record(
        name="Bicycle Parking",
        category="bicycle_parking",
        themes=["transport"],
        place_id="bike",
    )
    cafe = make_record(name="Cafe", category="cafe", themes=["coffee"], place_id="cafe")
    retriever = HybridRetriever(
        FakeVectorStore([(bicycle_parking, 0.8), (cafe, 0.9)]),
        FakeEmbedder(),
    )

    results = retriever.retrieve_with_embedding(
        query_embedding=[1.0, 0.0],
        top_k=2,
        requested_themes=["transport"],
        context=QueryContext(intent="transport"),
    )

    assert results[0].record.category == "bicycle_parking"
    assert results[0].contextual_score > 0
    assert results[0].is_strong_match is True
