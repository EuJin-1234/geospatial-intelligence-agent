from __future__ import annotations

import inspect

import app
from geoinsight.schemas import QueryResponse, RetrievedPlace


def test_streamlit_dataframe_helper_uses_response_only(make_record):
    retrieved = RetrievedPlace(
        record=make_record(name="Demo Cafe", category="cafe"),
        semantic_score=0.8,
        spatial_score=0.7,
        theme_score=1.0,
        contextual_score=0.6,
        combined_score=0.78,
        is_strong_match=True,
        match_reason="Strong match: matches study intent.",
    )
    response = QueryResponse(query="demo", answer="answer", retrieved_places=[retrieved])

    dataframe = app.retrieved_places_dataframe(response)

    assert list(dataframe["Name"]) == ["Demo Cafe"]
    assert list(dataframe["Match"]) == ["strong"]


def test_streamlit_app_calls_pipeline_wrapper_not_retrieval_logic():
    source = inspect.getsource(app)

    assert "run_spatial_query" in source
    assert "HybridRetriever" not in source
    assert "FaissVectorStore" not in source
