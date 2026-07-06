from __future__ import annotations

import pytest

from geoinsight.schemas import PlaceRecord, QueryRequest


def test_place_record_is_json_serialisable(make_record):
    record = make_record()
    dumped = record.model_dump(mode="json")
    assert dumped["name"] == "Hartley Library"
    assert isinstance(record.model_dump_json(), str)


def test_schema_rejects_invalid_coordinates(make_record):
    with pytest.raises(ValueError):
        PlaceRecord(**{**make_record().model_dump(), "latitude": 120})

    with pytest.raises(ValueError):
        QueryRequest(query="x", lat=50.0, lon=-200)


def test_query_request_requires_positive_top_k():
    with pytest.raises(ValueError):
        QueryRequest(query="x", lat=50.0, lon=-1.0, top_k=0)
