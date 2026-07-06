from __future__ import annotations

from geoinsight.processing.spatial_features import (
    add_spatial_features,
    haversine_distance_m,
    walking_distance_estimate_m,
)


def test_haversine_distance_zero():
    assert haversine_distance_m(50.935, -1.396, 50.935, -1.396) == 0


def test_walking_distance_estimate():
    assert walking_distance_estimate_m(100) == 125


def test_add_spatial_features_detects_nearby(make_record):
    library = make_record(place_id="library", category="library")
    cafe = make_record(place_id="cafe", name="Cafe", category="cafe")

    enriched = add_spatial_features([library, cafe], 50.935, -1.396)

    assert enriched[0].distance_to_origin_m == 0
    assert "cafe" in enriched[0].nearby_feature_types
    assert "within walking distance of campus" in enriched[0].spatial_context
