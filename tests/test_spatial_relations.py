from __future__ import annotations

from geoinsight.processing.spatial_relations import build_spatial_relations, relation_summary


def test_spatial_relations_include_origin_and_near_theme(make_record):
    cafe = make_record(
        name="Cafe 38",
        category="cafe",
        place_id="cafe",
        latitude=50.9350,
        longitude=-1.3960,
        distance=100,
    )
    library = make_record(
        name="Hartley Library",
        category="library",
        place_id="library",
        latitude=50.9354,
        longitude=-1.3960,
        distance=140,
    )

    enriched = build_spatial_relations([cafe, library], relation_radius_m=200)
    cafe_relations = enriched[0].spatial_relations

    assert any(item["relation"] == "within_walking_distance_of_origin" for item in cafe_relations)
    assert any(item["relation"] == "near" and item["target"] == "Hartley Library" for item in cafe_relations)
    assert any(item["relation"] == "near_theme" and item["target"] == "study" for item in cafe_relations)
    assert "near study-related places" in relation_summary(enriched[0])


def test_spatial_relations_avoid_unnamed_low_signal_parking(make_record):
    cafe = make_record(name="Cafe", category="cafe", place_id="cafe")
    parking = make_record(
        name="Unnamed bicycle parking",
        category="bicycle_parking",
        place_id="parking",
        latitude=50.9351,
        longitude=-1.3960,
    )

    enriched = build_spatial_relations([cafe, parking], relation_radius_m=200)

    assert not any(item["target"] == "Unnamed bicycle parking" for item in enriched[0].spatial_relations)
