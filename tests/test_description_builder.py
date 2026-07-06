from __future__ import annotations

from geoinsight.processing.description_builder import build_description


def test_description_uses_known_fields_only(make_record):
    record = make_record(name="Avenue Cafe", category="cafe", themes=["coffee", "study"])
    description = build_description(record)

    assert "Avenue Cafe is a cafe" in description
    assert "300m from the origin" in description
    assert "coffee, study" in description
    assert "Nearby feature types include cafe and public_transport" not in description
    assert "ratings" not in description.lower()
    assert "opening hours" not in description.lower()


def test_description_includes_relation_summary(make_record):
    record = make_record(
        name="Cafe 38",
        category="cafe",
        spatial_relations=[
            {
                "relation": "near_theme",
                "target": "study",
                "target_category": "semantic_theme",
                "distance_m": 80,
                "evidence": "Nearby library/university feature within 80m",
            }
        ],
    )

    assert "near study-related places" in build_description(record)
