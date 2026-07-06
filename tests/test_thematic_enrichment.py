from __future__ import annotations

from geoinsight.processing.thematic_enrichment import assign_themes


def test_assign_themes_library():
    assert assign_themes("library", None, {"amenity": "library"}) == [
        "campus",
        "quiet",
        "study",
    ]


def test_assign_themes_cafe_is_sorted_and_deterministic():
    assert assign_themes("cafe", None, {"amenity": "cafe"}) == [
        "coffee",
        "food",
        "social",
        "study",
    ]


def test_assign_themes_transport_and_park():
    assert assign_themes("public_transport", "platform", {"public_transport": "platform"}) == [
        "transport"
    ]
    assert assign_themes("park", None, {"leisure": "park"}) == ["outdoor", "quiet", "relax"]
