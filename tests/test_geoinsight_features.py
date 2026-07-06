from __future__ import annotations

from geoinsight.features.feature_pipeline import build_place_features
from geoinsight.features.semantic_features import infer_themes_from_rules


def test_feature_engineering_outputs_expected_columns(make_record):
    records = [
        make_record(name="Cafe", category="cafe", themes=["food", "study"], place_id="a"),
        make_record(name="Library", category="library", themes=["study"], place_id="b", latitude=50.936),
    ]

    rows = build_place_features(records, save=False)

    assert rows[0]["distance_band"]
    assert "food_density_250m" in rows[0]
    assert "walkability_proxy_score" in rows[0]
    assert rows[0]["primary_theme"] == "food"


def test_theme_rules_are_config_driven():
    assert "food" in infer_themes_from_rules("cafe", "coffee lunch")
