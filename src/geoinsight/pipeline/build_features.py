from __future__ import annotations

from geoinsight.eda.report_builder import load_records
from geoinsight.features.feature_pipeline import build_place_features


def build_features() -> list[dict]:
    return build_place_features(load_records())
