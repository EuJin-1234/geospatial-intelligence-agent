from __future__ import annotations

import math

from geoinsight.schemas import PlaceRecord

from geoinsight.config import load_config, load_json_config
from geoinsight.features.spatial_features import nearby_count


def build_density_features(records: list[PlaceRecord], radius_m: float = 250) -> list[dict]:
    groups = load_json_config(load_config().category_groups_path, {})
    area_sq_km = math.pi * (radius_m / 1000) ** 2
    return [
        {
            "place_id": record.place_id,
            "food_density_250m": _density(record, records, groups.get("food", []), radius_m, area_sq_km),
            "transport_density_250m": _density(record, records, groups.get("transport", []), radius_m, area_sq_km),
            "study_density_250m": _density(record, records, groups.get("study", []), radius_m, area_sq_km),
            "outdoor_density_250m": _density(record, records, groups.get("outdoor", []), radius_m, area_sq_km),
            "amenity_density_250m": round(nearby_count(record, records, radius_m) / area_sq_km, 3),
        }
        for record in records
    ]


def _density(
    record: PlaceRecord,
    records: list[PlaceRecord],
    categories: list[str],
    radius_m: float,
    area_sq_km: float,
) -> float:
    count = sum(
        1
        for other in records
        if other.place_id != record.place_id
        and other.category in set(categories)
        and nearby_count(record, [other], radius_m) == 1
    )
    return round(count / area_sq_km, 3)
