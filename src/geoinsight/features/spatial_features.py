from __future__ import annotations

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.schemas import PlaceRecord

from geoinsight.config import load_config, load_json_config
from geoinsight.eda.spatial_distribution import distance_band


def build_spatial_features(records: list[PlaceRecord]) -> list[dict]:
    groups = load_json_config(load_config().category_groups_path, {})
    return [
        {
            "place_id": record.place_id,
            "distance_to_origin_m": record.distance_to_origin_m,
            "walking_distance_estimate_m": record.walking_distance_estimate_m
            or record.distance_to_origin_m * 1.25,
            "distance_band": distance_band(record.distance_to_origin_m),
            "nearby_place_count_100m": nearby_count(record, records, 100),
            "nearby_place_count_250m": nearby_count(record, records, 250),
            "nearest_transport_distance_m": nearest_group_distance(record, records, groups.get("transport", [])),
            "nearest_food_distance_m": nearest_group_distance(record, records, groups.get("food", [])),
            "nearest_green_space_distance_m": nearest_group_distance(record, records, groups.get("outdoor", [])),
            "nearest_study_place_distance_m": nearest_group_distance(record, records, groups.get("study", [])),
        }
        for record in records
    ]


def nearby_count(record: PlaceRecord, records: list[PlaceRecord], radius_m: float) -> int:
    return sum(
        1
        for other in records
        if other.place_id != record.place_id
        and haversine_distance_m(record.latitude, record.longitude, other.latitude, other.longitude) <= radius_m
    )


def nearest_group_distance(record: PlaceRecord, records: list[PlaceRecord], categories: list[str]) -> float | None:
    distances = [
        haversine_distance_m(record.latitude, record.longitude, other.latitude, other.longitude)
        for other in records
        if other.place_id != record.place_id and other.category in set(categories)
    ]
    return round(min(distances), 3) if distances else None
