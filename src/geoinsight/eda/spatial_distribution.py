from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.schemas import PlaceRecord


DISTANCE_BANDS = (
    ("0-250m", 0, 250),
    ("250-500m", 250, 500),
    ("500-1000m", 500, 1000),
    ("1000m+", 1000, math.inf),
)


def distance_band(distance_m: float) -> str:
    for label, lower, upper in DISTANCE_BANDS:
        if lower <= distance_m < upper:
            return label
    return "1000m+"


def spatial_overview(records: list[PlaceRecord]) -> dict[str, Any]:
    if not records:
        return {
            "bounding_box": None,
            "centroid": None,
            "distance_distribution_from_origin": {},
            "nearest_neighbour_distance_summary": {},
            "category_distribution_by_distance_band": {},
        }

    lats = [record.latitude for record in records]
    lons = [record.longitude for record in records]
    distances = [record.distance_to_origin_m for record in records]
    by_band: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_band[distance_band(record.distance_to_origin_m)][record.category] += 1

    nearest_distances = [_nearest_distance(record, records) for record in records]
    area_sq_km = _bbox_area_sq_km(min(lats), min(lons), max(lats), max(lons))
    return {
        "bounding_box": {
            "min_lat": min(lats),
            "min_lon": min(lons),
            "max_lat": max(lats),
            "max_lon": max(lons),
            "area_sq_km_estimate": area_sq_km,
        },
        "centroid": {"lat": mean(lats), "lon": mean(lons)},
        "distance_distribution_from_origin": _summary(distances),
        "nearest_neighbour_distance_summary": _summary(nearest_distances),
        "category_distribution_by_distance_band": {
            band: dict(counter.most_common()) for band, counter in by_band.items()
        },
    }


def _nearest_distance(record: PlaceRecord, records: list[PlaceRecord]) -> float:
    distances = [
        haversine_distance_m(record.latitude, record.longitude, other.latitude, other.longitude)
        for other in records
        if other.place_id != record.place_id
    ]
    return min(distances) if distances else 0.0


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "min": round(min(values), 3),
        "median": round(median(values), 3),
        "mean": round(mean(values), 3),
        "max": round(max(values), 3),
    }


def _bbox_area_sq_km(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> float:
    width = haversine_distance_m(min_lat, min_lon, min_lat, max_lon)
    height = haversine_distance_m(min_lat, min_lon, max_lat, min_lon)
    return round((width * height) / 1_000_000, 4)
