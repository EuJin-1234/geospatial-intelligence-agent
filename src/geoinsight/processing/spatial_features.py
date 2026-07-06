from __future__ import annotations

import math
from collections.abc import Iterable

from geoinsight.schemas import PlaceRecord

EARTH_RADIUS_M = 6_371_000
NEARBY_RADIUS_M = 200


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def walking_distance_estimate_m(straight_line_distance_m: float) -> float:
    return max(straight_line_distance_m, 0) * 1.25


def build_spatial_context(category: str, distance_m: float, nearby_feature_types: Iterable[str]) -> str:
    phrases: list[str] = []
    nearby = set(nearby_feature_types)
    if distance_m <= 1200:
        phrases.append("within walking distance of campus")
    if nearby.intersection({"bus_station", "public_transport", "transport"}):
        phrases.append("near public transport")
    if nearby.intersection({"cafe", "restaurant", "fast_food", "food"}):
        phrases.append("close to food options")
    if nearby.intersection({"park", "garden", "outdoor"}):
        phrases.append("near green space")
    if category in {"university", "school"} or nearby.intersection({"university", "school"}):
        phrases.append("close to university facilities")
    return "; ".join(phrases) if phrases else "near the search origin"


def add_spatial_features(records: list[PlaceRecord], origin_lat: float, origin_lon: float, nearby_radius_m: int = NEARBY_RADIUS_M) -> list[PlaceRecord]:
    base = []
    for record in records:
        distance = haversine_distance_m(origin_lat, origin_lon, record.latitude, record.longitude)
        base.append((record, distance, walking_distance_estimate_m(distance)))
    enriched = []
    for record, distance, walking in base:
        nearby = sorted({other.category for other, _, _ in base if other.place_id != record.place_id and haversine_distance_m(record.latitude, record.longitude, other.latitude, other.longitude) <= nearby_radius_m})
        enriched.append(record.model_copy(update={
            "distance_to_origin_m": round(distance, 2),
            "walking_distance_estimate_m": round(walking, 2),
            "nearby_feature_types": nearby,
            "spatial_context": build_spatial_context(record.category, distance, nearby),
        }))
    return enriched