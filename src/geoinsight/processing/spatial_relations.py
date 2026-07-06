from __future__ import annotations

from typing import Any

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.schemas import PlaceRecord

USEFUL_CATEGORIES = {"library", "cafe", "restaurant", "fast_food", "garden", "park", "bus_station", "public_transport", "university", "school"}
LOW_SIGNAL_CATEGORIES = {"parking", "bicycle_parking"}
THEME_RELATIONS = {"study": {"library", "university", "school"}, "food": {"cafe", "restaurant", "fast_food"}, "outdoor": {"garden", "park"}, "transport": {"bus_station", "public_transport"}}


def build_spatial_relations(records: list[PlaceRecord], relation_radius_m: float = 200.0, origin_name: str = "University of Southampton Highfield Campus") -> list[PlaceRecord]:
    enriched = []
    for record in records:
        relations: list[dict[str, Any]] = []
        if record.distance_to_origin_m <= 1000:
            distance = round(record.distance_to_origin_m, 1)
            relations.append({"relation": "within_walking_distance_of_origin", "target": origin_name, "target_category": "origin", "distance_m": distance, "evidence": f"{origin_name} is within {distance:g}m"})
        nearby = []
        for other in records:
            if other.place_id == record.place_id: continue
            distance = haversine_distance_m(record.latitude, record.longitude, other.latitude, other.longitude)
            if distance <= relation_radius_m: nearby.append((other, distance))
        for other, distance in sorted(nearby, key=lambda item: (_relation_rank(item[0]), item[1])):
            if len(relations) >= 5: break
            if _is_low_signal_unnamed(other): continue
            distance = round(distance, 1)
            if other.category in USEFUL_CATEGORIES:
                relations.append({"relation": "near", "target": other.name, "target_category": other.category, "distance_m": distance, "evidence": f"{other.name} is within {distance:g}m"})
            elif other.category not in LOW_SIGNAL_CATEGORIES:
                relations.append({"relation": "near_category", "target": other.category, "target_category": other.category, "distance_m": distance, "evidence": f"Nearby {other.category} feature within {distance:g}m"})
            for theme, categories in THEME_RELATIONS.items():
                if len(relations) >= 5: break
                if other.category in categories and not any(r.get("relation") == "near_theme" and r.get("target") == theme for r in relations):
                    relations.append({"relation": "near_theme", "target": theme, "target_category": "semantic_theme", "distance_m": distance, "evidence": _theme_evidence(theme, distance)})
        enriched.append(record.model_copy(update={"spatial_relations": relations}))
    return enriched


def relation_summary(record: PlaceRecord) -> str:
    phrases = []
    targets = {(item.get("relation"), item.get("target")) for item in record.spatial_relations}
    if any(relation == "within_walking_distance_of_origin" for relation, _ in targets): phrases.append("within walking distance of the origin")
    if ("near_theme", "study") in targets: phrases.append("near study-related places")
    if ("near_theme", "food") in targets: phrases.append("near food options")
    if ("near_theme", "outdoor") in targets: phrases.append("near green space")
    if ("near_theme", "transport") in targets: phrases.append("near public transport")
    if any(item.get("target_category") in {"university", "school"} for item in record.spatial_relations): phrases.append("near university facilities")
    return ", ".join(dict.fromkeys(phrases))


def _relation_rank(record: PlaceRecord) -> int:
    if record.category in USEFUL_CATEGORIES: return 0
    if record.category in LOW_SIGNAL_CATEGORIES: return 2
    return 1


def _is_low_signal_unnamed(record: PlaceRecord) -> bool:
    return record.category in LOW_SIGNAL_CATEGORIES and record.name.lower().startswith("unnamed")


def _theme_evidence(theme: str, distance_m: float) -> str:
    labels = {"study": "library/university", "food": "cafe/restaurant", "outdoor": "garden/park", "transport": "transport"}
    return f"Nearby {labels.get(theme, theme)} feature within {distance_m:g}m"