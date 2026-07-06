from __future__ import annotations

import hashlib
import json
import logging
import string
import unicodedata
from typing import Any

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.processing.thematic_enrichment import assign_themes
from geoinsight.schemas import PlaceRecord

LOGGER = logging.getLogger(__name__)
CATEGORY_COLUMNS = ("amenity", "leisure", "shop", "public_transport", "tourism")
USEFUL_TAGS = ("amenity", "leisure", "shop", "public_transport", "tourism", "operator", "brand", "addr:street", "addr:housenumber")


def clean_osm_features(gdf) -> list[PlaceRecord]:
    if gdf is None or len(gdf) == 0: return []
    try:
        gdf = gdf.copy()
        if "geometry" not in gdf.columns: return []
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geometry.is_valid]
    except Exception as exc:
        LOGGER.warning("Could not validate geometries: %s", exc); return []
    records: list[PlaceRecord] = []
    seen: list[tuple[str, float, float]] = []
    for _, row in gdf.iterrows():
        geometry = row.get("geometry")
        if geometry is None: continue
        point = geometry if geometry.geom_type == "Point" else geometry.centroid
        lon, lat = float(point.x), float(point.y)
        tags = _extract_metadata(row)
        category, subcategory = _category_from_tags(tags)
        name = _name_from_row(row, category)
        normalised = normalise_place_name(name)
        if _is_duplicate(normalised, lat, lon, seen): continue
        seen.append((normalised, lat, lon))
        records.append(PlaceRecord(place_id=_stable_place_id(name, category, lat, lon), name=name, category=category, subcategory=subcategory, latitude=lat, longitude=lon, distance_to_origin_m=0, walking_distance_estimate_m=0, nearby_feature_types=[], themes=assign_themes(category, subcategory, tags), spatial_context="", llm_description="", metadata=tags))
    return records


def normalise_place_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", name)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value.lower().strip().translate(str.maketrans("", "", string.punctuation))


def _is_duplicate(normalised_name: str, lat: float, lon: float, seen: list[tuple[str, float, float]]) -> bool:
    return any(seen_name == normalised_name and haversine_distance_m(lat, lon, seen_lat, seen_lon) < 50 for seen_name, seen_lat, seen_lon in seen)


def _extract_metadata(row) -> dict[str, Any]:
    return {key: _json_safe(row[key]) for key in USEFUL_TAGS if key in row and _is_known(row[key])}


def _category_from_tags(tags: dict[str, Any]) -> tuple[str, str | None]:
    for column in CATEGORY_COLUMNS:
        value = tags.get(column)
        if value is None: continue
        if column in {"shop", "public_transport"}: return column, str(value) if value is not True else None
        return str(value), None
    return "place", None


def _name_from_row(row, category: str) -> str:
    if "name" in row and _is_known(row["name"]): return str(row["name"])
    if category == "cafe": return "Unnamed cafe"
    if category == "shop": return "Unnamed shop"
    if category in {"park", "garden"}: return "Unnamed park"
    return "Unnamed place"


def _stable_place_id(name: str, category: str, lat: float, lon: float) -> str:
    return hashlib.sha1(f"{name}|{category}|{round(lat, 6)}|{round(lon, 6)}".encode("utf-8")).hexdigest()[:16]


def _is_known(value: Any) -> bool:
    if value is None: return False
    try:
        import pandas as pd
        if pd.isna(value): return False
    except (ImportError, TypeError, ValueError):
        pass
    return value != ""


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value); return value
    except TypeError:
        return str(value)