from __future__ import annotations

from collections import Counter
from typing import Any

from geoinsight.schemas import PlaceRecord


def dataset_overview(
    records: list[PlaceRecord],
    raw_feature_count: int | None = None,
    invalid_geometry_count: int = 0,
) -> dict[str, Any]:
    category_counts = Counter(record.category for record in records)
    subcategory_counts = Counter(record.subcategory for record in records if record.subcategory)
    names = [record.name.strip().lower() for record in records if record.name]
    return {
        "raw_features": raw_feature_count if raw_feature_count is not None else len(records),
        "cleaned_places": len(records),
        "named_places": sum(1 for record in records if not record.name.lower().startswith("unnamed")),
        "unnamed_places": sum(1 for record in records if record.name.lower().startswith("unnamed")),
        "invalid_removed_geometries": invalid_geometry_count,
        "category_counts": dict(category_counts.most_common()),
        "subcategory_counts": dict(subcategory_counts.most_common()),
        "missing_value_counts": {
            "name": sum(1 for record in records if not record.name),
            "category": sum(1 for record in records if not record.category),
            "subcategory": sum(1 for record in records if not record.subcategory),
            "themes": sum(1 for record in records if not record.themes),
        },
        "duplicate_counts": {
            "duplicate_names": sum(count - 1 for count in Counter(names).values() if count > 1)
        },
    }


def amenity_theme_overview(records: list[PlaceRecord], area_sq_km: float | None = None) -> dict[str, Any]:
    category_counts = Counter(record.category for record in records)
    theme_counts = Counter(theme for record in records for theme in record.themes)
    themed_counts = {
        theme: sum(1 for record in records if theme in record.themes)
        for theme in ("study", "food", "transport", "outdoor", "social")
    }
    overview = {
        "top_categories": dict(category_counts.most_common(10)),
        "top_themes": dict(theme_counts.most_common(10)),
        "theme_place_counts": themed_counts,
    }
    if area_sq_km and area_sq_km > 0:
        overview["amenity_density_per_sq_km"] = round(len(records) / area_sq_km, 3)
    return overview
