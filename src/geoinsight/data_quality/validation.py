from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from geoinsight.processing.spatial_features import haversine_distance_m
from geoinsight.schemas import PlaceRecord

from geoinsight.config import ensure_data_dirs, load_config
from geoinsight.schemas import DataQualityReport


def validate_place_records(
    records: list[PlaceRecord],
    output_path: Path | None = None,
) -> DataQualityReport:
    warnings: Counter[str] = Counter()
    invalid_ids: set[str] = set()
    seen: list[PlaceRecord] = []

    for record in records:
        if not -90 <= record.latitude <= 90 or not -180 <= record.longitude <= 180:
            warnings["invalid_coordinates"] += 1
            invalid_ids.add(record.place_id)
        if record.distance_to_origin_m < 0 or record.walking_distance_estimate_m < 0:
            warnings["negative_distances"] += 1
            invalid_ids.add(record.place_id)
        if not record.name or record.name.lower().startswith("unnamed"):
            warnings["missing_names"] += 1
        if not record.category or record.category == "place":
            warnings["missing_categories"] += 1
        if not record.themes:
            warnings["empty_themes"] += 1
        if _metadata_score(record) < 2:
            warnings["thin_metadata"] += 1
        if _is_near_duplicate(record, seen):
            warnings["duplicate_names_nearby_coordinates"] += 1
        seen.append(record)

    report = DataQualityReport(
        total_records=len(records),
        valid_records=len(records) - len(invalid_ids),
        invalid_records=len(invalid_ids),
        warning_count=sum(warnings.values()),
        warnings_by_type=dict(warnings),
        recommendations=_recommendations(warnings),
    )
    if output_path is None:
        config = load_config()
        ensure_data_dirs(config)
        output_path = config.data_quality_report_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def validate_raw_geometries(gdf) -> dict[str, int]:
    if gdf is None or "geometry" not in getattr(gdf, "columns", []):
        return {"raw_features": 0, "missing_geometry": 0, "invalid_geometry": 0}
    missing = int(gdf.geometry.isna().sum())
    valid = gdf.geometry.dropna().is_valid
    return {
        "raw_features": int(len(gdf)),
        "missing_geometry": missing,
        "invalid_geometry": int((~valid).sum()),
    }


def load_data_quality_report(path: Path | None = None) -> DataQualityReport | None:
    path = path or load_config().data_quality_report_path
    if not path.exists():
        return None
    return DataQualityReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _metadata_score(record: PlaceRecord) -> int:
    return sum(
        1
        for value in (
            record.name,
            record.category,
            record.subcategory,
            record.llm_description,
            record.spatial_context,
            record.metadata,
        )
        if value
    )


def _is_near_duplicate(record: PlaceRecord, seen: list[PlaceRecord]) -> bool:
    name = record.name.strip().lower()
    if not name or name.startswith("unnamed"):
        return False
    return any(
        item.name.strip().lower() == name
        and haversine_distance_m(
            record.latitude,
            record.longitude,
            item.latitude,
            item.longitude,
        )
        < 50
        for item in seen
    )


def _recommendations(warnings: Counter[str]) -> list[str]:
    recommendations = []
    if warnings["missing_names"]:
        recommendations.append("Use category-aware generated labels for unnamed OSM features.")
    if warnings["empty_themes"]:
        recommendations.append("Review theme rules for categories that are not currently mapped.")
    if warnings["duplicate_names_nearby_coordinates"]:
        recommendations.append("Tighten deduplication by normalised name and coordinate proximity.")
    if warnings["thin_metadata"]:
        recommendations.append("Preserve more OSM tags during cleaning for stronger evidence.")
    if not recommendations:
        recommendations.append("No major data quality risks detected in structured records.")
    return recommendations
