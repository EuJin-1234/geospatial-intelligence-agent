from __future__ import annotations

import json
from pathlib import Path

from geoinsight.config import LegacyAppConfig, ensure_legacy_data_dirs, load_legacy_config
from geoinsight.processing.spatial_relations import relation_summary
from geoinsight.schemas import PlaceRecord


def build_description(record: PlaceRecord) -> str:
    lines = [f"{record.name} is a {record.category} {round(record.distance_to_origin_m)}m from the origin."]
    if record.themes: lines.append(f"It has themes {', '.join(record.themes)}.")
    if record.nearby_feature_types: lines.append(f"Nearby feature types include {', '.join(record.nearby_feature_types)}.")
    summary = relation_summary(record)
    if summary: lines.append(f"It is {summary}.")
    if record.spatial_context: lines.append(f"Spatial context: {record.spatial_context}.")
    return "\n".join(lines)


def add_descriptions(records: list[PlaceRecord]) -> list[PlaceRecord]:
    return [record.model_copy(update={"llm_description": build_description(record)}) for record in records]


def save_records(records: list[PlaceRecord], path: Path | None = None, config: LegacyAppConfig | None = None) -> Path:
    config = config or load_legacy_config()
    ensure_legacy_data_dirs(config)
    output_path = path or config.processed_records_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([record.model_dump(mode="json") for record in records], indent=2), encoding="utf-8")
    return output_path


def load_records(path: Path | None = None, config: LegacyAppConfig | None = None) -> list[PlaceRecord]:
    config = config or load_legacy_config()
    input_path = path or config.processed_records_path
    return [PlaceRecord.model_validate(item) for item in json.loads(input_path.read_text(encoding="utf-8"))]