from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from geoinsight.schemas import PlaceRecord

from geoinsight.config import ensure_data_dirs, load_config
from geoinsight.features.accessibility_features import build_accessibility_features
from geoinsight.features.density_features import build_density_features
from geoinsight.features.semantic_features import build_semantic_features
from geoinsight.features.spatial_features import build_spatial_features


def build_place_features(records: list[PlaceRecord], save: bool = True) -> list[dict[str, Any]]:
    spatial = _by_id(build_spatial_features(records))
    density = _by_id(build_density_features(records))
    accessibility = _by_id(build_accessibility_features(list(spatial.values())))
    semantic = _by_id(build_semantic_features(records))

    rows = []
    for record in records:
        row = {
            "place_id": record.place_id,
            "name": record.name,
            "category": record.category,
            "subcategory": record.subcategory,
            "latitude": record.latitude,
            "longitude": record.longitude,
        }
        for source in (spatial, density, accessibility, semantic):
            row.update(source.get(record.place_id, {}))
        rows.append(row)

    if save:
        save_place_features(rows)
    return rows


def save_place_features(rows: list[dict[str, Any]]) -> dict[str, str]:
    config = load_config()
    ensure_data_dirs(config)
    json_ready = json.loads(json.dumps(rows, default=str))
    config.place_features_json_path.write_text(json.dumps(json_ready, indent=2), encoding="utf-8")
    outputs = {"json": str(config.place_features_json_path)}
    try:
        pd.DataFrame(rows).to_parquet(config.place_features_parquet_path, index=False)
        outputs["parquet"] = str(config.place_features_parquet_path)
    except Exception:
        csv_path = config.features_dir / "place_features.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    return outputs


def load_place_features(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or load_config().place_features_json_path
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["place_id"]: row for row in rows}
