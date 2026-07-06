from __future__ import annotations

import json
from pathlib import Path

from geoinsight.config import load_config
from geoinsight.data_quality.validation import validate_place_records
from geoinsight.eda.report_builder import build_eda_report, load_records
from geoinsight.features.feature_pipeline import build_place_features
from geoinsight.pipeline.build_dataset import build_dataset
from geoinsight.pipeline.build_index import build_index
from geoinsight.schemas import BuildReport, DatasetBuildOptions, EDAReport


def build_dataset_service(options: DatasetBuildOptions | None = None) -> dict:
    report = build_dataset(options or DatasetBuildOptions())
    return {
        "status": "ok",
        "records_built": report.records_built,
        "outputs": report.outputs,
    }


def get_eda_summary() -> dict:
    config = load_config()
    if config.eda_summary_json_path.exists():
        return _read_json(config.eda_summary_json_path)
    return build_eda_report(load_records()).model_dump(mode="json")


def get_data_quality_report() -> dict:
    config = load_config()
    if not config.data_quality_report_path.exists():
        raise FileNotFoundError(
            "Data quality report not found. Run `python -m geoinsight.cli data-quality` first."
        )
    return _read_json(config.data_quality_report_path)


def get_feature_preview(limit: int = 20) -> list[dict]:
    config = load_config()
    limit = max(1, min(limit, 100))
    if config.place_features_json_path.exists():
        return _read_json(config.place_features_json_path)[:limit]
    if config.place_features_parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(config.place_features_parquet_path).head(limit).to_dict(orient="records")
    raise FileNotFoundError(
        "Feature file not found. Run `python -m geoinsight.cli build-features` first."
    )


def run_data_quality_report() -> dict:
    return validate_place_records(load_records()).model_dump(mode="json")


def run_eda_service() -> dict:
    config = load_config()
    report = build_eda_report(load_records())
    return {
        "status": "ok",
        "path": str(config.eda_summary_json_path),
        "data": report.model_dump(mode="json"),
        "message": "EDA report generated.",
    }


def build_features_service() -> dict:
    config = load_config()
    rows = build_place_features(load_records())
    return {
        "status": "ok",
        "path": str(config.place_features_json_path),
        "data": {"count": len(rows)},
        "message": "Feature file generated.",
    }


def build_index_service(options: DatasetBuildOptions | None = None) -> dict:
    config = load_config()
    options = options or DatasetBuildOptions()
    count = build_index(
        lat=options.lat,
        lon=options.lon,
        radius_m=options.radius_m,
        force_refresh=options.force_refresh,
        embedding_model=options.embedding_model,
        embedding_backend=options.embedding_backend,
        onnx_model_dir=options.onnx_model_dir,
    )
    return {
        "status": "ok",
        "path": str(config.legacy.faiss_index_path),
        "data": {"records_indexed": count},
        "message": "Vector index built.",
    }


def run_feature_build() -> list[dict]:
    return build_place_features(load_records())


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))
