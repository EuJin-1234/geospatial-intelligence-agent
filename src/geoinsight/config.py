from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency path
    def load_dotenv(*args, **kwargs):
        return False

DEFAULT_LAT = 50.9350
DEFAULT_LON = -1.3960
DEFAULT_RADIUS_M = 1500


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LegacyAppConfig:
    root_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    index_dir: Path
    maps_dir: Path
    raw_osm_path: Path
    processed_records_path: Path
    faiss_index_path: Path
    metadata_path: Path
    latest_map_path: Path
    latency_report_path: Path
    ollama_base_url: str
    ollama_model: str
    embedding_model_name: str
    embedding_device: str
    embedding_backend: str
    onnx_embedding_model_dir: Path
    request_timeout_s: int = 30


@dataclass(frozen=True)
class GeoInsightConfig:
    legacy: LegacyAppConfig
    features_dir: Path
    reports_dir: Path
    theme_rules_path: Path
    retrieval_weights_path: Path
    category_groups_path: Path
    place_features_json_path: Path
    place_features_parquet_path: Path
    eda_summary_json_path: Path
    eda_summary_md_path: Path
    data_quality_report_path: Path
    evaluation_report_path: Path
    llm_provider: str
    azure_openai_endpoint: str | None
    azure_openai_deployment: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str
    api_host: str
    api_port: int
    api_reload: bool
    api_cors_origins: list[str]
    environment: str
    log_level: str

    @property
    def root_dir(self) -> Path:
        return self.legacy.root_dir


def load_legacy_config() -> LegacyAppConfig:
    root = project_root()
    load_dotenv(root / ".env")
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    index_dir = data_dir / "index"
    maps_dir = data_dir / "maps"
    return LegacyAppConfig(
        root_dir=root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        index_dir=index_dir,
        maps_dir=maps_dir,
        raw_osm_path=raw_dir / "osm_features.geojson",
        processed_records_path=processed_dir / "place_records.json",
        faiss_index_path=index_dir / "places.faiss",
        metadata_path=index_dir / "places_metadata.json",
        latest_map_path=maps_dir / "latest_query_map.html",
        latency_report_path=processed_dir / "latency_report.json",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformer"),
        onnx_embedding_model_dir=Path(
            os.getenv("ONNX_EMBEDDING_MODEL_DIR", str(index_dir / "embedding_onnx"))
        ),
    )


def load_config() -> GeoInsightConfig:
    legacy = load_legacy_config()
    configs_dir = project_root() / "configs"
    features_dir = legacy.data_dir / "features"
    reports_dir = legacy.data_dir / "reports"
    return GeoInsightConfig(
        legacy=legacy,
        features_dir=features_dir,
        reports_dir=reports_dir,
        theme_rules_path=configs_dir / "theme_rules.json",
        retrieval_weights_path=configs_dir / "retrieval_weights.json",
        category_groups_path=configs_dir / "category_groups.json",
        place_features_json_path=features_dir / "place_features.json",
        place_features_parquet_path=features_dir / "place_features.parquet",
        eda_summary_json_path=reports_dir / "eda_summary.json",
        eda_summary_md_path=reports_dir / "eda_summary.md",
        data_quality_report_path=reports_dir / "data_quality_report.json",
        evaluation_report_path=reports_dir / "evaluation_report.json",
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or None,
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or None,
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY") or None,
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_reload=os.getenv("API_RELOAD", "true").lower() in {"1", "true", "yes"},
        api_cors_origins=[
            origin.strip()
            for origin in os.getenv(
                "API_CORS_ORIGINS",
                "http://localhost:8501,http://127.0.0.1:8501",
            ).split(",")
            if origin.strip()
        ],
        environment=os.getenv("ENVIRONMENT", "local"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def ensure_legacy_data_dirs(config: LegacyAppConfig | None = None) -> None:
    config = config or load_legacy_config()
    for path in (config.raw_dir, config.processed_dir, config.index_dir, config.maps_dir):
        path.mkdir(parents=True, exist_ok=True)


def ensure_data_dirs(config: GeoInsightConfig | None = None) -> None:
    config = config or load_config()
    ensure_legacy_data_dirs(config.legacy)
    for path in (config.features_dir, config.reports_dir, config.legacy.maps_dir):
        path.mkdir(parents=True, exist_ok=True)


def load_json_config(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))