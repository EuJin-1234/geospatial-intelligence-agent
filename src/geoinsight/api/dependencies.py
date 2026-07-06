from __future__ import annotations

from pathlib import Path

import requests

from geoinsight.config import GeoInsightConfig, load_config


def get_config() -> GeoInsightConfig:
    return load_config()


def readiness_checks(config: GeoInsightConfig) -> tuple[dict[str, bool], list[str], list[str]]:
    paths: dict[str, Path] = {
        "eda_report": config.eda_summary_json_path,
        "data_quality_report": config.data_quality_report_path,
        "features": config.place_features_json_path,
        "features_parquet": config.place_features_parquet_path,
        "vector_index": config.legacy.faiss_index_path,
        "vector_metadata": config.legacy.metadata_path,
    }
    checks = {name: path.exists() for name, path in paths.items()}
    provider_available, warning = llm_provider_available(config)
    checks["llm_provider_available"] = provider_available
    missing = [
        name
        for name, exists in checks.items()
        if not exists and not (name == "llm_provider_available" and config.llm_provider == "ollama")
    ]
    warnings = [warning] if warning else []
    return checks, missing, warnings


def llm_provider_available(config: GeoInsightConfig) -> tuple[bool, str | None]:
    if config.llm_provider == "template":
        return True, None
    if config.llm_provider == "azure":
        if (
            config.azure_openai_endpoint
            and config.azure_openai_deployment
            and config.azure_openai_api_key
        ):
            return True, None
        return False, "Azure provider selected but endpoint/deployment/API key settings are missing."
    if config.llm_provider == "ollama":
        try:
            response = requests.get(
                f"{config.legacy.ollama_base_url.rstrip('/')}/api/tags",
                timeout=1.0,
            )
            return response.ok, None if response.ok else "Ollama health check returned a non-OK response."
        except requests.RequestException:
            return False, "Ollama is not reachable; API can still start, but LLM calls may use fallback behavior."
    return False, f"Unsupported LLM_PROVIDER={config.llm_provider!r}."
