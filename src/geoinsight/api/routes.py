from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from geoinsight.api.dependencies import get_config, readiness_checks
from geoinsight.api.errors import actionable_file_error, not_found, service_unavailable
from geoinsight.api.schemas import (
    FeaturePreviewResponse,
    HealthResponse,
    QueryAPIRequest,
    QueryAPIResponse,
    ReadinessResponse,
    ReportResponse,
    RetrievedPlaceAPI,
)
from geoinsight.config import GeoInsightConfig
from geoinsight.retrieval.intent_parser import parse_intent
from geoinsight.schemas import DatasetBuildOptions
from geoinsight.services import data_service, query_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="GeoInsight Agent API", version="0.1.0")


@router.get("/ready", response_model=ReadinessResponse)
def ready(config: GeoInsightConfig = Depends(get_config)) -> ReadinessResponse:
    checks, missing, warnings = readiness_checks(config)
    status = "ready" if not missing else "not_ready"
    message = None if not missing else "Run the build, EDA, data-quality, feature, and index commands."
    return ReadinessResponse(
        status=status,
        environment=config.environment,
        llm_provider=config.llm_provider,
        checks=checks,
        missing=missing,
        warnings=warnings,
        message=message,
    )


@router.post("/query", response_model=QueryAPIResponse)
def query_places(request: QueryAPIRequest, config: GeoInsightConfig = Depends(get_config)) -> QueryAPIResponse:
    try:
        response = query_service.run_agent_query(
            request.query,
            intent=request.intent,
            weather=request.weather,
            time_of_day=request.time_of_day,
            top_k=request.top_k,
            max_distance_m=request.max_distance_m,
            generate_map=request.generate_map,
            use_agent=request.use_agent,
        )
    except FileNotFoundError as exc:
        raise actionable_file_error(exc) from exc
    except (NotImplementedError, RuntimeError) as exc:
        raise service_unavailable(str(exc)) from exc

    parsed = parse_intent(request.query)
    warnings = []
    if response.fallback_used:
        warnings.append("Fallback answer was used because retrieved evidence or LLM generation was limited.")
    return QueryAPIResponse(
        query=response.query,
        answer=response.answer,
        task_type=parsed.task_type,
        retrieved_places=[_retrieved_place(item) for item in response.retrieved_places],
        map_path=response.map_path,
        latency_ms=response.latency_ms,
        provider=config.llm_provider,
        warnings=warnings,
    )


@router.get("/reports/eda", response_model=ReportResponse)
def get_eda_report(config: GeoInsightConfig = Depends(get_config)) -> ReportResponse:
    if not config.eda_summary_json_path.exists():
        raise not_found("EDA report not found. Run `python -m geoinsight.cli run-eda` first.")
    return ReportResponse(
        status="ok",
        path=str(config.eda_summary_json_path),
        data=data_service.get_eda_summary(),
    )


@router.get("/reports/data-quality", response_model=ReportResponse)
def get_data_quality_report(config: GeoInsightConfig = Depends(get_config)) -> ReportResponse:
    if not config.data_quality_report_path.exists():
        raise not_found(
            "Data quality report not found. Run `python -m geoinsight.cli data-quality` first."
        )
    return ReportResponse(
        status="ok",
        path=str(config.data_quality_report_path),
        data=data_service.get_data_quality_report(),
    )


@router.get("/features/preview", response_model=FeaturePreviewResponse)
def feature_preview(limit: int = Query(20, ge=1, le=100)) -> FeaturePreviewResponse:
    try:
        records = data_service.get_feature_preview(limit)
    except FileNotFoundError as exc:
        raise actionable_file_error(exc) from exc
    return FeaturePreviewResponse(status="ok", count=len(records), records=records)


@router.post("/build/dataset", response_model=ReportResponse)
def build_dataset() -> ReportResponse:
    result = data_service.build_dataset_service()
    return ReportResponse(
        status=result.get("status", "ok"),
        data=result,
        message="Dataset build completed.",
    )


@router.post("/build/eda", response_model=ReportResponse)
def build_eda() -> ReportResponse:
    return ReportResponse.model_validate(data_service.run_eda_service())


@router.post("/build/features", response_model=ReportResponse)
def build_features() -> ReportResponse:
    return ReportResponse.model_validate(data_service.build_features_service())


@router.post("/build/index", response_model=ReportResponse)
def build_index() -> ReportResponse:
    result = data_service.build_index_service(DatasetBuildOptions())
    return ReportResponse.model_validate(result)


def _retrieved_place(item) -> RetrievedPlaceAPI:
    record = item.record
    return RetrievedPlaceAPI(
        name=record.name,
        category=record.category,
        distance_m=record.distance_to_origin_m,
        semantic_score=item.semantic_score,
        spatial_score=item.spatial_score,
        theme_score=item.theme_score,
        context_score=item.contextual_score,
        combined_score=item.combined_score,
        themes=record.themes,
        evidence=item.contextual_evidence or [item.match_reason],
        is_strong_match=item.is_strong_match,
    )
