from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator


def _validate_theme_list(value: list[str]) -> list[str]:
    if not isinstance(value, list):
        raise TypeError("themes must be a list")
    return value


class PlaceRecord(BaseModel):
    place_id: str
    name: str
    category: str
    subcategory: str | None = None
    latitude: float
    longitude: float
    distance_to_origin_m: float = Field(ge=0)
    walking_distance_estimate_m: float = Field(ge=0)
    nearby_feature_types: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    spatial_relations: list[dict[str, Any]] = Field(default_factory=list)
    spatial_context: str
    llm_description: str
    source: str = "openstreetmap"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("latitude")
    @classmethod
    def latitude_range(cls, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return value

    @field_validator("longitude")
    @classmethod
    def longitude_range(cls, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return value

    @field_validator("themes", "nearby_feature_types")
    @classmethod
    def list_fields(cls, value: list[str]) -> list[str]:
        return _validate_theme_list(value)


class QueryContext(BaseModel):
    intent: Literal["study", "relax", "food", "transport", "social", "coffee", "outdoor"] | None = None
    weather: Literal["clear", "rainy", "cold", "hot"] | None = None
    time_of_day: Literal["morning", "afternoon", "evening", "night"] | None = None


class RetrievedPlace(BaseModel):
    record: PlaceRecord
    semantic_score: float
    spatial_score: float
    theme_score: float
    contextual_score: float = 0.0
    contextual_evidence: list[str] = Field(default_factory=list)
    combined_score: float
    is_strong_match: bool = True
    match_reason: str = "Strong match: relevant retrieval result."


class QueryRequest(BaseModel):
    query: str
    lat: float
    lon: float
    top_k: int = Field(default=5, gt=0)
    max_distance_m: int = Field(default=1500, ge=0)
    requested_themes: list[str] = Field(default_factory=list)
    map_requested: bool = False
    context: QueryContext | None = None

    @field_validator("lat")
    @classmethod
    def latitude_range(cls, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return value

    @field_validator("lon")
    @classmethod
    def longitude_range(cls, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return value

    @field_validator("requested_themes")
    @classmethod
    def themes_are_list(cls, value: list[str]) -> list[str]:
        return _validate_theme_list(value)


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_places: list[RetrievedPlace]
    context: QueryContext | None = None
    map_path: str | None = None
    latency_ms: float | None = None
    fallback_used: bool = False


class DataQualityReport(BaseModel):
    total_records: int
    valid_records: int
    invalid_records: int
    warning_count: int
    warnings_by_type: dict[str, int] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class EDAReport(BaseModel):
    dataset_overview: dict[str, Any] = Field(default_factory=dict)
    spatial_overview: dict[str, Any] = Field(default_factory=dict)
    amenity_theme_overview: dict[str, Any] = Field(default_factory=dict)


class QueryOptions(BaseModel):
    top_k: int = 5
    max_distance_m: int = 1500
    requested_themes: list[str] = Field(default_factory=list)
    map_requested: bool = False
    use_graph: bool = True
    context: QueryContext | None = None


class DatasetBuildOptions(BaseModel):
    lat: float = 50.9350
    lon: float = -1.3960
    radius_m: int = 1500
    force_refresh: bool = False
    embedding_model: str | None = None
    embedding_backend: str | None = None
    onnx_model_dir: str | None = None


class BuildReport(BaseModel):
    records_built: int
    outputs: dict[str, str] = Field(default_factory=dict)


class ParsedIntent(BaseModel):
    intent: str | None = None
    requested_themes: list[str] = Field(default_factory=list)
    distance_constraint_m: int | None = None
    category_preference: str | None = None
    task_type: str = "place_recommendation"


class RankedPlace(BaseModel):
    record: PlaceRecord
    semantic_score: float = 0.0
    spatial_score: float = 0.0
    theme_score: float = 0.0
    accessibility_score: float = 0.0
    density_score: float = 0.0
    context_score: float = 0.0
    combined_score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class LLMProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...
