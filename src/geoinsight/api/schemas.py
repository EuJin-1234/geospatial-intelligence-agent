from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryAPIRequest(BaseModel):
    query: str
    intent: str | None = None
    weather: str | None = None
    time_of_day: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    max_distance_m: int = Field(default=1500, ge=100, le=10000)
    generate_map: bool = False
    use_agent: bool = True

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be empty")
        return value.strip()


class RetrievedPlaceAPI(BaseModel):
    name: str
    category: str | None = None
    distance_m: float | None = None
    semantic_score: float | None = None
    spatial_score: float | None = None
    theme_score: float | None = None
    accessibility_score: float | None = None
    density_score: float | None = None
    context_score: float | None = None
    combined_score: float | None = None
    themes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    is_strong_match: bool | None = None


class QueryAPIResponse(BaseModel):
    query: str
    answer: str
    task_type: str | None = None
    retrieved_places: list[RetrievedPlaceAPI] = Field(default_factory=list)
    map_path: str | None = None
    latency_ms: float | None = None
    provider: str | None = None
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    environment: str | None = None
    llm_provider: str | None = None
    checks: dict[str, bool]
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None


class ReportResponse(BaseModel):
    status: str
    path: str | None = None
    data: dict | None = None
    message: str | None = None


class FeaturePreviewResponse(BaseModel):
    status: str
    count: int
    records: list[dict]
