from __future__ import annotations

from geoinsight.eda.report_builder import build_eda_report, load_records
from geoinsight.features.feature_pipeline import load_place_features
from geoinsight.services.query_service import run_agent_query
from geoinsight.schemas import QueryOptions


def retrieve_places_tool(query: str, top_k: int = 5) -> dict:
    response = run_agent_query(query, QueryOptions(top_k=top_k, map_requested=False))
    return response.model_dump(mode="json")


def summarise_area_tool() -> dict:
    records = load_records()
    report = build_eda_report(records, write_charts=False)
    return report.model_dump(mode="json")


def compare_areas_tool() -> dict:
    features = load_place_features()
    return {
        "feature_rows": len(features),
        "note": "Area comparison currently uses available engineered feature summaries.",
    }


def generate_map_tool(query: str) -> dict:
    response = run_agent_query(query, QueryOptions(map_requested=True))
    return {"map_path": response.map_path}


def get_eda_summary_tool() -> dict:
    records = load_records()
    return build_eda_report(records, write_charts=False).model_dump(mode="json")
