from __future__ import annotations

import time

from geoinsight.agent.tools import get_eda_summary_tool
from geoinsight.retrieval.intent_parser import parse_intent
from geoinsight.services.query_service import run_agent_query
from geoinsight.schemas import QueryOptions


class GeoInsightNodes:
    def understand_query(self, state: dict) -> dict:
        parsed = parse_intent(state["query"])
        state["parsed_intent"] = parsed.model_dump(mode="json")
        state["task_type"] = parsed.task_type
        return state

    def retrieve_places(self, state: dict) -> dict:
        response = run_agent_query(
            state["query"],
            QueryOptions(
                top_k=state.get("top_k", 5),
                max_distance_m=state.get("max_distance_m", 1500),
                requested_themes=state.get("parsed_intent", {}).get("requested_themes", []),
                map_requested=False,
                use_graph=False,
            ),
        )
        state["retrieved_places"] = [item.model_dump(mode="json") for item in response.retrieved_places]
        state["context"] = response.context.model_dump(mode="json", exclude_none=True) if response.context else {}
        state["answer"] = response.answer
        state["latency_ms"] = response.latency_ms
        return state

    def get_area_stats(self, state: dict) -> dict:
        state["eda_summary"] = get_eda_summary_tool()
        return state

    def compare_area_features(self, state: dict) -> dict:
        state["comparison_result"] = {"summary": "Comparison is based on engineered amenity features."}
        return state

    def generate_map(self, state: dict) -> dict:
        response = run_agent_query(state["query"], QueryOptions(map_requested=True, use_graph=False))
        state["map_path"] = response.map_path
        state["retrieved_places"] = [item.model_dump(mode="json") for item in response.retrieved_places]
        return state

    def generate_answer(self, state: dict) -> dict:
        if state.get("answer"):
            return state
        task = state.get("task_type")
        if task == "data_question":
            overview = state.get("eda_summary", {}).get("dataset_overview", {})
            state["answer"] = f"The dataset contains {overview.get('cleaned_places', 0)} cleaned places. Top categories are {overview.get('category_counts', {})}."
        elif task == "area_summary":
            themes = state.get("eda_summary", {}).get("amenity_theme_overview", {}).get("theme_place_counts", {})
            state["answer"] = f"The local area contains a mix of amenities. Theme counts: {themes}."
        elif task == "area_comparison":
            state["answer"] = state.get("comparison_result", {}).get("summary", "Comparison is unavailable.")
        else:
            state["answer"] = "GeoInsight completed the requested analysis."
        return state

    def validate_answer(self, state: dict) -> dict:
        if not state.get("answer"):
            state.setdefault("errors", []).append("No answer generated")
            state["answer"] = "I could not generate a grounded answer from the available data."
        state.setdefault("latency_ms", 0.0)
        return state


class SequentialGeoInsightWorkflow:
    def __init__(self, nodes: GeoInsightNodes):
        self.nodes = nodes

    def invoke(self, state: dict) -> dict:
        start = time.perf_counter()
        state = self.nodes.understand_query(state)
        task = state.get("task_type")
        if task == "data_question":
            state = self.nodes.get_area_stats(state)
        elif task == "area_summary":
            state = self.nodes.get_area_stats(state)
        elif task == "area_comparison":
            state = self.nodes.compare_area_features(state)
        elif task == "map_request":
            state = self.nodes.generate_map(state)
        else:
            state = self.nodes.retrieve_places(state)
        state = self.nodes.generate_answer(state)
        state = self.nodes.validate_answer(state)
        state["latency_ms"] = state.get("latency_ms") or round((time.perf_counter() - start) * 1000, 2)
        return state
