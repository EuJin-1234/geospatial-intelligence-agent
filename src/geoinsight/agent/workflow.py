from __future__ import annotations

from geoinsight.agent.nodes import GeoInsightNodes, SequentialGeoInsightWorkflow


def create_geoinsight_workflow(nodes: GeoInsightNodes | None = None):
    nodes = nodes or GeoInsightNodes()
    try:
        from langgraph.graph import END, START, StateGraph

        from geoinsight.agent.state import GeoInsightAgentState
    except ImportError:
        return SequentialGeoInsightWorkflow(nodes)

    graph = StateGraph(GeoInsightAgentState)
    graph.add_node("understand_query", nodes.understand_query)
    graph.add_node("retrieve_places", nodes.retrieve_places)
    graph.add_node("get_area_stats", nodes.get_area_stats)
    graph.add_node("compare_area_features", nodes.compare_area_features)
    graph.add_node("generate_map", nodes.generate_map)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("validate_answer", nodes.validate_answer)
    graph.add_edge(START, "understand_query")
    graph.add_conditional_edges(
        "understand_query",
        lambda state: state.get("task_type", "place_recommendation"),
        {
            "place_recommendation": "retrieve_places",
            "area_summary": "get_area_stats",
            "area_comparison": "compare_area_features",
            "data_question": "get_area_stats",
            "map_request": "generate_map",
        },
    )
    for node_name in ("retrieve_places", "get_area_stats", "compare_area_features", "generate_map"):
        graph.add_edge(node_name, "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")
    graph.add_edge("validate_answer", END)
    return graph.compile()
