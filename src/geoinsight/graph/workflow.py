from __future__ import annotations

from geoinsight.graph.nodes import GeoQueryNodes


class SequentialGeoQueryWorkflow:
    def __init__(self, nodes: GeoQueryNodes):
        self.nodes = nodes

    def invoke(self, state: dict) -> dict:
        state = self.nodes.infer_context(state)
        state = self.nodes.embed_query(state)
        state = self.nodes.retrieve_places(state)
        state = self.nodes.evaluate_retrieval(state)
        state = self.nodes.generate_answer(state) if state.get("retrieval_quality") == "good" else self.nodes.fallback_answer(state)
        if state.get("map_requested"):
            state = self.nodes.build_map(state)
        return state


def create_geo_query_workflow(nodes: GeoQueryNodes):
    try:
        from langgraph.graph import END, START, StateGraph
        from geoinsight.graph.state import GeoQueryState
    except ImportError:
        return SequentialGeoQueryWorkflow(nodes)
    graph = StateGraph(GeoQueryState)
    graph.add_node("infer_context", nodes.infer_context)
    graph.add_node("embed_query", nodes.embed_query)
    graph.add_node("retrieve_places", nodes.retrieve_places)
    graph.add_node("evaluate_retrieval", nodes.evaluate_retrieval)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("fallback_answer", nodes.fallback_answer)
    graph.add_node("build_map", nodes.build_map)
    graph.add_edge(START, "infer_context")
    graph.add_edge("infer_context", "embed_query")
    graph.add_edge("embed_query", "retrieve_places")
    graph.add_edge("retrieve_places", "evaluate_retrieval")
    graph.add_conditional_edges("evaluate_retrieval", lambda state: "good" if state.get("retrieval_quality") == "good" else "weak", {"good": "generate_answer", "weak": "fallback_answer"})
    graph.add_conditional_edges("generate_answer", lambda state: "map" if state.get("map_requested") else "end", {"map": "build_map", "end": END})
    graph.add_conditional_edges("fallback_answer", lambda state: "map" if state.get("map_requested") else "end", {"map": "build_map", "end": END})
    graph.add_edge("build_map", END)
    return graph.compile()