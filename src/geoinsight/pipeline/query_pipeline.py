from __future__ import annotations

import time
from pathlib import Path

from geoinsight.config import DEFAULT_LAT, DEFAULT_LON, load_config
from geoinsight.indexing.embedder import create_embedder
from geoinsight.indexing.vector_store import FaissVectorStore
from geoinsight.llm.ollama_client import OllamaClient
from geoinsight.processing.contextual_thematic import infer_query_context
from geoinsight.retrieval.retriever import HybridRetriever
from geoinsight.schemas import QueryContext, QueryRequest, QueryResponse
from geoinsight.visualisation.map_builder import FoliumMapBuilder


def load_runtime(embedding_model: str | None = None, embedding_backend: str | None = None, onnx_model_dir: str | Path | None = None):
    config = load_config().legacy
    embedder = create_embedder(embedding_model or config.embedding_model_name, device=config.embedding_device, backend=embedding_backend or config.embedding_backend, onnx_model_dir=onnx_model_dir or config.onnx_embedding_model_dir)
    store = FaissVectorStore(embedder=embedder, config=config)
    store.load()
    return config, embedder, HybridRetriever(store, embedder)


def query_with_graph(request: QueryRequest, embedding_model: str | None = None) -> QueryResponse:
    return query_without_graph(request, embedding_model)


def query_without_graph(request: QueryRequest, embedding_model: str | None = None) -> QueryResponse:
    config, _, retriever = load_runtime(embedding_model)
    started = time.perf_counter()
    context = infer_query_context(request.query, request.context)
    retrieved = retriever.retrieve(request.query, request.top_k, request.max_distance_m, request.requested_themes, context)
    result = OllamaClient(config).generate(request.query, retrieved, context)
    map_path = str(FoliumMapBuilder(config).build_map(request.lat, request.lon, retrieved)) if request.map_requested else None
    return QueryResponse(query=request.query, answer=result.answer, retrieved_places=retrieved, context=context, map_path=map_path, latency_ms=round((time.perf_counter() - started) * 1000, 2), fallback_used=result.fallback_used)


def make_request(query: str, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, top_k: int = 5, max_distance_m: int = 1500, themes: list[str] | None = None, map_requested: bool = False, context: QueryContext | None = None) -> QueryRequest:
    return QueryRequest(query=query, lat=lat, lon=lon, top_k=top_k, max_distance_m=max_distance_m, requested_themes=themes or [], map_requested=map_requested, context=context)


def run_spatial_query(query: str, intent: str | None = None, weather: str | None = None, time_of_day: str | None = None, top_k: int = 5, max_distance_m: int = 1500, map_requested: bool = True, use_graph: bool = True) -> QueryResponse:
    context = QueryContext(intent=None if intent in {None, "auto"} else intent, weather=None if weather in {None, "none"} else weather, time_of_day=None if time_of_day in {None, "none"} else time_of_day)
    request = make_request(query=query, top_k=top_k, max_distance_m=max_distance_m, map_requested=map_requested, context=context)
    return query_with_graph(request) if use_graph else query_without_graph(request)