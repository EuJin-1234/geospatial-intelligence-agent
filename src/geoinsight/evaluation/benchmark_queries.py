from __future__ import annotations

import json
from pathlib import Path

from geoinsight.config import ensure_data_dirs, load_config
from geoinsight.evaluation.metrics import evaluate_response
from geoinsight.schemas import QueryOptions
from geoinsight.services.query_service import run_agent_query

BENCHMARK_QUERIES = [
    "Find study-friendly places near campus.",
    "Find relaxing outdoor places.",
    "Find food options near transport.",
    "Summarise the local area.",
    "Which area is better for studying?",
]


def run_benchmark(output_path: Path | None = None) -> Path:
    config = load_config()
    ensure_data_dirs(config)
    output_path = output_path or config.evaluation_report_path
    rows = []
    for query in BENCHMARK_QUERIES:
        try:
            response = run_agent_query(query, QueryOptions(map_requested=False))
            rows.append({"query": query, **evaluate_response(response)})
        except Exception as exc:
            rows.append({"query": query, "error": str(exc), "answer_generated": False})
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return output_path
