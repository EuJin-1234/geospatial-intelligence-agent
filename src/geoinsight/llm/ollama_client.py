from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from geoinsight.config import LegacyAppConfig, load_legacy_config
from geoinsight.llm.prompts import build_grounded_prompt
from geoinsight.schemas import QueryContext, RetrievedPlace


@dataclass(frozen=True)
class LLMResult:
    answer: str
    latency_ms: float
    fallback_used: bool


class OllamaClient:
    def __init__(self, config: LegacyAppConfig | None = None):
        self.config = config or load_legacy_config()

    def generate(self, query: str, retrieved_places: list[RetrievedPlace], context: QueryContext | None = None) -> LLMResult:
        prompt_places, no_strong = select_places_for_llm(retrieved_places)
        prompt = build_grounded_prompt(query, prompt_places, context, no_strong)
        started = time.perf_counter()
        try:
            response = requests.post(f"{self.config.ollama_base_url.rstrip('/')}/api/generate", json={"model": self.config.ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}, timeout=self.config.request_timeout_s)
            response.raise_for_status()
            answer = str(response.json().get("response", "")).strip()
            if not answer: raise ValueError("Ollama returned an empty response")
            return LLMResult(answer, (time.perf_counter() - started) * 1000, False)
        except Exception:
            return LLMResult(build_fallback_answer(query, prompt_places, context, no_strong), (time.perf_counter() - started) * 1000, True)


def select_places_for_llm(retrieved_places: list[RetrievedPlace], max_places: int = 3) -> tuple[list[RetrievedPlace], bool]:
    strong = [item for item in retrieved_places if item.is_strong_match]
    if strong: return strong[:max_places], False
    return retrieved_places[:max_places], bool(retrieved_places)


def build_fallback_answer(query: str, retrieved_places: list[RetrievedPlace], context: QueryContext | None = None, no_strong_match: bool = False) -> str:
    if not retrieved_places:
        return "Retrieved spatial context was limited, so I cannot recommend a specific place without inventing facts.\n\nCaveat:\n- OpenStreetMap metadata may be incomplete."
    best = retrieved_places[0]
    record = best.record
    lines = []
    if no_strong_match: lines.append("No strong match was found. The following are the closest weak matches.\n")
    lines.extend(["Best recommendation:", f"- {record.name}", f"- Reason: It is a {record.category} about {round(record.distance_to_origin_m)}m from the origin with themes: {', '.join(record.themes) if record.themes else 'none'}.", "", "Alternatives:"])
    alternatives = retrieved_places[1:3]
    if alternatives:
        for idx, item in enumerate(alternatives, start=1): lines.append(f"{idx}. {item.record.name} ({item.record.category}, {round(item.record.distance_to_origin_m)}m)")
    else:
        lines.append("Only one strong match was found. Lower-ranked results were excluded because they were weak matches." if not no_strong_match else "1. No additional weak alternatives were retrieved.")
    lines.extend(["", "Why this fits:", f"- Semantic match: semantic score {best.semantic_score:.3f}", f"- Spatial relevance: approximately {round(record.distance_to_origin_m)}m from the origin", f"- Theme/context match: themes={', '.join(record.themes) if record.themes else 'none'}; context intent={context.intent if context else None}", f"- Spatial relations: {record.spatial_context or 'none listed'}", "", "Caveat:", "- OpenStreetMap metadata may be incomplete."])
    return "\n".join(lines)