from __future__ import annotations

from geoinsight.llm.ollama_client import build_fallback_answer, select_places_for_llm
from geoinsight.schemas import QueryContext, RetrievedPlace

from geoinsight.llm.provider import LLMProvider


class TemplateLLMProvider(LLMProvider):
    """Deterministic no-LLM provider for free cloud demos."""

    def generate(self, prompt: str, **kwargs) -> str:
        retrieved_places: list[RetrievedPlace] = kwargs.get("retrieved_places", [])
        context: QueryContext | None = kwargs.get("context")
        prompt_places, no_strong_match = select_places_for_llm(retrieved_places)
        prefix = (
            "LLM provider is disabled in this deployment. "
            "Here is an evidence-based summary generated from retrieved records.\n\n"
        )
        return prefix + build_fallback_answer(prompt, prompt_places, context, no_strong_match)
