from __future__ import annotations

import pytest

from geoinsight.llm.azure_provider import AzureOpenAIProvider
from geoinsight.llm.ollama_provider import OllamaProvider, get_llm_provider
from geoinsight.llm.template_provider import TemplateLLMProvider


def test_template_provider_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "template")

    provider = get_llm_provider()

    assert isinstance(provider, TemplateLLMProvider)
    assert "LLM provider is disabled" in provider.generate("query", retrieved_places=[])


def test_ollama_provider_still_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    provider = get_llm_provider()

    assert isinstance(provider, OllamaProvider)


def test_azure_provider_fails_clearly_without_required_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        get_llm_provider()


def test_azure_provider_placeholder_does_not_need_credentials():
    provider = AzureOpenAIProvider(endpoint="https://example.openai.azure.com", deployment="demo")

    with pytest.raises(NotImplementedError, match="prepared but not enabled"):
        provider.generate("hello")
