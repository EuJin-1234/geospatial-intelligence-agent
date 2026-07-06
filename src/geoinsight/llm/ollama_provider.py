from __future__ import annotations

from geoinsight.llm.ollama_client import OllamaClient

from geoinsight.config import load_config
from geoinsight.llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, client: OllamaClient | None = None):
        config = load_config()
        self.client = client or OllamaClient(config.legacy)

    def generate(self, prompt: str, **kwargs) -> str:
        retrieved_places = kwargs.get("retrieved_places", [])
        context = kwargs.get("context")
        return self.client.generate(prompt, retrieved_places, context).answer


def get_llm_provider() -> LLMProvider:
    config = load_config()
    if config.llm_provider == "template":
        from geoinsight.llm.template_provider import TemplateLLMProvider

        return TemplateLLMProvider()
    if config.llm_provider == "azure":
        from geoinsight.llm.azure_provider import AzureOpenAIProvider

        if (
            not config.azure_openai_endpoint
            or not config.azure_openai_deployment
            or not config.azure_openai_api_key
        ):
            raise RuntimeError(
                "Azure OpenAI provider selected but AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_DEPLOYMENT, and AZURE_OPENAI_API_KEY are not configured."
            )
        return AzureOpenAIProvider(
            endpoint=config.azure_openai_endpoint,
            deployment=config.azure_openai_deployment,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
        )
    if config.llm_provider != "ollama":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={config.llm_provider!r}. "
            "Use one of: ollama, azure, template."
        )
    return OllamaProvider()
