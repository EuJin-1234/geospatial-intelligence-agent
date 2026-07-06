from __future__ import annotations

from geoinsight.llm.provider import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """Placeholder for future Azure OpenAI integration."""

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-02-15-preview",
    ):
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_key = api_key
        self.api_version = api_version

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.endpoint or not self.deployment:
            raise RuntimeError(
                "Azure OpenAI provider selected but endpoint/deployment settings are missing."
            )
        raise NotImplementedError(
            "Azure OpenAI integration is prepared but not enabled yet. "
            "Set LLM_PROVIDER=azure and implement the Azure SDK call here."
        )
