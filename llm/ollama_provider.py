"""Ollama local LLM provider (reuses OpenAI-compatible endpoint)."""

from __future__ import annotations

from llm.openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """
    Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
    No API key is needed; pass a dummy value to satisfy the OpenAI client.
    """

    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.85,
        max_tokens: int = 256,
        max_history_turns: int = 10,
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama ignores the key
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_history_turns=max_history_turns,
            base_url=base_url,
        )
