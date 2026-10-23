from typing import Protocol

from anthropic import Anthropic

from docmind.config import settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._model = model or settings.llm_model

    def complete(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in message.content if b.type == "text")