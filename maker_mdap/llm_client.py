"""LLM client wrappers for OpenAI-compatible APIs."""
from __future__ import annotations

from typing import Callable

import openai

from .config import LLMConfig


class LLMClient:
    """Wrapper around the OpenAI chat completions API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url if config.base_url else None,
        )

    def generate_step_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Generate a completion for a single MAKER subtask."""

        completion = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens,
            timeout=self.config.request_timeout_seconds,
        )
        choice = completion.choices[0]
        return choice.message.content or ""


class FakeLLMClient:
    """Deterministic LLM client for tests and offline simulations."""

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder

    def generate_step_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        return self.responder(system_prompt, user_prompt)

