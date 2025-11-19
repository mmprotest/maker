"""Configuration models for LLM and MAKER settings."""
from __future__ import annotations

import os
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible language model."""

    model: str
    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: str | None = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    temperature: float = 0.1
    max_output_tokens: int = 512
    request_timeout_seconds: int = 60


class MAKERConfig(BaseModel):
    """Configuration controlling MAKER voting and red flagging behavior."""

    k: int = 3
    max_votes_per_step: int = 50
    max_red_flag_retries_per_step: int = 100
    max_response_characters_for_red_flag: int = 4000
    log_level: str = "INFO"


def load_llm_config_from_env() -> LLMConfig:
    """Load an LLMConfig using environment variables where available."""

    model = os.getenv("MAKER_MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MAKER_BASE_URL")
    temperature = float(os.getenv("MAKER_TEMPERATURE", "0.1"))
    max_output_tokens = int(os.getenv("MAKER_MAX_OUTPUT_TOKENS", "512"))
    request_timeout_seconds = int(os.getenv("MAKER_REQUEST_TIMEOUT", "60"))
    return LLMConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        request_timeout_seconds=request_timeout_seconds,
    )


def load_maker_config_from_env_or_defaults() -> MAKERConfig:
    """Load MAKERConfig using environment variables or defaults."""

    k = int(os.getenv("MAKER_K", "3"))
    max_votes_per_step = int(os.getenv("MAKER_MAX_VOTES_PER_STEP", "50"))
    max_red_flag_retries_per_step = int(
        os.getenv("MAKER_MAX_RED_FLAG_RETRIES_PER_STEP", "100")
    )
    max_response_characters_for_red_flag = int(
        os.getenv("MAKER_MAX_RESPONSE_CHARS", "4000")
    )
    log_level = os.getenv("MAKER_LOG_LEVEL", "INFO")
    return MAKERConfig(
        k=k,
        max_votes_per_step=max_votes_per_step,
        max_red_flag_retries_per_step=max_red_flag_retries_per_step,
        max_response_characters_for_red_flag=max_response_characters_for_red_flag,
        log_level=log_level,
    )

