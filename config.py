from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class ConfigError(ValueError):
    """Raised when required runtime config is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    request_timeout: float = 30.0


@dataclass(frozen=True)
class AgentConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    embed_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    request_timeout: float = 30.0


def _read_required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"Missinos.environg required environment variable: {key}")
    return value


def _read_optional_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key} must be a float, got: {raw}") from exc
    if value <= 0:
        raise ConfigError(f"Environment variable {key} must be > 0, got: {value}")
    return value


def _read_preferred(
    env: Mapping[str, str],
    keys: list[str],
    *,
    required: bool,
    default: str = "",
) -> str:
    for key in keys:
        value = env.get(key, "").strip()
        if value:
            return value
    if required:
        joined = ", ".join(keys)
        raise ConfigError(f"Missing required environment variable (any of): {joined}")
    return default


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """
    Load LLM-only runtime config.

    Priority:
      - API key: LLM_API_KEY > VLLM_API_KEY > DEEPSEEK_API_KEY (required)
      - Base URL: LLM_BASE_URL > VLLM_BASE_URL > DEEPSEEK_BASE_URL > https://api.deepseek.com
      - Model: LLM_MODEL > VLLM_MODEL > DEEPSEEK_MODEL > deepseek-chat
      - Timeout: LLM_REQUEST_TIMEOUT (default 30.0)
    """
    source = env if env is not None else os.environ
    return LLMConfig(
        llm_base_url=_read_preferred(
            source,
            ["LLM_BASE_URL", "VLLM_BASE_URL", "DEEPSEEK_BASE_URL"],
            required=False,
            default="https://api.deepseek.com",
        ),
        llm_api_key=_read_preferred(
            source,
            ["LLM_API_KEY", "VLLM_API_KEY", "DEEPSEEK_API_KEY"],
            required=True,
        ),
        llm_model=_read_preferred(
            source,
            ["LLM_MODEL", "VLLM_MODEL", "DEEPSEEK_MODEL"],
            required=False,
            default="deepseek-chat",
        ),
        request_timeout=_read_optional_float(source, "LLM_REQUEST_TIMEOUT", 30.0),
    )


def load_config(env: Mapping[str, str] | None = None) -> AgentConfig:
    """
    Load and validate runtime config from environment variables.

    Required:
      - NEO4J_URI
      - NEO4J_USER
      - NEO4J_PASSWORD
      - EMBED_MODEL
      - LLM_BASE_URL
      - LLM_API_KEY
      - LLM_MODEL

    Optional:
      - LLM_REQUEST_TIMEOUT (default: 30.0 seconds)
    """
    source = env if env is not None else os.environ
    llm_cfg = load_llm_config(source)
    return AgentConfig(
        neo4j_uri=_read_required(source, "NEO4J_URI"),
        neo4j_user=_read_required(source, "NEO4J_USER"),
        neo4j_password=_read_required(source, "NEO4J_PASSWORD"),
        embed_model=_read_required(source, "EMBED_MODEL"),
        llm_base_url=llm_cfg.llm_base_url,
        llm_api_key=llm_cfg.llm_api_key,
        llm_model=llm_cfg.llm_model,
        request_timeout=llm_cfg.request_timeout,
    )
