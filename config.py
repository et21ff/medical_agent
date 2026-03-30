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


@dataclass(frozen=True)
class APIConfig:
    host: str
    port: int
    debug: bool
    evidence_preview_limit: int
    vector_index_path: str
    vector_meta_path: str
    graph_top_k: int
    text_top_k: int
    text_recall_k: int
    evidence_top_k: int


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


def _read_optional_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {key} must be an int, got: {raw}") from exc
    if value <= 0:
        raise ConfigError(f"Environment variable {key} must be > 0, got: {value}")
    return value


def _read_optional_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Environment variable {key} must be a boolean, got: {raw}")


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


def load_api_config(env: Mapping[str, str] | None = None) -> APIConfig:
    """
    Load API runtime config.

    Optional environment variables:
      - API_HOST (default: 0.0.0.0)
      - API_PORT (default: 8080)
      - API_DEBUG (default: false)
      - EVIDENCE_PREVIEW_LIMIT (default: 3)
      - VECTOR_INDEX_PATH (default: /root/llm_learning/data/EXAM/exam_rag_faiss.index)
      - VECTOR_META_PATH (default: /root/llm_learning/data/EXAM/exam_rag_meta.jsonl)
      - GRAPH_TOP_K (default: 3)
      - TEXT_TOP_K (default: 5)
      - TEXT_RECALL_K (default: 20)
      - EVIDENCE_TOP_K (default: 5)
    """
    source = env if env is not None else os.environ
    return APIConfig(
        host=source.get("API_HOST", "").strip() or "0.0.0.0",
        port=_read_optional_int(source, "API_PORT", 8080),
        debug=_read_optional_bool(source, "API_DEBUG", False),
        evidence_preview_limit=_read_optional_int(source, "EVIDENCE_PREVIEW_LIMIT", 3),
        vector_index_path=source.get("VECTOR_INDEX_PATH", "").strip()
        or "/root/llm_learning/data/EXAM/exam_rag_faiss.index",
        vector_meta_path=source.get("VECTOR_META_PATH", "").strip()
        or "/root/llm_learning/data/EXAM/exam_rag_meta.jsonl",
        graph_top_k=_read_optional_int(source, "GRAPH_TOP_K", 3),
        text_top_k=_read_optional_int(source, "TEXT_TOP_K", 5),
        text_recall_k=_read_optional_int(source, "TEXT_RECALL_K", 20),
        evidence_top_k=_read_optional_int(source, "EVIDENCE_TOP_K", 5),
    )
