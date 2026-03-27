from __future__ import annotations

import sys
import types

import pytest

from medical_agent.config import AgentConfig
from medical_agent.embedding_provider import EmbeddingProvider, build_embedding_provider


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.last_text: str | None = None

    def embed_query(self, text: str):
        self.last_text = text
        return (1, 2.5, 3)


def test_embed_query_strips_text_and_returns_float_list() -> None:
    client = FakeEmbeddingClient()
    provider = EmbeddingProvider(client=client)

    vector = provider.embed_query("  孕晚期胎儿偏小三周怎么办？  ")

    assert client.last_text == "孕晚期胎儿偏小三周怎么办？"
    assert vector == [1.0, 2.5, 3.0]


def test_embed_query_rejects_empty_text() -> None:
    provider = EmbeddingProvider(client=FakeEmbeddingClient())

    with pytest.raises(ValueError):
        provider.embed_query("   ")


def test_build_embedding_provider_uses_embed_model_from_config(monkeypatch) -> None:
    calls: dict[str, str] = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name

        def embed_query(self, text: str):
            return [0.1, 0.2, 0.3]

    fake_module = types.SimpleNamespace(HuggingFaceEmbeddings=FakeHuggingFaceEmbeddings)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_module)

    cfg = AgentConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        embed_model="shibing624/text2vec-base-chinese",
        llm_base_url="https://api.deepseek.com",
        llm_api_key="sk-test",
        llm_model="deepseek-chat",
        request_timeout=30.0,
    )

    provider = build_embedding_provider(config=cfg)

    assert calls["model_name"] == "shibing624/text2vec-base-chinese"
    assert provider.embed_query("test") == [0.1, 0.2, 0.3]
