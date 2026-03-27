"""Embedding 提供模块。

这个模块负责两件事：
1. 初始化 embedding 模型
2. 对外提供统一的 `embed_query(text)` 调用

这样做的目的是把“文本转向量”的能力独立出来，
避免 Neo4j 检索、工具层、Agent 层都直接依赖具体的 embedding 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import AgentConfig, load_config


class SupportsEmbedQuery(Protocol):
    """本模块所需的 embedding 客户端最小协议。"""

    def embed_query(self, text: str) -> list[float] | tuple[float, ...]:
        ...


@dataclass
class EmbeddingProvider:
    """对 embedding 客户端做一层轻量封装。"""

    client: SupportsEmbedQuery

    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本编码为向量。"""
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        vector = self.client.embed_query(text.strip())
        return [float(value) for value in vector]


def build_embedding_provider(config: AgentConfig | None = None) -> EmbeddingProvider:
    """根据配置构造 `EmbeddingProvider`。"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain_huggingface package is required for embedding. Install it first."
        ) from exc

    cfg = config or load_config()
    client = HuggingFaceEmbeddings(model_name=cfg.embed_model)
    return EmbeddingProvider(client=client)
