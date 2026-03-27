"""向量库检索模块。

这个模块负责文本向量召回侧的工作，不负责：
- 问题改写
- 图检索
- 最终回答生成

它的职责是：
1. 通过注入的 embedding 函数把 query 编码成向量
2. 调用向量索引（当前是 FAISS）召回候选文本
3. 可选地通过 rerank provider 做精排
4. 把结果整理成标准化记录，供后续融合或格式化使用
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .embedding_provider import EmbeddingProvider, build_embedding_provider
from .rerank_provider import RerankProvider


class SupportsFaissIndex(Protocol):
    """本模块所需的向量索引最小协议。"""

    def search(self, x: Any, k: int) -> tuple[Any, Any]:
        ...


@dataclass(frozen=True)
class RetrievedText:
    """文本向量召回后的标准化记录。"""

    id: str
    question: str
    answer_text: str
    text: str
    faiss_score: float
    rerank_score: float | None
    final_score: float
    source: str


@dataclass
class VectorRetriever:
    """封装向量召回与可选 rerank。"""

    index: SupportsFaissIndex
    meta_rows: list[dict[str, Any]]
    embed_query: Any
    source: str = "exam_faiss"
    rerank_provider: RerankProvider | None = None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        recall_k: int = 20,
    ) -> list[RetrievedText]:
        """执行一次完整文本检索：先召回，再可选 rerank。"""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if recall_k <= 0:
            raise ValueError("recall_k must be > 0")

        import numpy as np

        normalized_question = question.strip()
        query_vector = self.embed_query(normalized_question)
        query_vector = _l2_normalize([float(value) for value in query_vector])
        scores, indices = self.index.search(
            np.array([query_vector], dtype="float32"),
            max(top_k, recall_k),
        )

        candidates: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_rows):
                continue
            row = self.meta_rows[idx]
            candidates.append(
                {
                    "row": row,
                    "faiss_score": float(score),
                }
            )

        if not candidates:
            return []

        if self.rerank_provider is None:
            return [
                RetrievedText(
                    id=str(item["row"].get("id", "")),
                    question=str(item["row"].get("question", "")),
                    answer_text=str(item["row"].get("answer_text", "")),
                    text=str(item["row"].get("text", "")),
                    faiss_score=float(item["faiss_score"]),
                    rerank_score=None,
                    final_score=float(item["faiss_score"]),
                    source=self.source,
                )
                for item in candidates[:top_k]
            ]

        documents = [str(item["row"].get("text", "")) for item in candidates]
        reranked = self.rerank_provider.rerank(
            normalized_question,
            documents,
            top_k=top_k,
        )

        results: list[RetrievedText] = []
        for item in reranked:
            candidate = candidates[item.index]
            row = candidate["row"]
            results.append(
                RetrievedText(
                    id=str(row.get("id", "")),
                    question=str(row.get("question", "")),
                    answer_text=str(row.get("answer_text", "")),
                    text=str(row.get("text", "")),
                    faiss_score=float(candidate["faiss_score"]),
                    rerank_score=float(item.score),
                    final_score=float(item.score),
                    source=self.source,
                )
            )
        return results


def load_meta_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def build_vector_retriever(
    *,
    index_path: str,
    meta_path: str,
    embedding_provider: EmbeddingProvider | None = None,
    rerank_provider: RerankProvider | None = None,
    source: str = "exam_faiss",
) -> VectorRetriever:
    """根据本地 FAISS 索引和元数据构造 `VectorRetriever`。"""
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for vector retrieval. Install `faiss-cpu` first."
        ) from exc

    index = faiss.read_index(index_path)
    meta_rows = load_meta_rows(Path(meta_path))
    provider = embedding_provider or build_embedding_provider()
    return VectorRetriever(
        index=index,
        meta_rows=meta_rows,
        embed_query=provider.embed_query,
        rerank_provider=rerank_provider,
        source=source,
    )


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return list(vector)
    return [float(value) / norm for value in vector]
