"""Rerank 提供模块。

这个模块负责把“query + 候选文档”进一步精排。

定位上，它和 `embedding_provider.py` 类似：
- embedding_provider 负责文本 -> 向量
- rerank_provider 负责 query/doc 对 -> 相关性分数

这样做的好处是：
1. 把 rerank 逻辑从具体检索脚本中抽离出来
2. 以后可以把当前模型换成别的 reranker，而不用改上层调用代码
3. 方便单独测试 rerank 行为
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SupportsRerankClient(Protocol):
    """本模块所需 rerank 客户端的最小协议。"""

    def score(self, query: str, document: str) -> float:
        ...


@dataclass(frozen=True)
class RerankResult:
    """单条候选经过 rerank 后的结果。"""

    index: int
    score: float
    document: str


@dataclass
class RerankProvider:
    """对 rerank 客户端做一层轻量封装。"""

    client: SupportsRerankClient

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """对候选文档做精排并返回排序后的结果。"""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not documents:
            return []

        normalized_query = query.strip()
        scored: list[RerankResult] = []
        for index, document in enumerate(documents):
            if not document or not document.strip():
                continue
            score = float(self.client.score(normalized_query, document.strip()))
            scored.append(
                RerankResult(
                    index=index,
                    score=score,
                    document=document,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return scored
        if top_k <= 0:
            return []
        return scored[:top_k]


class HuggingFaceCrossEncoderClient:
    """基于 Hugging Face sequence-classification 模型的简易 rerank 客户端。"""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer: Any,
        model: Any,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def score(self, query: str, document: str) -> float:
        """对单个 query/document 对打分。"""
        import torch

        encoded = self.tokenizer(
            query,
            document,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if logits.ndim == 2 and logits.shape[-1] == 1:
            return float(logits[0, 0].item())
        if logits.ndim == 2:
            return float(logits[0, -1].item())
        return float(logits.squeeze().item())


def build_rerank_provider(model_name: str = "BAAI/bge-reranker-v2-m3") -> RerankProvider:
    """构造基于 Hugging Face 本地模型的 rerank provider。"""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers and torch are required for reranking. Install them first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    client = HuggingFaceCrossEncoderClient(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    return RerankProvider(client=client)
