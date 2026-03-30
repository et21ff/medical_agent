from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from .config import load_api_config, load_config, load_llm_config
from .embedding_provider import build_embedding_provider
from .langchain_tools import format_retrieval_bundle
from .neo4j_retriever import build_neo4j_retriever
from .rerank_provider import build_rerank_provider
from .retrieval_pipeline import RetrievalBundle, RetrievalOptions, RetrievalPipeline
from .vector_retriever import build_vector_retriever

SYSTEM_PROMPT = (
    "你是一名严谨的医疗问答助手。"
    "如果证据不足、证据偏题或无法直接支持结论，必须明确说明“证据不足”或“未检索到直接证据”。"
    "不要补充证据中没有直接支持的医学结论，不要把治疗表述升级为治愈。"
    "如果你判断为证据不足或未检索到直接证据，回答必须在该结论处结束，不得再补充常识建议、预防措施或治疗方案。"
)


class SupportsChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class ChatResult:
    answer: str
    evidence_preview: list[dict[str, Any]]
    query_variants: list[str]


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


@dataclass
class MedicalQAService:
    pipeline: RetrievalPipeline
    llm_client: SupportsChatClient
    default_options: RetrievalOptions
    evidence_preview_limit: int = 3

    def ask(
        self,
        question: str,
        *,
        options: RetrievalOptions | None = None,
    ) -> ChatResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        opts = options or self.default_options
        bundle = self.pipeline.retrieve(normalized_question, options=opts)
        evidence_text = format_retrieval_bundle(bundle)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{evidence_text}\n\n请严格遵循“回答约束”并直接给出最终答案。",
            },
        ]
        answer = self.llm_client.complete(messages)
        return ChatResult(
            answer=answer,
            evidence_preview=self._build_evidence_preview(bundle),
            query_variants=list(bundle.query_variants),
        )

    def _build_evidence_preview(self, bundle: RetrievalBundle) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for item in bundle.evidence_items[: self.evidence_preview_limit]:
            rows.append(
                {
                    "source": item.source_type,
                    "score": item.final_score,
                    "text": item.text,
                }
            )

        if rows:
            return rows

        for item in bundle.graph_evidence_texts[: self.evidence_preview_limit]:
            rows.append(
                {
                    "source": "graph",
                    "score": item.score,
                    "text": item.text,
                }
            )

        remaining = self.evidence_preview_limit - len(rows)
        if remaining > 0:
            for item in bundle.text_results[:remaining]:
                rows.append(
                    {
                        "source": "text",
                        "score": item.final_score,
                        "text": item.text,
                    }
                )
        return rows


def build_default_service() -> MedicalQAService:
    cfg = load_config()
    api_cfg = load_api_config()
    llm_cfg = load_llm_config()

    embedding_provider = build_embedding_provider(cfg)
    rerank_provider = build_rerank_provider(
        model_name=os.environ.get("RERANK_MODEL", "").strip() or "BAAI/bge-reranker-v2-m3"
    )

    pipeline = RetrievalPipeline(
        query_rewriter=None,
        neo4j_retriever=build_neo4j_retriever(
            embed_query=embedding_provider.embed_query,
            config=cfg,
        ),
        vector_retriever=build_vector_retriever(
            index_path=api_cfg.vector_index_path,
            meta_path=api_cfg.vector_meta_path,
            embedding_provider=embedding_provider,
            rerank_provider=None,
        ),
        evidence_rerank_provider=rerank_provider,
    )
    options = RetrievalOptions(
        use_rewrite=False,
        use_graph=True,
        use_text=True,
        graph_top_k=api_cfg.graph_top_k,
        text_top_k=api_cfg.text_top_k,
        text_recall_k=api_cfg.text_recall_k,
        evidence_top_k=api_cfg.evidence_top_k,
    )
    llm_client = OpenAIChatClient(
        base_url=llm_cfg.llm_base_url,
        api_key=llm_cfg.llm_api_key,
        model=llm_cfg.llm_model,
        timeout=llm_cfg.request_timeout,
    )
    return MedicalQAService(
        pipeline=pipeline,
        llm_client=llm_client,
        default_options=options,
        evidence_preview_limit=api_cfg.evidence_preview_limit,
    )

