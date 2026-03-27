"""业务检索 pipeline。

这个模块负责把已经拆好的底层能力模块真正串起来：
- QueryRewriter
- Neo4jRetriever
- VectorRetriever

设计目标：
1. 只有一个统一 pipeline，不为每种组合单独写一个大函数
2. 通过“可选组件 + 可选参数”控制流程开关
3. 对外返回统一结构，便于上层 tool / agent / API 复用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .graph_text_formatter import GraphEvidenceText, GraphTextFormatter
from .neo4j_retriever import Neo4jRetriever, RetrievedRelation
from .query_rewriter import QueryRewriter
from .rerank_provider import RerankProvider
from .vector_retriever import RetrievedText, VectorRetriever


class SupportsRewrite(Protocol):
    def rewrite(self, question: str) -> list[str]:
        ...


class SupportsRerank(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[Any]:
        ...


@dataclass(frozen=True)
class RetrievalOptions:
    """一次检索调用的选项。"""

    use_rewrite: bool = True
    rewrite_min_chars: int = 0
    use_graph: bool = True
    use_text: bool = True
    graph_top_k: int = 5
    text_top_k: int = 5
    text_recall_k: int = 20
    evidence_top_k: int = 5


@dataclass(frozen=True)
class EvidenceItem:
    """统一证据项，便于图文融合与统一排序。"""

    text: str
    source_type: str
    source_id: str
    query_variant: str
    graph_score: float | None
    faiss_score: float | None
    rerank_score: float | None
    final_score: float


@dataclass(frozen=True)
class RetrievalBundle:
    """统一的检索输出。"""

    original_question: str
    query_variants: list[str]
    graph_results: list[RetrievedRelation] = field(default_factory=list)
    graph_evidence_texts: list[GraphEvidenceText] = field(default_factory=list)
    text_results: list[RetrievedText] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class RetrievalPipeline:
    """统一业务 pipeline。

    设计约束：
    - 组件都是可选的，缺失时跳过对应步骤
    - pipeline 自己不做模型初始化，假定外部传入的是长生命周期组件
    - 默认行为偏向“完整检索”：rewrite + graph + text
    """

    query_rewriter: SupportsRewrite | None = None
    neo4j_retriever: Neo4jRetriever | None = None
    vector_retriever: VectorRetriever | None = None
    graph_text_formatter: GraphTextFormatter | None = None
    evidence_rerank_provider: SupportsRerank | None = None

    def retrieve(
        self,
        question: str,
        *,
        options: RetrievalOptions | None = None,
    ) -> RetrievalBundle:
        """执行一次完整检索。"""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        opts = options or RetrievalOptions()
        normalized_question = question.strip()
        query_variants = self._build_query_variants(normalized_question, opts)

        graph_results: list[RetrievedRelation] = []
        if opts.use_graph and self.neo4j_retriever is not None:
            for idx, variant in enumerate(query_variants, 1):
                query_variant = "original" if len(query_variants) == 1 else f"rewrite_{idx}"
                graph_results.extend(
                    self.neo4j_retriever.retrieve(
                        variant,
                        top_k=opts.graph_top_k,
                        query_variant=query_variant,
                    )
                )

        graph_evidence_texts: list[GraphEvidenceText] = []
        if graph_results:
            formatter = self.graph_text_formatter or GraphTextFormatter()
            graph_evidence_texts = formatter.format_relations(graph_results)

        text_results: list[RetrievedText] = []
        if opts.use_text and self.vector_retriever is not None:
            text_results = self._retrieve_text_results(query_variants, opts)

        evidence_items = self._build_evidence_items(
            normalized_question,
            graph_evidence_texts,
            text_results,
            opts,
        )

        return RetrievalBundle(
            original_question=normalized_question,
            query_variants=query_variants,
            graph_results=graph_results,
            graph_evidence_texts=graph_evidence_texts,
            text_results=text_results,
            evidence_items=evidence_items,
        )

    def _build_query_variants(
        self,
        question: str,
        options: RetrievalOptions,
    ) -> list[str]:
        """根据配置决定是否调用 rewrite。"""
        if not options.use_rewrite or self.query_rewriter is None:
            return [question]
        if options.rewrite_min_chars > 0 and len(question) < options.rewrite_min_chars:
            return [question]

        queries = self.query_rewriter.rewrite(question)
        cleaned = [query.strip() for query in queries if query and query.strip()]
        return cleaned or [question]

    def _retrieve_text_results(
        self,
        query_variants: list[str],
        options: RetrievalOptions,
    ) -> list[RetrievedText]:
        """对 query variants 分别做文本召回，再统一去重排序。"""
        assert self.vector_retriever is not None

        merged: dict[str, dict[str, Any]] = {}
        for variant in query_variants:
            results = self.vector_retriever.retrieve(
                variant,
                top_k=options.text_recall_k,
                recall_k=options.text_recall_k,
            )
            for item in results:
                key = item.id or item.text
                existing = merged.get(key)
                if existing is None:
                    merged[key] = {
                        "item": item,
                        "hit_count": 1,
                        "max_faiss_score": item.faiss_score,
                    }
                    continue

                existing["hit_count"] += 1
                existing["max_faiss_score"] = max(existing["max_faiss_score"], item.faiss_score)
                if item.faiss_score > existing["item"].faiss_score:
                    existing["item"] = item

        ranked = sorted(
            merged.values(),
            key=lambda row: (
                row["hit_count"],
                row["max_faiss_score"],
                row["item"].final_score,
            ),
            reverse=True,
        )
        return [row["item"] for row in ranked[: options.text_top_k]]

    def _build_evidence_items(
        self,
        question: str,
        graph_evidence_texts: list[GraphEvidenceText],
        text_results: list[RetrievedText],
        options: RetrievalOptions,
    ) -> list[EvidenceItem]:
        """把图/文证据统一成一个候选池，并可选做统一 rerank。"""
        items: list[EvidenceItem] = []

        for idx, item in enumerate(graph_evidence_texts, 1):
            items.append(
                EvidenceItem(
                    text=item.text,
                    source_type="graph",
                    source_id=f"graph_{idx}",
                    query_variant=item.query_variant,
                    graph_score=item.score,
                    faiss_score=None,
                    rerank_score=None,
                    final_score=item.score,
                )
            )

        for item in text_results:
            items.append(
                EvidenceItem(
                    text=item.text,
                    source_type="text",
                    source_id=item.id,
                    query_variant="text",
                    graph_score=None,
                    faiss_score=item.faiss_score,
                    rerank_score=item.rerank_score,
                    final_score=item.final_score,
                )
            )

        if not items:
            return []

        if self.evidence_rerank_provider is not None:
            documents = [item.text for item in items]
            reranked = self.evidence_rerank_provider.rerank(
                question,
                documents,
                top_k=min(len(items), options.evidence_top_k),
            )
            ranked_items: list[EvidenceItem] = []
            for result in reranked:
                item = items[result.index]
                ranked_items.append(
                    EvidenceItem(
                        text=item.text,
                        source_type=item.source_type,
                        source_id=item.source_id,
                        query_variant=item.query_variant,
                        graph_score=item.graph_score,
                        faiss_score=item.faiss_score,
                        rerank_score=result.score,
                        final_score=result.score,
                    )
                )
            return ranked_items

        ranked = sorted(items, key=lambda item: item.final_score, reverse=True)
        return ranked[: options.evidence_top_k]
