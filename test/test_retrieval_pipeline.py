from dataclasses import dataclass, field

from medical_agent.retrieval_pipeline import RetrievalOptions, RetrievalPipeline


@dataclass(frozen=True)
class FakeRelation:
    sub: str
    rel: str
    obj: str
    neg: bool
    reason: str | None
    score: float
    query_variant: str
    matched_name: str


@dataclass(frozen=True)
class FakeText:
    id: str
    question: str
    answer_text: str
    text: str
    faiss_score: float
    rerank_score: float | None
    final_score: float
    source: str = "fake"


class FakeRewriter:
    def rewrite(self, question: str) -> list[str]:
        return [f"{question}-症状", f"{question}-治疗", f"{question}-风险"]


class FakeGraphRetriever:
    def retrieve(self, question: str, *, top_k: int = 5, query_variant: str = "original"):
        return [
            FakeRelation(
                sub=question,
                rel="相关关系",
                obj=f"top_{top_k}",
                neg=False,
                reason=f"{query_variant}:{question}:{top_k}",
                score=0.9,
                query_variant=query_variant,
                matched_name=question,
            )
        ]


@dataclass
class FakeVectorRetriever:
    calls: list[tuple[str, int, int]] = field(default_factory=list)

    def retrieve(self, question: str, *, top_k: int = 5, recall_k: int = 20):
        self.calls.append((question, top_k, recall_k))
        duplicate_id = "dup" if "治疗" in question or "风险" in question else question
        bonus = 0.2 if "治疗" in question else (0.1 if "风险" in question else 0.0)
        rows = [
            FakeText(
                id=duplicate_id,
                question=question,
                answer_text=f"answer:{question}",
                text=f"{question}:{top_k}:{recall_k}",
                faiss_score=0.5 + bonus,
                rerank_score=1.0 + bonus,
                final_score=1.0 + bonus,
            )
        ]
        if "症状" in question:
            rows.append(
                FakeText(
                    id="symptom-extra",
                    question=question,
                    answer_text=f"extra:{question}",
                    text=f"extra:{question}:{top_k}:{recall_k}",
                    faiss_score=0.95,
                    rerank_score=0.95,
                    final_score=0.95,
                )
            )
        return rows


class FakeRerankResult:
    def __init__(self, index: int, score: float) -> None:
        self.index = index
        self.score = score


class FakeEvidenceReranker:
    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None):
        scored = []
        for index, document in enumerate(documents):
            score = 10.0 if "图谱证据" in document else (5.0 if "extra" in document else 1.0)
            scored.append(FakeRerankResult(index=index, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        if top_k is None:
            return scored
        return scored[:top_k]


def test_pipeline_skips_rewrite_when_rewriter_is_none() -> None:
    retriever = FakeVectorRetriever()
    pipeline = RetrievalPipeline(
        query_rewriter=None,
        neo4j_retriever=FakeGraphRetriever(),
        vector_retriever=retriever,
    )

    bundle = pipeline.retrieve("口渴")

    assert bundle.query_variants == ["口渴"]
    assert len(bundle.graph_results) == 1
    assert len(bundle.graph_evidence_texts) == 1
    assert len(bundle.text_results) == 1
    assert retriever.calls == [("口渴", 20, 20)]


def test_pipeline_uses_rewriter_and_calls_graph_per_variant() -> None:
    pipeline = RetrievalPipeline(
        query_rewriter=FakeRewriter(),
        neo4j_retriever=FakeGraphRetriever(),
        vector_retriever=None,
    )

    bundle = pipeline.retrieve("咳嗽")

    assert bundle.query_variants == ["咳嗽-症状", "咳嗽-治疗", "咳嗽-风险"]
    assert [item.reason for item in bundle.graph_results] == [
        "rewrite_1:咳嗽-症状:5",
        "rewrite_2:咳嗽-治疗:5",
        "rewrite_3:咳嗽-风险:5",
    ]
    assert len(bundle.graph_evidence_texts) == 3


def test_pipeline_respects_options_and_can_disable_graph_or_text() -> None:
    retriever = FakeVectorRetriever()
    pipeline = RetrievalPipeline(
        query_rewriter=FakeRewriter(),
        neo4j_retriever=FakeGraphRetriever(),
        vector_retriever=retriever,
    )

    bundle = pipeline.retrieve(
        "发热",
        options=RetrievalOptions(
            use_rewrite=False,
            use_graph=False,
            use_text=True,
            text_top_k=3,
            text_recall_k=9,
        ),
    )

    assert bundle.query_variants == ["发热"]
    assert bundle.graph_results == []
    assert bundle.graph_evidence_texts == []
    assert [item.text for item in bundle.text_results] == ["发热:9:9"]
    assert retriever.calls == [("发热", 9, 9)]


def test_pipeline_uses_text_retrieval_per_variant_and_deduplicates() -> None:
    retriever = FakeVectorRetriever()
    pipeline = RetrievalPipeline(
        query_rewriter=FakeRewriter(),
        neo4j_retriever=None,
        vector_retriever=retriever,
    )

    bundle = pipeline.retrieve(
        "胸痛",
        options=RetrievalOptions(
            use_rewrite=True,
            use_graph=False,
            use_text=True,
            text_top_k=2,
            text_recall_k=7,
        ),
    )

    assert bundle.query_variants == ["胸痛-症状", "胸痛-治疗", "胸痛-风险"]
    assert retriever.calls == [
        ("胸痛-症状", 7, 7),
        ("胸痛-治疗", 7, 7),
        ("胸痛-风险", 7, 7),
    ]
    assert [item.id for item in bundle.text_results] == ["dup", "symptom-extra"]
    assert bundle.text_results[0].faiss_score == 0.7


def test_pipeline_skips_rewrite_for_short_questions_under_threshold() -> None:
    retriever = FakeVectorRetriever()
    pipeline = RetrievalPipeline(
        query_rewriter=FakeRewriter(),
        neo4j_retriever=None,
        vector_retriever=retriever,
    )

    bundle = pipeline.retrieve(
        "咳嗽",
        options=RetrievalOptions(
            use_rewrite=True,
            rewrite_min_chars=10,
            use_graph=False,
            use_text=True,
            text_top_k=2,
            text_recall_k=6,
        ),
    )

    assert bundle.query_variants == ["咳嗽"]
    assert retriever.calls == [("咳嗽", 6, 6)]


def test_pipeline_builds_unified_evidence_items_and_can_rerank_them() -> None:
    retriever = FakeVectorRetriever()
    pipeline = RetrievalPipeline(
        query_rewriter=None,
        neo4j_retriever=FakeGraphRetriever(),
        vector_retriever=retriever,
        evidence_rerank_provider=FakeEvidenceReranker(),
    )

    bundle = pipeline.retrieve(
        "graph优先",
        options=RetrievalOptions(
            use_rewrite=False,
            use_graph=True,
            use_text=True,
            graph_top_k=1,
            text_top_k=2,
            text_recall_k=2,
            evidence_top_k=2,
        ),
    )

    assert len(bundle.evidence_items) == 2
    assert bundle.evidence_items[0].source_type == "graph"
    assert bundle.evidence_items[0].rerank_score == 10.0
    assert bundle.evidence_items[1].source_type == "text"
