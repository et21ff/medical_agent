from __future__ import annotations

from medical_agent.graph_text_formatter import GraphEvidenceText
from medical_agent.rag_cache import RAGCacheStore, build_cache_key, normalize_query
from medical_agent.retrieval_pipeline import EvidenceItem, RetrievalBundle, RetrievalOptions
from medical_agent.vector_retriever import RetrievedText


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str):  # noqa: ARG002
        self.data[key] = value


class BrokenRedis:
    def get(self, key: str):  # noqa: ARG002
        raise RuntimeError("boom")

    def setex(self, key: str, ttl: int, value: str):  # noqa: ARG002
        raise RuntimeError("boom")


def _bundle() -> RetrievalBundle:
    return RetrievalBundle(
        original_question="q",
        query_variants=["q"],
        graph_evidence_texts=[
            GraphEvidenceText(
                text="g",
                sub="s",
                rel="r",
                obj="o",
                neg=False,
                reason="x",
                score=1.1,
                query_variant="original",
                matched_name="s",
            )
        ],
        text_results=[
            RetrievedText(
                id="id1",
                question="q1",
                answer_text="a1",
                text="t1",
                faiss_score=0.7,
                rerank_score=0.9,
                final_score=0.9,
                source="exam_faiss",
            )
        ],
        evidence_items=[
            EvidenceItem(
                text="e1",
                source_type="text",
                source_id="id1",
                query_variant="text",
                graph_score=None,
                faiss_score=0.7,
                rerank_score=0.9,
                final_score=0.9,
            )
        ],
    )


def test_build_cache_key_is_stable() -> None:
    opts = RetrievalOptions(use_rewrite=False, graph_top_k=3, text_top_k=5, text_recall_k=20, evidence_top_k=5)
    q = normalize_query("  如何 预防 结膜炎 ")
    key1 = build_cache_key(q, opts, corpus_version="exam_v1", key_version="v1")
    key2 = build_cache_key(q, opts, corpus_version="exam_v1", key_version="v1")
    assert key1 == key2


def test_rag_cache_roundtrip() -> None:
    store = RAGCacheStore(client=FakeRedis(), enabled=True, ttl_s=100, key_version="v1", corpus_version="exam_v1")
    opts = RetrievalOptions(use_rewrite=False)
    key = store.build_cache_key(normalize_query("q"), opts)
    bundle = _bundle()
    assert store.set(key, bundle)
    restored = store.get(key)
    assert restored is not None
    assert restored.original_question == "q"
    assert restored.query_variants == ["q"]
    assert restored.evidence_items[0].text == "e1"


def test_rag_cache_fails_open_when_redis_errors() -> None:
    store = RAGCacheStore(client=BrokenRedis(), enabled=True)
    key = "rag:v1:k"
    assert store.get(key) is None
    assert store.set(key, _bundle()) is False

