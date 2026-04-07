from __future__ import annotations

from dataclasses import dataclass

from medical_agent.rag_cache import RAGCacheStore
from medical_agent.retrieval_pipeline import RetrievalBundle, RetrievalOptions
from medical_agent.service import MedicalQAService


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str):
        return self.data.get(key)

    def setex(self, key: str, ttl: int, value: str):  # noqa: ARG002
        self.data[key] = value


@dataclass
class FakePipeline:
    calls: int = 0

    def retrieve(self, question: str, *, options: RetrievalOptions | None = None) -> RetrievalBundle:  # noqa: ARG002
        self.calls += 1
        return RetrievalBundle(original_question=question, query_variants=[question])


class FakeLLM:
    def complete(self, messages: list[dict[str, str]]) -> str:  # noqa: ARG002
        return "ok"


def test_service_cache_miss_then_hit() -> None:
    pipeline = FakePipeline()
    store = RAGCacheStore(client=FakeRedis(), enabled=True, ttl_s=1800, key_version="v1", corpus_version="exam_v1")
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=FakeLLM(),
        default_options=RetrievalOptions(use_rewrite=False),
        cache_store=store,
    )

    first = service.ask("如何预防结膜炎")
    second = service.ask("如何预防结膜炎")

    assert pipeline.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.answer == "ok"
    assert second.answer == "ok"

