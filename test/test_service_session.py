from __future__ import annotations

from dataclasses import dataclass

from medical_agent.retrieval_pipeline import EvidenceItem, RetrievalBundle, RetrievalOptions
from medical_agent.service import MedicalQAService
from medical_agent.session_memory import SessionMemoryStore


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    def get(self, key: str):
        return self.kv.get(key)

    def setex(self, key: str, ttl: int, value: str):  # noqa: ARG002
        self.kv[key] = value

    def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        if not values:
            return []
        size = len(values)
        norm_start = start if start >= 0 else max(0, size + start)
        norm_end = end if end >= 0 else size + end
        norm_end = min(norm_end, size - 1)
        if norm_start > norm_end:
            return []
        return values[norm_start : norm_end + 1]

    def ltrim(self, key: str, start: int, end: int):
        values = self.lists.get(key, [])
        if not values:
            return
        size = len(values)
        norm_start = start if start >= 0 else max(0, size + start)
        norm_end = end if end >= 0 else size + end
        norm_end = min(norm_end, size - 1)
        if norm_start > norm_end:
            self.lists[key] = []
            return
        self.lists[key] = values[norm_start : norm_end + 1]

    def expire(self, key: str, ttl: int):  # noqa: ARG002
        return None


@dataclass
class FakePipeline:
    calls: int = 0

    def retrieve(self, question: str, *, options: RetrievalOptions | None = None) -> RetrievalBundle:  # noqa: ARG002
        self.calls += 1
        return RetrievalBundle(
            original_question=question,
            query_variants=[question],
            evidence_items=[
                EvidenceItem(
                    text=f"{question} 对应的证据。",
                    source_type="text",
                    source_id=f"doc-{self.calls}",
                    query_variant=question,
                    graph_score=None,
                    faiss_score=0.7,
                    rerank_score=0.8,
                    final_score=0.8,
                )
            ],
        )


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "允许的 decision 只有：" in content and "answer" in content:
            return '{"decision":"answer","reason":"evidence is sufficient","focused_question":""}'
        return "ok"


def test_service_creates_session_and_persists_messages() -> None:
    pipeline = FakePipeline()
    llm = FakeLLM()
    session_store = SessionMemoryStore(client=FakeRedis(), enabled=True, max_history_turns=3)
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
        session_store=session_store,
    )

    first = service.ask("u1", "左心衰症状")
    assert first.session_id
    assert first.history_turns_used == 0
    assert pipeline.calls == 1

    second = service.ask("u1", "高血压检查", session_id=first.session_id)
    assert second.session_id == first.session_id
    assert second.history_turns_used == 1
    assert pipeline.calls == 2
    final_answer_messages = llm.calls[-1]
    assert "如果检索证据中已经存在能够直接回答原始问题的明确结论" in final_answer_messages[0]["content"]
    assert any(message["content"] == "左心衰症状" for message in final_answer_messages)
    assert any(message["content"] == "ok" for message in final_answer_messages)


def test_service_can_return_session_history_items() -> None:
    pipeline = FakePipeline()
    llm = FakeLLM()
    session_store = SessionMemoryStore(client=FakeRedis(), enabled=True, max_history_turns=3)
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
        session_store=session_store,
    )

    first = service.ask("u1", "左心衰症状")
    history = service.get_session_messages("u1", first.session_id)

    assert history.user_id == "u1"
    assert history.session_id == first.session_id
    assert len(history.messages) == 2
    assert history.messages[0]["role"] == "user"
    assert history.messages[0]["content"] == "左心衰症状"
    assert isinstance(history.messages[0]["ts"], int)
