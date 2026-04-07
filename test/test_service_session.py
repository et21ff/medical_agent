from __future__ import annotations

from dataclasses import dataclass

from medical_agent.retrieval_pipeline import RetrievalBundle, RetrievalOptions
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
        return RetrievalBundle(original_question=question, query_variants=[question])


class FakeLLM:
    def __init__(self) -> None:
        self.seen_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.seen_messages = messages
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

    first = service.ask("u1", "q1")
    assert first.session_id
    assert first.history_turns_used == 0
    assert pipeline.calls == 1

    second = service.ask("u1", "q2", session_id=first.session_id)
    assert second.session_id == first.session_id
    assert second.history_turns_used == 1
    assert pipeline.calls == 2
    assert any(message["content"] == "q1" for message in llm.seen_messages)
    assert any(message["content"] == "ok" for message in llm.seen_messages)
