from __future__ import annotations

from dataclasses import dataclass

from medical_agent.retrieval_pipeline import EvidenceItem, RetrievalBundle, RetrievalOptions
from medical_agent.service import (
    DEFAULT_CLARIFICATION_QUESTION,
    DEFAULT_INSUFFICIENT_ANSWER,
    MedicalQAService,
)


@dataclass
class FakePipeline:
    calls: int = 0
    last_question: str | None = None

    def retrieve(self, question: str, *, options: RetrievalOptions | None = None) -> RetrievalBundle:  # noqa: ARG002
        self.calls += 1
        self.last_question = question
        return RetrievalBundle(
            original_question=question,
            query_variants=[question],
            evidence_items=[
                EvidenceItem(
                    text="左心衰患者早期可出现活动后气促等表现。",
                    source_type="text",
                    source_id="doc-1",
                    query_variant=question,
                    graph_score=None,
                    faiss_score=0.9,
                    rerank_score=0.95,
                    final_score=0.95,
                )
            ],
        )


class FakeDecisionLLM:
    def __init__(self, *, precheck_json: str | None = None, judge_json: str | None = None, final_answer: str = "final") -> None:
        self.precheck_json = precheck_json
        self.judge_json = judge_json
        self.final_answer = final_answer
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        content = messages[-1]["content"]
        if "允许的 decision 只有：" in content and "retrieve_directly" in content:
            return self.precheck_json or '{"decision":"retrieve_directly","reason":"clear enough","focused_question":""}'
        if "允许的 decision 只有：" in content and "answer" in content:
            return self.judge_json or '{"decision":"answer","reason":"evidence sufficient","focused_question":""}'
        return self.final_answer


class FakeSessionStore:
    enabled = True

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self._messages = messages

    def create_session(self, user_id: str) -> str:  # noqa: ARG002
        return "session-1"

    def load_recent_messages(
        self,
        user_id: str,  # noqa: ARG002
        session_id: str,  # noqa: ARG002
        max_history_turns: int,  # noqa: ARG002
    ) -> list[dict[str, str]]:
        return list(self._messages)

    def append_message(
        self,
        user_id: str,  # noqa: ARG002
        session_id: str,  # noqa: ARG002
        role: str,  # noqa: ARG002
        content: str,  # noqa: ARG002
    ) -> None:
        return None


def test_rule_gate_clarification_skips_retrieval() -> None:
    pipeline = FakePipeline()
    llm = FakeDecisionLLM()
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
    )

    result = service.ask("u1", "它严重吗")

    assert result.answer == DEFAULT_CLARIFICATION_QUESTION
    assert result.retrieve_ms == 0
    assert result.query_variants == []
    assert pipeline.calls == 0


def test_short_followup_with_history_skips_rule_gate() -> None:
    pipeline = FakePipeline()
    llm = FakeDecisionLLM(
        judge_json='{"decision":"answer","reason":"history makes target clear","focused_question":""}',
        final_answer="history-aware-answer",
    )
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
        session_store=FakeSessionStore(
            [
                {"role": "user", "content": "左心衰严重吗"},
                {"role": "assistant", "content": "需要结合症状和检查判断。"},
            ]
        ),
    )

    result = service.ask("u1", "严重吗", session_id="session-1")

    assert pipeline.calls == 1
    assert result.answer == "history-aware-answer"


def test_long_question_can_trigger_rewrite_first() -> None:
    pipeline = FakePipeline()
    llm = FakeDecisionLLM(
        precheck_json='{"decision":"rewrite_first","reason":"too verbose","focused_question":"左心衰的早期症状有哪些"}',
        judge_json='{"decision":"answer","reason":"enough","focused_question":""}',
        final_answer="answer-after-rewrite",
    )
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
    )

    result = service.ask("u1", "我妈妈这几天总觉得胸闷乏力，之前还有高血压和糖尿病，请问是否存在左心衰风险，左心衰早期症状有哪些？")

    assert pipeline.calls == 1
    assert pipeline.last_question == "左心衰的早期症状有哪些"
    assert result.answer == "answer-after-rewrite"


def test_invalid_evidence_judge_falls_back_to_insufficient() -> None:
    pipeline = FakePipeline()
    llm = FakeDecisionLLM(judge_json="not-json")
    service = MedicalQAService(
        pipeline=pipeline,
        llm_client=llm,
        default_options=RetrievalOptions(use_rewrite=False),
    )

    result = service.ask("u1", "左心衰最早症状是什么")

    assert pipeline.calls == 1
    assert result.answer == DEFAULT_INSUFFICIENT_ANSWER
