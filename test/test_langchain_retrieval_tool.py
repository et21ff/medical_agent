from __future__ import annotations

from dataclasses import dataclass

import pytest

from medical_agent.graph_text_formatter import GraphEvidenceText
from medical_agent.langchain_tools import build_retrieval_tool, build_retrieval_tool_json
from medical_agent.retrieval_pipeline import EvidenceItem, RetrievalBundle, RetrievalOptions
from medical_agent.vector_retriever import RetrievedText


@dataclass
class FakePipeline:
    last_question: str | None = None
    last_options: RetrievalOptions | None = None

    def retrieve(self, question: str, *, options: RetrievalOptions | None = None) -> RetrievalBundle:
        self.last_question = question
        self.last_options = options
        return RetrievalBundle(
            original_question=question,
            query_variants=[question, f"{question} 的相关证据"],
            graph_evidence_texts=[
                GraphEvidenceText(
                    text="图谱证据：左心衰竭相关早期症状可见劳力性呼吸困难。",
                    sub="左心衰竭",
                    rel="相关症状",
                    obj="劳力性呼吸困难",
                    neg=False,
                    reason="示例",
                    score=0.9,
                    query_variant="original",
                    matched_name="左心衰竭",
                    source="graph",
                )
            ],
            text_results=[
                RetrievedText(
                    id="exam_001",
                    question="左心衰竭时，最早出现的症状是",
                    answer_text="劳力性呼吸困难",
                    text="题目：左心衰竭时，最早出现的症状是\n正确选项：劳力性呼吸困难",
                    faiss_score=0.8,
                    rerank_score=1.1,
                    final_score=1.1,
                    source="exam_faiss",
                )
            ],
            evidence_items=[
                EvidenceItem(
                    text="图谱证据：左心衰竭相关早期症状可见劳力性呼吸困难。",
                    source_type="graph",
                    source_id="graph_1",
                    query_variant="original",
                    graph_score=0.9,
                    faiss_score=None,
                    rerank_score=2.3,
                    final_score=2.3,
                ),
                EvidenceItem(
                    text="题目：左心衰竭时，最早出现的症状是\n正确选项：劳力性呼吸困难",
                    source_type="text",
                    source_id="exam_001",
                    query_variant="text",
                    graph_score=None,
                    faiss_score=0.8,
                    rerank_score=1.1,
                    final_score=1.1,
                ),
            ],
        )


def test_build_retrieval_tool_invokes_pipeline() -> None:
    pytest.importorskip("langchain_core.tools")

    pipeline = FakePipeline()
    tool = build_retrieval_tool(
        pipeline,
        default_options=RetrievalOptions(use_rewrite=False, use_graph=True, use_text=True),
    )

    result = tool.invoke({"question": "左心衰竭最早症状是什么"})

    assert pipeline.last_question == "左心衰竭最早症状是什么"
    assert pipeline.last_options is not None
    assert "原始问题：左心衰竭最早症状是什么" in result
    assert "融合证据：" in result
    assert "[graph] [score=2.3000]" in result
    assert "[text] [score=1.1000]" in result
    assert "回答约束：" in result
    assert "证据不足" in result


def test_build_retrieval_tool_json_returns_json_string() -> None:
    pytest.importorskip("langchain_core.tools")

    pipeline = FakePipeline()
    tool = build_retrieval_tool_json(pipeline)

    result = tool.invoke({"question": "左心衰竭最早症状是什么"})

    assert isinstance(result, str)
    assert '"original_question": "左心衰竭最早症状是什么"' in result
    assert '"text_results"' in result
    assert '"evidence_items"' in result
