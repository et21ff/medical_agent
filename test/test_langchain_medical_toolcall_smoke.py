from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from medical_agent.config import load_config, load_llm_config
from medical_agent.graph_text_formatter import GraphEvidenceText
from medical_agent.langchain_tools import build_retrieval_tool
from medical_agent.retrieval_pipeline import RetrievalBundle, RetrievalOptions
from medical_agent.embedding_provider import build_embedding_provider
from medical_agent.neo4j_retriever import build_neo4j_retriever
from medical_agent.rerank_provider import build_rerank_provider
from medical_agent.retrieval_pipeline import RetrievalPipeline
from medical_agent.vector_retriever import build_vector_retriever
from medical_agent.vector_retriever import RetrievedText


def _load_dotenv_if_needed() -> None:
    needed = {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"}
    if needed.issubset(os.environ.keys()):
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _has_langchain_llm_config() -> bool:
    _load_dotenv_if_needed()
    return bool(
        os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )


def _has_real_pipeline_config() -> bool:
    _load_dotenv_if_needed()
    needed = [
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "EMBED_MODEL",
    ]
    required_files = [
        Path("/root/llm_learning/data/EXAM/exam_rag_faiss.index"),
        Path("/root/llm_learning/data/EXAM/exam_rag_meta.jsonl"),
    ]
    return all(os.environ.get(key, "").strip() for key in needed) and all(
        path.exists() for path in required_files
    )


def _estimate_input_tokens(llm, messages) -> int | None:
    """尽量用 LangChain 自带接口估算输入 token 数。"""
    try:
        return int(llm.get_num_tokens_from_messages(messages))
    except Exception:
        return None


def _medical_answer_fewshot_messages() -> list[tuple[str, str]]:
    """用于约束“只按证据回答”的轻量 few-shot。"""
    return [
        (
            "human",
            """原始问题：关心和理解艾滋病病毒感染者，最需掌握的生活技能是
融合证据：
1. [graph] [score=-3.2000] 图谱证据：关系为“治疗”，与生活技能问题不直接对应
回答约束：
1. 只能依据以上检索证据回答，不要补充证据中没有直接支持的医学结论。
2. 如果证据偏题、证据不足或无法直接回答原始问题，必须明确说明“证据不足”或“未检索到直接证据”。
3. 优先引用最直接相关的证据，不要因为医学常识相近而自行扩展到其他疾病、治疗或预防方案。""",
        ),
        (
            "assistant",
            "未检索到直接证据，证据不足，无法直接回答该问题。",
        ),
        (
            "human",
            """原始问题：左心衰竭时，最早出现的症状是
融合证据：
1. [text] [score=1.1000] 题目：左心衰竭时，最早出现的症状是；正确选项：劳力性呼吸困难
回答约束：
1. 只能依据以上检索证据回答，不要补充证据中没有直接支持的医学结论。""",
        ),
        (
            "assistant",
            "根据检索证据，左心衰竭时最早出现的症状是劳力性呼吸困难。",
        ),
    ]


@dataclass
class FakePipeline:
    def retrieve(
        self, question: str, *, options: RetrievalOptions | None = None
    ) -> RetrievalBundle:
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
        )


@pytest.mark.skipif(
    not _has_langchain_llm_config(),
    reason="Smoke test requires DeepSeek/OpenAI-compatible LLM config in environment or .env",
)
def test_langchain_medical_retrieval_toolcall_smoke() -> None:
    langchain_openai = pytest.importorskip("langchain_openai")

    pipeline = FakePipeline()
    tool = build_retrieval_tool(
        pipeline,
        default_options=RetrievalOptions(
            use_rewrite=False, use_graph=True, use_text=True
        ),
    )

    cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools([tool])
    response = llm_with_tools.invoke(
        "你必须调用工具 retrieve_medical_evidence 来检索“左心衰竭最早出现的症状是什么”相关证据，先不要直接给最终解释。"
    )

    print("model:", cfg.llm_model)
    print("base_url:", cfg.llm_base_url)
    print("response_content:", response.content)
    print("tool_calls:", getattr(response, "tool_calls", None))

    assert hasattr(response, "tool_calls")
    assert response.tool_calls, "模型没有返回任何 tool call"
    assert response.tool_calls[0]["name"] == "retrieve_medical_evidence"
    assert "左心衰竭" in response.tool_calls[0]["args"]["question"]


@pytest.mark.skipif(
    not _has_langchain_llm_config(),
    reason="Smoke test requires DeepSeek/OpenAI-compatible LLM config in environment or .env",
)
def test_langchain_medical_retrieval_toolcall_roundtrip_smoke() -> None:
    """真实 smoke test：完成医疗检索 tool call -> tool result -> final answer 单轮闭环。"""
    langchain_openai = pytest.importorskip("langchain_openai")
    langchain_messages = pytest.importorskip("langchain_core.messages")

    pipeline = FakePipeline()
    tool = build_retrieval_tool(
        pipeline,
        default_options=RetrievalOptions(
            use_rewrite=False, use_graph=True, use_text=True
        ),
    )

    cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools([tool])
    user_input = "你必须调用工具 retrieve_medical_evidence 来检索“左心衰竭最早出现的症状是什么”相关证据，然后基于证据回答。"
    first_response = llm_with_tools.invoke(user_input)

    assert first_response.tool_calls, "第一轮没有返回 tool call"
    tool_call = first_response.tool_calls[0]
    bundle = pipeline.retrieve(
        tool_call["args"]["question"],
        options=RetrievalOptions(use_rewrite=False, use_graph=True, use_text=True),
    )
    tool_result = tool.invoke(tool_call["args"])

    tool_message = langchain_messages.ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],
    )
    second_round_messages = [
        *_medical_answer_fewshot_messages(),
        ("human", user_input),
        first_response,
        tool_message,
    ]
    input_tokens = _estimate_input_tokens(llm_with_tools, second_round_messages)

    second_response = llm_with_tools.invoke(second_round_messages)

    print("first_response_content:", first_response.content)
    print("first_tool_calls:", first_response.tool_calls)
    print("graph_count:", len(bundle.graph_evidence_texts))
    print("text_count:", len(bundle.text_results))
    print("input_tokens_est:", input_tokens)
    print("tool_result_preview:", str(tool_result)[:300])
    print("second_response_content:", second_response.content)

    assert isinstance(second_response.content, str)
    assert second_response.content.strip()
    assert "劳力性呼吸困难" in second_response.content


@pytest.mark.skipif(
    not (_has_langchain_llm_config() and _has_real_pipeline_config()),
    reason="Real pipeline smoke test requires LLM, Neo4j, embedding config, and local FAISS files",
)
def test_langchain_medical_retrieval_real_pipeline_roundtrip_smoke() -> None:
    """真实 smoke test：使用真实 RetrievalPipeline 完成单轮 tool call 闭环。"""
    langchain_openai = pytest.importorskip("langchain_openai")
    langchain_messages = pytest.importorskip("langchain_core.messages")

    cfg = load_config()
    embedding_provider = build_embedding_provider(cfg)
    rerank_provider = build_rerank_provider(
        model_name=os.environ.get("RERANK_MODEL", "").strip()
        or "BAAI/bge-reranker-v2-m3"
    )
    pipeline = RetrievalPipeline(
        query_rewriter=None,
        neo4j_retriever=build_neo4j_retriever(
            embed_query=embedding_provider.embed_query,
            config=cfg,
        ),
        vector_retriever=build_vector_retriever(
            index_path="/root/llm_learning/data/EXAM/exam_rag_faiss.index",
            meta_path="/root/llm_learning/data/EXAM/exam_rag_meta.jsonl",
            embedding_provider=embedding_provider,
            rerank_provider=None,
        ),
        evidence_rerank_provider=rerank_provider,
    )

    options = RetrievalOptions(
        use_rewrite=False,
        use_graph=True,
        use_text=True,
        graph_top_k=3,
        text_top_k=5,
        text_recall_k=20,
        evidence_top_k=5,
    )
    tool = build_retrieval_tool(
        pipeline,
        default_options=options,
    )

    llm_cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=llm_cfg.llm_model,
        api_key=llm_cfg.llm_api_key,
        base_url=llm_cfg.llm_base_url,
        timeout=llm_cfg.request_timeout,
        temperature=0.0,
    )

    system_prompt = """
你是一名严谨的医疗问答助手。如果证据不足、证据偏题或无法直接支持结论，必须明确说明“证据不足”或“未检索到直接证据”。不要补充证据中没有直接支持的医学结论，不要把治疗表述升级为治愈。
如果你判断为“证据不足”或“未检索到直接证据”，回答必须在该结论处结束，不得再补充任何常识建议、预防措施或治疗方案。"""

    llm_with_tools = llm.bind_tools([tool])
    user_input = (
        "请求医生提供如何预防结膜炎的方案。\n居住在高污染地区，容易长时间面对电脑屏幕。"
    )
    first_response = llm_with_tools.invoke(
        [
            ("system", system_prompt),
            ("human", user_input),
        ]
    )

    assert first_response.tool_calls, "第一轮没有返回 tool call"
    tool_call = first_response.tool_calls[0]
    bundle = pipeline.retrieve(tool_call["args"]["question"], options=options)
    tool_result = tool.invoke(tool_call["args"])

    tool_message = langchain_messages.ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],
    )
    second_round_messages = [
        ("system", system_prompt),
        *_medical_answer_fewshot_messages(),
        ("human", user_input),
        first_response,
        tool_message,
    ]
    second_response = llm_with_tools.invoke(second_round_messages)
    input_tokens = _estimate_input_tokens(llm_with_tools, second_round_messages)

    second_response = llm_with_tools.invoke(second_round_messages)

    print("real_first_tool_calls:", first_response.tool_calls)
    print("real_graph_count:", len(bundle.graph_evidence_texts))
    print("real_text_count:", len(bundle.text_results))
    print("real_evidence_count:", len(bundle.evidence_items))
    print("real_input_tokens_est:", input_tokens)
    print("real_tool_result_preview:", str(tool_result)[:2000])
    print("real_second_response_content:", second_response.content)

    assert isinstance(second_response.content, str)
    assert second_response.content.strip()
