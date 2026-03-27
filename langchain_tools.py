from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .retrieval_pipeline import RetrievalBundle, RetrievalOptions, RetrievalPipeline


def format_retrieval_bundle(bundle: RetrievalBundle) -> str:
    """把检索结果整理成适合 LLM 阅读的文本。"""
    lines: list[str] = []
    lines.append(f"原始问题：{bundle.original_question}")
    if bundle.query_variants:
        lines.append("检索问题：")
        for idx, query in enumerate(bundle.query_variants, 1):
            lines.append(f"{idx}. {query}")

    if bundle.evidence_items:
        lines.append("融合证据：")
        for idx, item in enumerate(bundle.evidence_items, 1):
            score_text = f"{item.final_score:.4f}"
            lines.append(
                f"{idx}. [{item.source_type}] [score={score_text}] {item.text}"
            )
    else:
        if bundle.graph_evidence_texts:
            lines.append("图谱证据：")
            for idx, item in enumerate(bundle.graph_evidence_texts, 1):
                lines.append(f"{idx}. {item.text}")

        if bundle.text_results:
            lines.append("文本证据：")
            for idx, item in enumerate(bundle.text_results, 1):
                lines.append(f"{idx}. {item.text}")

    if not bundle.evidence_items and not bundle.graph_evidence_texts and not bundle.text_results:
        lines.append("未检索到有效证据。")

    lines.append("回答约束：")
    lines.append("1. 只能依据以上检索证据回答，不要补充证据中没有直接支持的医学结论。")
    lines.append("2. 如果证据偏题、证据不足或无法直接回答原始问题，必须明确说明“证据不足”或“未检索到直接证据”。")
    lines.append("3. 优先引用最直接相关的证据，不要因为医学常识相近而自行扩展到其他疾病、治疗或预防方案。")
    lines.append("4. 一旦判定为证据不足或未检索到直接证据，答案必须立即结束，不得补充常识性建议、预防措施或治疗方案。")

    return "\n".join(lines)


def retrieval_bundle_to_dict(bundle: RetrievalBundle) -> dict[str, Any]:
    """把检索结果转成结构化 dict，便于调试或 API 输出。"""
    return {
        "original_question": bundle.original_question,
        "query_variants": list(bundle.query_variants),
        "graph_results": [asdict(item) for item in bundle.graph_results],
        "graph_evidence_texts": [asdict(item) for item in bundle.graph_evidence_texts],
        "text_results": [asdict(item) for item in bundle.text_results],
        "evidence_items": [asdict(item) for item in bundle.evidence_items],
    }


def build_retrieval_tool(
    pipeline: RetrievalPipeline,
    *,
    default_options: RetrievalOptions | None = None,
    name: str = "retrieve_medical_evidence",
    description: str | None = None,
):
    """把 `RetrievalPipeline` 包装成 LangChain tool。

    这里不要求继承任何 LangChain 基类。
    只要把普通 Python 函数交给 `StructuredTool.from_function(...)` 即可。
    """
    langchain_tools = __import__("langchain_core.tools", fromlist=["StructuredTool"])

    tool_description = description or (
        "检索医疗问题相关证据。输入应是完整、明确的医疗问题。"
        "返回图谱证据和文本证据的整理结果，供后续回答使用。"
        "如果证据不足，后续回答必须明确说明不足，不得凭常识补充。"
    )
    options = default_options or RetrievalOptions()

    def retrieve_medical_evidence(question: str) -> str:
        """检索医疗问题相关证据。"""
        bundle = pipeline.retrieve(question, options=options)
        return format_retrieval_bundle(bundle)

    return langchain_tools.StructuredTool.from_function(
        func=retrieve_medical_evidence,
        name=name,
        description=tool_description,
    )


def build_retrieval_tool_json(
    pipeline: RetrievalPipeline,
    *,
    default_options: RetrievalOptions | None = None,
    name: str = "retrieve_medical_evidence_json",
    description: str | None = None,
):
    """返回 JSON 字符串的 tool 版本，便于调试。"""
    langchain_tools = __import__("langchain_core.tools", fromlist=["StructuredTool"])

    tool_description = description or (
        "检索医疗问题相关证据并返回 JSON 字符串。适合调试和结构化处理。"
    )
    options = default_options or RetrievalOptions()

    def retrieve_medical_evidence_json(question: str) -> str:
        """检索医疗问题相关证据，返回 JSON 字符串。"""
        bundle = pipeline.retrieve(question, options=options)
        return json.dumps(retrieval_bundle_to_dict(bundle), ensure_ascii=False)

    return langchain_tools.StructuredTool.from_function(
        func=retrieve_medical_evidence_json,
        name=name,
        description=tool_description,
    )
