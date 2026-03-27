"""预生成问题改写结果的 rewriter。

这个模块用于评测或离线批处理场景：
- 先把一批问题的 rewrite 结果跑出来并落盘
- 后续检索时不再重复调用 LLM
- 直接按问题查表返回改写结果
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrecomputedQueryRewriter:
    """按问题查表返回预生成 rewrite。"""

    query_map: dict[str, list[str]]
    fallback_to_original: bool = False

    def rewrite(self, question: str) -> list[str]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        queries = self.query_map.get(normalized_question)
        if queries:
            cleaned = [query.strip() for query in queries if query and query.strip()]
            if cleaned:
                return cleaned

        if self.fallback_to_original:
            return [normalized_question]
        raise KeyError(f"question not found in precomputed rewrite map: {normalized_question}")


def load_rewrite_map(path: str | Path) -> dict[str, list[str]]:
    file_path = Path(path)
    query_map: dict[str, list[str]] = {}
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            question = str(row.get("question", "")).strip()
            queries = row.get("queries", [])
            if not question or not isinstance(queries, list):
                continue
            query_map[question] = [str(query).strip() for query in queries if str(query).strip()]
    return query_map


def build_precomputed_query_rewriter(
    path: str | Path,
    *,
    fallback_to_original: bool = False,
) -> PrecomputedQueryRewriter:
    return PrecomputedQueryRewriter(
        query_map=load_rewrite_map(path),
        fallback_to_original=fallback_to_original,
    )
