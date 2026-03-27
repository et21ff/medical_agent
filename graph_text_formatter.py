"""图检索结果文本化模块。

这个模块的职责很单一：
- 把 `RetrievedRelation` 这种结构化图结果
- 转成后续更容易展示、融合、rerank 的文本证据

当前先采用“通用模板”文本化，不对不同关系类型做专门语义模板。
这样能先把图证据统一进文本候选空间，后面如果验证收益明显，
再针对高频关系做更自然的专门模板。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .neo4j_retriever import RetrievedRelation


@dataclass(frozen=True)
class GraphEvidenceText:
    """图关系文本化后的标准记录。"""

    text: str
    sub: str
    rel: str
    obj: str
    neg: bool
    reason: str | None
    score: float
    query_variant: str
    matched_name: str
    source: str = "graph"


@dataclass
class GraphTextFormatter:
    """把图关系结果转成统一文本证据。"""

    def format_relation(self, relation: RetrievedRelation) -> GraphEvidenceText:
        """把单条关系转成文本。"""
        if relation.neg:
            body = (
                f"图谱证据：在当前知识图谱中，"
                f"“{relation.sub}”与“{relation.obj}”之间不成立关系“{relation.rel}”。"
            )
        else:
            body = (
                f"图谱证据：在当前知识图谱中，"
                f"“{relation.sub}”与“{relation.obj}”之间存在关系“{relation.rel}”。"
            )

        if relation.reason and relation.reason.strip():
            body = f"{body} 说明：{relation.reason.strip()}"

        return GraphEvidenceText(
            text=body,
            sub=relation.sub,
            rel=relation.rel,
            obj=relation.obj,
            neg=relation.neg,
            reason=relation.reason,
            score=relation.score,
            query_variant=relation.query_variant,
            matched_name=relation.matched_name,
        )

    def format_relations(
        self,
        relations: Sequence[RetrievedRelation],
    ) -> list[GraphEvidenceText]:
        """批量文本化图关系。"""
        return [self.format_relation(relation) for relation in relations]
