"""Neo4j 图检索模块。

这个模块只负责图检索侧的工作，不负责：
- 问题改写
- 结果融合
- 最终回答生成

它的职责是：
1. 通过注入的 embedding 函数把问题转成向量
2. 调用 Neo4j 向量索引召回相似节点
3. 围绕这些节点展开 `REL` 关系
4. 把结果整理成下游可直接使用的标准化记录
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .config import AgentConfig, load_config


class SupportsSession(Protocol):
    """本模块所需的 Neo4j driver 最小协议。"""

    def session(self) -> Any: ...


EmbedQueryFn = Callable[[str], list[float]]


@dataclass(frozen=True)
class RetrievedNode:
    """向量召回得到的节点记录。"""

    name: str
    score: float
    query_variant: str


@dataclass(frozen=True)
class RetrievedRelation:
    """关系展开后的标准化记录，供后续融合或格式化使用。"""

    sub: str
    rel: str
    obj: str
    neg: bool
    reason: str | None
    score: float
    query_variant: str
    matched_name: str


class Neo4jRetriever:
    """封装 Neo4j 的向量召回与关系展开逻辑。"""

    def __init__(
        self,
        driver: SupportsSession,
        embed_query: EmbedQueryFn,
        *,
        vector_index_name: str = "node_embedding",
    ) -> None:
        self.driver = driver
        self.embed_query = embed_query
        self.vector_index_name = vector_index_name

    # 练手建议 2：
    # 这个函数适合第二个重写。
    # 它的核心是：文本 -> 向量 -> Neo4j 向量检索 -> 标准化节点结果。
    # 你会练到参数校验、外部函数调用，以及数据库结果到 dataclass 的映射。
    def find_similar_nodes(
        self,
        question: str,
        *,
        top_k: int = 5,
        query_variant: str = "original",
    ) -> list[RetrievedNode]:
        """执行向量检索，返回命中节点及分数。"""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        # 这里不关心 embedding 是什么模型实现，
        # 只要求外部传进来的函数能够把文本转成向量即可。
        query_vec = self.embed_query(question.strip())
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes('{self.vector_index_name}', $k, $vec)
                YIELD node, score
                RETURN node.name AS name, score
                """,
                k=top_k,
                vec=query_vec,
            )
            return [
                RetrievedNode(
                    name=row["name"],
                    score=float(row["score"]),
                    query_variant=query_variant,
                )
                for row in result
                if row["name"]
            ]

    def get_relations_for_nodes(
        self,
        nodes: Sequence[RetrievedNode],
    ) -> list[RetrievedRelation]:
        """围绕命中节点展开 `REL` 关系，并整理为标准输出。"""
        if not nodes:
            return []

        # 为每个命中节点保留检索得分，后面关系记录会继承这个分数。
        score_map = {node.name: node.score for node in nodes}
        names = list(score_map.keys())
        query_variant = nodes[0].query_variant

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a)-[r:REL]->(b)
                WHERE a.name IN $names OR b.name IN $names
                RETURN a.name AS sub,
                       r.rel AS rel,
                       b.name AS obj,
                       r.neg AS neg,
                       r.reason AS reason
                """,
                names=names,
            )

            normalized: list[RetrievedRelation] = []
            for row in result:
                sub = row["sub"]
                obj = row["obj"]
                # 如果关系两端有一端来自召回节点，就用该节点的召回分数。
                matched_name = sub if sub in score_map else obj
                normalized.append(
                    RetrievedRelation(
                        sub=sub,
                        rel=row["rel"],
                        obj=obj,
                        neg=bool(row["neg"]),
                        reason=row["reason"],
                        score=score_map[matched_name],
                        query_variant=query_variant,
                        matched_name=matched_name,
                    )
                )
            return normalized

    # 练手建议 3：
    # 这是一个很适合在前两个函数都理解后再重写的“串联层”。
    # 它本身逻辑不重，主要作用是把“节点召回”和“关系展开”拼起来。
    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        query_variant: str = "original",
    ) -> list[RetrievedRelation]:
        """一次完整检索的便捷封装：先召回节点，再展开关系。"""
        nodes = self.find_similar_nodes(
            question,
            top_k=top_k,
            query_variant=query_variant,
        )
        return self.get_relations_for_nodes(nodes)


# 练手建议 4：
# 这一部分更偏工程装配层，建议最后再改。
# 它主要负责：读取配置、创建 Neo4j driver、把 embedding 函数注入 retriever。
def build_neo4j_retriever(
    embed_query: EmbedQueryFn,
    *,
    config: AgentConfig | None = None,
    vector_index_name: str = "node_embedding",
) -> Neo4jRetriever:
    """根据配置和 embedding 函数构造 `Neo4jRetriever`。"""
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "neo4j package is required for Neo4j retrieval. Install it first."
        ) from exc

    # 如果没有显式传配置，就默认从环境变量加载。
    cfg = config or load_config()
    driver = GraphDatabase.driver(
        cfg.neo4j_uri,
        auth=(cfg.neo4j_user, cfg.neo4j_password),
    )
    return Neo4jRetriever(
        driver=driver,
        embed_query=embed_query,
        vector_index_name=vector_index_name,
    )
