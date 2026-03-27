
    # 练手建议 1：
    # 这是最推荐先自己重写的部分。
    # 它不需要真实 embedding，只依赖传入的节点列表和数据库返回行，
    # 很适合用 fake driver / fake session 来做单测。
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