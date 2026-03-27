from medical_agent.neo4j_retriever import Neo4jRetriever


class FakeSession:
    def __init__(self, node_rows, relation_rows):
        self.node_rows = node_rows
        self.relation_rows = relation_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        if "queryNodes" in query:
            return self.node_rows
        return self.relation_rows


class FakeDriver:
    def __init__(self, node_rows, relation_rows):
        self.node_rows = node_rows
        self.relation_rows = relation_rows

    def session(self):
        return FakeSession(self.node_rows, self.relation_rows)


def test_retrieve_returns_normalized_relations():
    node_rows = [
        {"name": "胎儿偏小", "score": 0.91},
        {"name": "胎儿生长受限", "score": 0.84},
    ]
    relation_rows = [
        {
            "sub": "胎儿偏小",
            "rel": "病因",
            "obj": "胎盘功能异常",
            "neg": False,
            "reason": "解析原文：可能与胎盘功能异常相关",
        },
        {
            "sub": "胎儿生长受限",
            "rel": "诊断依据",
            "obj": "超声提示孕周偏小",
            "neg": False,
            "reason": "解析原文：超声提示孕周偏小",
        },
    ]

    retriever = Neo4jRetriever(
        driver=FakeDriver(node_rows, relation_rows),
        embed_query=lambda text: [0.1, 0.2, 0.3],
    )

    records = retriever.retrieve(
        "孕晚期胎儿偏小三周怎么办？",
        top_k=5,
        query_variant="原因向",
    )

    assert len(records) == 2
    assert records[0].sub == "胎儿偏小"
    assert records[0].score == 0.91
    assert records[0].query_variant == "原因向"
    assert records[0].matched_name == "胎儿偏小"
    assert records[1].obj == "超声提示孕周偏小"


def test_get_relations_for_nodes_returns_empty_when_no_nodes():
    retriever = Neo4jRetriever(
        driver=FakeDriver([], []),
        embed_query=lambda text: [0.1, 0.2, 0.3],
    )

    records = retriever.get_relations_for_nodes([])

    assert records == []
