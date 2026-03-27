from medical_agent.rerank_provider import RerankProvider


class FakeRerankClient:
    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores

    def score(self, query: str, document: str) -> float:
        return self.scores[(query, document)]


def test_rerank_provider_sorts_by_score_desc() -> None:
    provider = RerankProvider(
        client=FakeRerankClient(
            {
                ("口渴", "文档A"): 0.2,
                ("口渴", "文档B"): 0.9,
                ("口渴", "文档C"): 0.5,
            }
        )
    )

    results = provider.rerank("口渴", ["文档A", "文档B", "文档C"])

    assert [item.document for item in results] == ["文档B", "文档C", "文档A"]
    assert [item.index for item in results] == [1, 2, 0]


def test_rerank_provider_respects_top_k() -> None:
    provider = RerankProvider(
        client=FakeRerankClient(
            {
                ("咳嗽", "文档A"): 0.6,
                ("咳嗽", "文档B"): 0.7,
                ("咳嗽", "文档C"): 0.3,
            }
        )
    )

    results = provider.rerank("咳嗽", ["文档A", "文档B", "文档C"], top_k=2)

    assert len(results) == 2
    assert [item.document for item in results] == ["文档B", "文档A"]


def test_rerank_provider_skips_empty_documents_and_validates_query() -> None:
    provider = RerankProvider(
        client=FakeRerankClient(
            {
                ("发热", "文档A"): 0.8,
            }
        )
    )

    results = provider.rerank("发热", ["", "文档A", "   "])
    assert len(results) == 1
    assert results[0].document == "文档A"

    try:
        provider.rerank("   ", ["文档A"])
        assert False, "expected ValueError for empty query"
    except ValueError:
        pass
