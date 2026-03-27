import numpy as np

from medical_agent.rerank_provider import RerankProvider
from medical_agent.vector_retriever import VectorRetriever


class FakeIndex:
    def __init__(self, scores, indices) -> None:
        self.scores = np.array([scores], dtype="float32")
        self.indices = np.array([indices], dtype="int64")

    def search(self, x, k):
        return self.scores[:, :k], self.indices[:, :k]


class FakeRerankClient:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query: str, document: str) -> float:
        return self.scores[(query, document)]


def test_vector_retriever_returns_faiss_results_without_rerank() -> None:
    retriever = VectorRetriever(
        index=FakeIndex([0.9, 0.8], [1, 0]),
        meta_rows=[
            {"id": "doc_0", "question": "Q0", "answer_text": "A0", "text": "文档0"},
            {"id": "doc_1", "question": "Q1", "answer_text": "A1", "text": "文档1"},
        ],
        embed_query=lambda _: [1.0, 0.0],
    )

    results = retriever.retrieve("口渴", top_k=2, recall_k=2)

    assert [item.id for item in results] == ["doc_1", "doc_0"]
    assert results[0].rerank_score is None
    assert results[0].final_score == results[0].faiss_score


def test_vector_retriever_uses_rerank_provider_for_final_order() -> None:
    rerank_provider = RerankProvider(
        client=FakeRerankClient(
            {
                ("咳嗽", "文档0"): 0.1,
                ("咳嗽", "文档1"): 0.9,
            }
        )
    )
    retriever = VectorRetriever(
        index=FakeIndex([0.95, 0.85], [0, 1]),
        meta_rows=[
            {"id": "doc_0", "question": "Q0", "answer_text": "A0", "text": "文档0"},
            {"id": "doc_1", "question": "Q1", "answer_text": "A1", "text": "文档1"},
        ],
        embed_query=lambda _: [0.0, 1.0],
        rerank_provider=rerank_provider,
    )

    results = retriever.retrieve("咳嗽", top_k=2, recall_k=2)

    assert [item.id for item in results] == ["doc_1", "doc_0"]
    assert results[0].rerank_score == 0.9
    assert results[0].final_score == 0.9


def test_vector_retriever_validates_question_and_k() -> None:
    retriever = VectorRetriever(
        index=FakeIndex([0.9], [0]),
        meta_rows=[{"id": "doc_0", "question": "Q0", "answer_text": "A0", "text": "文档0"}],
        embed_query=lambda _: [1.0],
    )

    try:
        retriever.retrieve("   ")
        assert False, "expected ValueError for empty question"
    except ValueError:
        pass

    try:
        retriever.retrieve("发热", top_k=0)
        assert False, "expected ValueError for invalid top_k"
    except ValueError:
        pass
