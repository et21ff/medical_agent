from __future__ import annotations

from dataclasses import dataclass

import pytest

from medical_agent.retrieval_pipeline import RetrievalOptions
from medical_agent.service import ChatResult


@dataclass
class FakeService:
    default_options: RetrievalOptions = RetrievalOptions(
        use_rewrite=False,
        use_graph=True,
        use_text=True,
        graph_top_k=3,
        text_top_k=5,
        text_recall_k=20,
        evidence_top_k=5,
    )
    seen_options: RetrievalOptions | None = None

    def ask(
        self,
        user_id: str,
        question: str,
        *,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> ChatResult:
        if question == "bad":
            raise ValueError("question must not be empty")
        if question == "boom":
            raise RuntimeError("upstream failed")
        self.seen_options = options
        return ChatResult(
            user_id=user_id,
            session_id=session_id or "sid-test",
            history_turns_used=0,
            answer=f"ans:{question}",
            evidence_preview=[{"source": "text", "score": 1.2, "text": "证据"}],
            query_variants=[question],
            cache_hit=False,
            retrieve_ms=12,
            llm_ms=34,
            total_ms=46,
        )


def _build_client():
    fastapi = pytest.importorskip("fastapi")
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from medical_agent.api_app import create_app

    service = FakeService()
    app = create_app(service=service)
    return fastapi_testclient.TestClient(app), service


def test_health_and_ready() -> None:
    client, _ = _build_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


def test_chat_success() -> None:
    client, _ = _build_client()
    response = client.post("/chat", json={"user_id": "u1", "question": "左心衰竭最早症状"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "u1"
    assert data["session_id"] == "sid-test"
    assert data["history_turns_used"] == 0
    assert data["answer"] == "ans:左心衰竭最早症状"
    assert data["evidence_preview"]
    assert data["query_variants"] == ["左心衰竭最早症状"]
    assert data["cache_hit"] is False
    assert data["retrieve_ms"] == 12
    assert data["llm_ms"] == 34
    assert data["total_ms"] == 46
    assert data["request_id"]


def test_chat_passes_override_options() -> None:
    client, service = _build_client()
    response = client.post(
        "/chat",
        json={
            "user_id": "u1",
            "question": "q",
            "retrieval_options": {
                "text_top_k": 2,
                "graph_top_k": 1,
            },
        },
    )
    assert response.status_code == 200
    assert service.seen_options is not None
    assert service.seen_options.text_top_k == 2
    assert service.seen_options.graph_top_k == 1
    assert service.seen_options.use_graph is True


def test_chat_value_error_to_400() -> None:
    client, _ = _build_client()
    response = client.post("/chat", json={"user_id": "u1", "question": "bad"})
    assert response.status_code == 400


def test_chat_unexpected_error_to_502() -> None:
    client, _ = _build_client()
    response = client.post("/chat", json={"user_id": "u1", "question": "boom"})
    assert response.status_code == 502


def test_chat_missing_user_id_to_422() -> None:
    client, _ = _build_client()
    response = client.post("/chat", json={"question": "q"})
    assert response.status_code == 422
