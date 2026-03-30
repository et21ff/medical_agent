from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalOptionsPayload(BaseModel):
    use_rewrite: bool | None = None
    use_graph: bool | None = None
    use_text: bool | None = None
    graph_top_k: int | None = Field(default=None, gt=0)
    text_top_k: int | None = Field(default=None, gt=0)
    text_recall_k: int | None = Field(default=None, gt=0)
    evidence_top_k: int | None = Field(default=None, gt=0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    retrieval_options: RetrievalOptionsPayload | None = None


class ChatResponse(BaseModel):
    answer: str
    evidence_preview: list[dict[str, Any]]
    query_variants: list[str]
    request_id: str
    latency_ms: int

