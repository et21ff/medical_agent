from __future__ import annotations

import uuid

from .config import load_api_config
from .retrieval_pipeline import RetrievalOptions
from .schemas import ChatRequest, ChatResponse, RetrievalOptionsPayload
from .service import MedicalQAService, build_default_service


def _merge_retrieval_options(
    base: RetrievalOptions,
    override: RetrievalOptionsPayload | None,
) -> RetrievalOptions:
    if override is None:
        return base
    return RetrievalOptions(
        use_rewrite=base.use_rewrite if override.use_rewrite is None else override.use_rewrite,
        use_graph=base.use_graph if override.use_graph is None else override.use_graph,
        use_text=base.use_text if override.use_text is None else override.use_text,
        graph_top_k=base.graph_top_k if override.graph_top_k is None else override.graph_top_k,
        text_top_k=base.text_top_k if override.text_top_k is None else override.text_top_k,
        text_recall_k=base.text_recall_k
        if override.text_recall_k is None
        else override.text_recall_k,
        evidence_top_k=base.evidence_top_k
        if override.evidence_top_k is None
        else override.evidence_top_k,
    )


def create_app(service: MedicalQAService | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "fastapi is required to run API app. Install fastapi and uvicorn first."
        ) from exc

    app = FastAPI(title="medical_agent", version="0.1.0")
    svc = service or build_default_service()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        request_id = str(uuid.uuid4())
        try:
            options = _merge_retrieval_options(svc.default_options, request.retrieval_options)
            result = svc.ask(request.question, options=options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

        return ChatResponse(
            answer=result.answer,
            evidence_preview=result.evidence_preview,
            query_variants=result.query_variants,
            cache_hit=result.cache_hit,
            retrieve_ms=result.retrieve_ms,
            llm_ms=result.llm_ms,
            total_ms=result.total_ms,
            request_id=request_id,
            latency_ms=result.total_ms,
        )

    return app


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "uvicorn is required to run API app. Install uvicorn first."
        ) from exc

    cfg = load_api_config()
    uvicorn.run(
        "medical_agent.api_app:create_app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
        factory=True,
    )


if __name__ == "__main__":
    main()
