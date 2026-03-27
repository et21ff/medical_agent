from __future__ import annotations

import os
from pathlib import Path

import pytest

from medical_agent.config import load_config
from medical_agent.neo4j_retriever import build_neo4j_retriever


def _load_dotenv_if_needed() -> None:
    """给 smoke test 做一个轻量兜底：环境里没有时，从项目根目录 .env 补变量。"""
    required_keys = {
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "EMBED_MODEL",
    }
    if required_keys.issubset(os.environ.keys()):
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _has_smoke_config() -> bool:
    _load_dotenv_if_needed()
    needed = [
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "EMBED_MODEL",
    ]
    return all(os.environ.get(key, "").strip() for key in needed)


@pytest.mark.skipif(
    not _has_smoke_config(),
    reason="Smoke test requires Neo4j and embedding config in environment or .env",
)
def test_neo4j_retriever_smoke() -> None:
    """真实 smoke test：加载 embedding，连接 Neo4j，执行一次完整检索。"""
    langchain_hf = pytest.importorskip("langchain_huggingface")

    cfg = load_config()
    embedding = langchain_hf.HuggingFaceEmbeddings(model_name=cfg.embed_model)
    retriever = build_neo4j_retriever(
        embed_query=embedding.embed_query,
        config=cfg,
    )

    question = "连续一周异常口渴，可能提示哪些健康问题？"
    records = retriever.retrieve(
        question,
        top_k=3,
        query_variant="smoke",
    )

    print(f"question: {question}")
    print(f"records: {len(records)}")
    for record in records[:5]:
        print(
            {
                "sub": record.sub,
                "rel": record.rel,
                "obj": record.obj,
                "neg": record.neg,
                "score": round(record.score, 4),
                "query_variant": record.query_variant,
                "matched_name": record.matched_name,
            }
        )

    assert isinstance(records, list)
    assert all(record.query_variant == "smoke" for record in records)
