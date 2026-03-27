from pathlib import Path

from medical_agent.precomputed_rewriter import (
    build_precomputed_query_rewriter,
    load_rewrite_map,
)


def test_load_rewrite_map_reads_question_queries(tmp_path: Path) -> None:
    path = tmp_path / "rewrite_cache.jsonl"
    path.write_text(
        '{"question":"口渴怎么办","queries":["口渴怎么办","口渴提示什么问题","口渴需要做什么检查"]}\n',
        encoding="utf-8",
    )

    query_map = load_rewrite_map(path)

    assert query_map["口渴怎么办"] == [
        "口渴怎么办",
        "口渴提示什么问题",
        "口渴需要做什么检查",
    ]


def test_precomputed_rewriter_returns_cached_queries(tmp_path: Path) -> None:
    path = tmp_path / "rewrite_cache.jsonl"
    path.write_text(
        '{"question":"咳嗽","queries":["咳嗽原因","咳嗽用药","咳嗽何时就医"]}\n',
        encoding="utf-8",
    )
    rewriter = build_precomputed_query_rewriter(path)

    assert rewriter.rewrite("咳嗽") == ["咳嗽原因", "咳嗽用药", "咳嗽何时就医"]


def test_precomputed_rewriter_can_fallback_to_original(tmp_path: Path) -> None:
    path = tmp_path / "rewrite_cache.jsonl"
    path.write_text("", encoding="utf-8")
    rewriter = build_precomputed_query_rewriter(path, fallback_to_original=True)

    assert rewriter.rewrite("发热") == ["发热"]
