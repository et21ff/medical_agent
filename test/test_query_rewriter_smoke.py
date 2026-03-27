import os

import pytest

from medical_agent.query_rewriter import rewrite_queries


def _has_llm_credentials() -> bool:
    return bool(
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    )


@pytest.mark.skipif(
    not _has_llm_credentials(),
    reason="Smoke test requires LLM_API_KEY or DEEPSEEK_API_KEY in environment",
)
def test_rewrite_queries_smoke() -> None:
    question = "怀孕8个月了，之前体检一直都正常，孩子长的也都刚好不大不小的，但是这一次我去孕检了，医生竟然说孩子偏小三周，我心里面就挺担心的，我平常吃的也挺多的，还挺胖的。孕晚期胎儿偏小三周怎么办？"

    queries = rewrite_queries(question)

    assert isinstance(queries, list)
    assert len(queries) in (1, 3)
    assert all(isinstance(item, str) and item.strip() for item in queries)
    if len(queries) == 3:
        assert len(set(queries)) == 3
        for query in queries:
            print(query)
