from medical_agent.query_rewriter import _extract_response_text


def test_extract_response_text_accepts_raw_string() -> None:
    raw = '{"queries":["q1","q2","q3"]}'
    assert _extract_response_text(raw) == raw
