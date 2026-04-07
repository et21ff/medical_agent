from __future__ import annotations

import json

from medical_agent.session_memory import SessionMemoryStore


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.ttl: dict[str, int] = {}

    def get(self, key: str):
        return self.kv.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.kv[key] = value
        self.ttl[key] = ttl

    def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        if not values:
            return []
        size = len(values)
        norm_start = start if start >= 0 else max(0, size + start)
        if end < 0:
            norm_end = size + end
        else:
            norm_end = end
        norm_end = min(norm_end, size - 1)
        if norm_start > norm_end:
            return []
        return values[norm_start : norm_end + 1]

    def ltrim(self, key: str, start: int, end: int):
        values = self.lists.get(key, [])
        size = len(values)
        if not values:
            self.lists[key] = []
            return
        norm_start = start if start >= 0 else max(0, size + start)
        norm_end = end if end >= 0 else size + end
        norm_end = min(norm_end, size - 1)
        if norm_start > norm_end:
            self.lists[key] = []
            return
        self.lists[key] = values[norm_start : norm_end + 1]

    def expire(self, key: str, ttl: int):
        self.ttl[key] = ttl


class BrokenRedis(FakeRedis):
    def rpush(self, key: str, value: str):  # noqa: ARG002
        raise RuntimeError("boom")


def test_create_and_roundtrip_messages() -> None:
    store = SessionMemoryStore(client=FakeRedis(), enabled=True, ttl_s=100, max_history_turns=3)
    session_id = store.create_session("u1")
    assert session_id

    assert store.append_message("u1", session_id, "user", "q1")
    assert store.append_message("u1", session_id, "assistant", "a1")

    history = store.load_recent_messages("u1", session_id, max_turns=3)
    assert history == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]


def test_max_history_window_is_limited() -> None:
    store = SessionMemoryStore(client=FakeRedis(), enabled=True, ttl_s=100, max_history_turns=2)
    session_id = store.create_session("u1")
    for idx in range(4):
        assert store.append_message("u1", session_id, "user", f"q{idx}")
        assert store.append_message("u1", session_id, "assistant", f"a{idx}")

    # max_history_turns=2 -> keep latest 4 items (2 turns)
    history = store.load_recent_messages("u1", session_id)
    assert history == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_user_session_isolation() -> None:
    store = SessionMemoryStore(client=FakeRedis(), enabled=True)
    sid = "same-session"
    assert store.append_message("u1", sid, "user", "u1-q")
    assert store.append_message("u2", sid, "user", "u2-q")

    u1_history = store.load_recent_messages("u1", sid)
    u2_history = store.load_recent_messages("u2", sid)
    assert u1_history == [{"role": "user", "content": "u1-q"}]
    assert u2_history == [{"role": "user", "content": "u2-q"}]


def test_invalid_role_raises() -> None:
    store = SessionMemoryStore(client=FakeRedis(), enabled=True)
    sid = store.create_session("u1")
    try:
        store.append_message("u1", sid, "tool", "x")
    except ValueError as exc:
        assert "unsupported role" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_session_store_fails_open_on_redis_error() -> None:
    store = SessionMemoryStore(client=BrokenRedis(), enabled=True)
    sid = store.create_session("u1")
    assert sid
    assert store.append_message("u1", sid, "user", "q") is False
    assert store.load_recent_messages("u1", sid) == []


def test_create_session_writes_meta_payload() -> None:
    redis = FakeRedis()
    store = SessionMemoryStore(client=redis, enabled=True, ttl_s=123)
    sid = store.create_session("u1")
    meta_keys = [k for k in redis.kv if k.startswith("chat:session:meta:u1:")]
    assert meta_keys
    payload = json.loads(redis.kv[meta_keys[0]])
    assert payload["user_id"] == "u1"
    assert payload["session_id"] == sid
    assert redis.ttl[meta_keys[0]] == 123
