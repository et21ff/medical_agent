from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SupportsRedisSessionClient(Protocol):
    def get(self, key: str) -> Any:
        ...

    def setex(self, key: str, ttl: int, value: str) -> Any:
        ...

    def rpush(self, key: str, value: str) -> Any:
        ...

    def lrange(self, key: str, start: int, end: int) -> list[Any]:
        ...

    def ltrim(self, key: str, start: int, end: int) -> Any:
        ...

    def expire(self, key: str, ttl: int) -> Any:
        ...


def _session_meta_key(user_id: str, session_id: str) -> str:
    return f"chat:session:meta:{user_id}:{session_id}"


def _session_msgs_key(user_id: str, session_id: str) -> str:
    return f"chat:session:msgs:{user_id}:{session_id}"


@dataclass
class SessionMemoryStore:
    client: SupportsRedisSessionClient | None
    enabled: bool = True
    ttl_s: int = 7 * 24 * 3600
    max_history_turns: int = 6

    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        if not self.enabled or self.client is None:
            return session_id
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": int(time.time()),
            }
            self.client.setex(
                _session_meta_key(user_id, session_id),
                int(self.ttl_s),
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("session create failed: %s", exc)
        return session_id

    def append_message(self, user_id: str, session_id: str, role: str, content: str) -> bool:
        if not self.enabled or self.client is None:
            return False
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported role: {role}")
        try:
            item = {
                "role": role,
                "content": content,
                "ts": int(time.time()),
            }
            msgs_key = _session_msgs_key(user_id, session_id)
            self.client.rpush(msgs_key, json.dumps(item, ensure_ascii=False))
            # Keep only recent messages for short-term memory.
            keep_items = max(1, int(self.max_history_turns) * 2)
            self.client.ltrim(msgs_key, -keep_items, -1)
            self.client.expire(msgs_key, int(self.ttl_s))
            self.client.expire(_session_meta_key(user_id, session_id), int(self.ttl_s))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("session append failed: %s", exc)
            return False

    def load_recent_messages(
        self, user_id: str, session_id: str, max_turns: int | None = None
    ) -> list[dict[str, str]]:
        if not self.enabled or self.client is None:
            return []
        turns = max_turns if max_turns is not None else self.max_history_turns
        keep_items = max(1, int(turns) * 2)
        try:
            raw_items = self.client.lrange(_session_msgs_key(user_id, session_id), -keep_items, -1)
            messages: list[dict[str, str]] = []
            for raw in raw_items:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                item = json.loads(raw)
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", ""))
                if role and content:
                    messages.append({"role": role, "content": content})
            return messages
        except Exception as exc:  # noqa: BLE001
            logger.warning("session load failed: %s", exc)
            return []


def build_session_redis_client(redis_url: str) -> SupportsRedisSessionClient:
    try:
        import redis
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "redis package is required when SESSION_ENABLED is true. Install redis first."
        ) from exc
    return redis.Redis.from_url(redis_url)

