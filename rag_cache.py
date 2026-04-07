from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .graph_text_formatter import GraphEvidenceText
from .retrieval_pipeline import EvidenceItem, RetrievalBundle, RetrievalOptions
from .vector_retriever import RetrievedText

logger = logging.getLogger(__name__)


class SupportsRedisClient(Protocol):
    def get(self, key: str) -> Any:
        ...

    def setex(self, key: str, ttl: int, value: str) -> Any:
        ...


def normalize_query(query: str) -> str:
    text = query.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_cache_key(
    normalized_query: str,
    options: RetrievalOptions,
    *,
    corpus_version: str,
    key_version: str = "v1",
) -> str:
    payload = {
        "query": normalized_query,
        "options": asdict(options),
        "corpus_version": corpus_version,
        "key_version": key_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"rag:{key_version}:{digest}"


@dataclass
class RAGCacheStore:
    client: SupportsRedisClient | None
    enabled: bool = True
    ttl_s: int = 1800
    key_version: str = "v1"
    corpus_version: str = "exam_v1"

    def build_cache_key(self, normalized_query: str, options: RetrievalOptions) -> str:
        return build_cache_key(
            normalized_query,
            options,
            corpus_version=self.corpus_version,
            key_version=self.key_version,
        )

    def get(self, key: str) -> RetrievalBundle | None:
        if not self.enabled or self.client is None:
            return None
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
            return self._deserialize_bundle(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag cache get failed: %s", exc)
            return None

    def set(self, key: str, bundle: RetrievalBundle, ttl_s: int | None = None) -> bool:
        if not self.enabled or self.client is None:
            return False
        effective_ttl = ttl_s if ttl_s is not None else self.ttl_s
        try:
            payload = self._serialize_bundle(bundle)
            self.client.setex(key, int(effective_ttl), json.dumps(payload, ensure_ascii=False))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag cache set failed: %s", exc)
            return False

    @staticmethod
    def _serialize_bundle(bundle: RetrievalBundle) -> dict[str, Any]:
        return {
            "original_question": bundle.original_question,
            "query_variants": list(bundle.query_variants),
            "graph_evidence_texts": [asdict(item) for item in bundle.graph_evidence_texts],
            "text_results": [asdict(item) for item in bundle.text_results],
            "evidence_items": [asdict(item) for item in bundle.evidence_items],
        }

    @staticmethod
    def _deserialize_bundle(payload: dict[str, Any]) -> RetrievalBundle:
        graph_evidence = [
            GraphEvidenceText(**item) for item in payload.get("graph_evidence_texts", [])
        ]
        text_results = [RetrievedText(**item) for item in payload.get("text_results", [])]
        evidence_items = [EvidenceItem(**item) for item in payload.get("evidence_items", [])]
        return RetrievalBundle(
            original_question=str(payload.get("original_question", "")),
            query_variants=list(payload.get("query_variants", [])),
            graph_evidence_texts=graph_evidence,
            text_results=text_results,
            evidence_items=evidence_items,
        )


def build_redis_client(redis_url: str) -> SupportsRedisClient:
    try:
        import redis
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "redis package is required when CACHE_ENABLED is true. Install redis first."
        ) from exc
    return redis.Redis.from_url(redis_url)

