from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .config import load_api_config, load_config, load_llm_config
from .embedding_provider import build_embedding_provider
from .langchain_tools import format_retrieval_bundle
from .neo4j_retriever import build_neo4j_retriever
from .rag_cache import RAGCacheStore, build_redis_client, normalize_query
from .rerank_provider import build_rerank_provider
from .retrieval_pipeline import RetrievalBundle, RetrievalOptions, RetrievalPipeline
from .session_memory import SessionMemoryStore, build_session_redis_client
from .vector_retriever import build_vector_retriever

SYSTEM_PROMPT = (
    "你是一名严谨的医疗问答助手。"
    "如果检索证据中已经存在能够直接回答原始问题的明确结论，应直接围绕该结论作答，"
    "不要额外讨论未被询问的其他候选项，也不要补充“其他方面证据不足”之类的延伸说明。"
    "如果证据不足、证据偏题或无法直接支持结论，必须明确说明“证据不足”或“未检索到直接证据”。"
    "不要补充证据中没有直接支持的医学结论，不要把治疗表述升级为治愈。"
    "如果你判断为证据不足或未检索到直接证据，回答必须在该结论处结束，不得再补充常识建议、预防措施或治疗方案。"
)

PRECHECK_SYSTEM_PROMPT = (
    "你是医疗检索前置判断器。"
    "你的任务不是回答医学问题，而是判断该问题是否可以直接进入检索。"
    "你只能输出 JSON，不得输出额外解释。"
)

PRECHECK_USER_PROMPT = """请根据下面的问题和最近对话，返回一个 JSON 对象：

允许的 decision 只有：
- retrieve_directly
- rewrite_first
- needs_clarification

判定规则：
1. 如果问题已经足够清楚、适合直接做医疗证据检索，输出 retrieve_directly。
2. 如果问题很长、叙述化、包含多段病史或口语表达，需要先压缩成一个更适合检索的聚焦问题，输出 rewrite_first。
3. 如果问题缺少关键疾病/症状/检查对象，或指代不清，无法安全检索，输出 needs_clarification。
4. 绝对不要直接回答医学问题。
5. focused_question:
   - 当 decision=retrieve_directly 时必须为空字符串。
   - 当 decision=rewrite_first 时填写重写后的单个聚焦检索问题。
   - 当 decision=needs_clarification 时填写一句简短追问。

输出格式：
{{
  "decision": "retrieve_directly" | "rewrite_first" | "needs_clarification",
  "reason": "简短原因",
  "focused_question": "..."
}}

原始问题：
{question}

最近对话：
{history}
"""

EVIDENCE_JUDGE_SYSTEM_PROMPT = (
    "你是医疗证据裁决器。"
    "你的任务不是直接回答医学问题，而是判断当前检索证据是否足以支持回答。"
    "你只能输出 JSON，不得输出额外解释。"
)

EVIDENCE_JUDGE_USER_PROMPT = """请根据原始问题、最近对话和检索证据，返回一个 JSON 对象：

允许的 decision 只有：
- answer
- insufficient
- needs_clarification

判定规则：
1. 如果证据足以直接支持回答原问题，输出 answer。
2. 如果证据不足、偏题或无法直接支持原问题，输出 insufficient。
3. 如果问题对象仍然不明确，即使检索后仍无法对准证据，输出 needs_clarification。
4. 绝对不要直接回答问题。
5. focused_question:
   - 当 decision=answer 或 insufficient 时必须为空字符串。
   - 当 decision=needs_clarification 时填写一句简短追问。

输出格式：
{{
  "decision": "answer" | "insufficient" | "needs_clarification",
  "reason": "简短原因",
  "focused_question": "..."
}}

原始问题：
{question}

最近对话：
{history}

检索证据：
{evidence}
"""

DEFAULT_CLARIFICATION_QUESTION = "请明确你指的是哪种疾病、症状、检查结果或诊断。"
DEFAULT_INSUFFICIENT_ANSWER = "证据不足，未检索到可直接支持该问题的证据。"

BINARY_RISK_TOKENS = (
    "严重吗",
    "会好吗",
    "能治好吗",
    "要紧吗",
    "危险吗",
    "会传染吗",
    "能活多久",
    "正常吗",
    "高吗",
    "低吗",
)

MEDICAL_HINT_TOKENS = (
    "病",
    "症",
    "炎",
    "癌",
    "瘤",
    "综合征",
    "感染",
    "发热",
    "咳嗽",
    "头痛",
    "胸痛",
    "腹痛",
    "血压",
    "血糖",
    "心率",
    "检查",
    "化验",
)


class SupportsChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class ChatResult:
    user_id: str
    session_id: str
    history_turns_used: int
    answer: str
    evidence_preview: list[dict[str, Any]]
    query_variants: list[str]
    cache_hit: bool
    retrieve_ms: int
    llm_ms: int
    total_ms: int


@dataclass(frozen=True)
class PrecheckDecision:
    decision: Literal["retrieve_directly", "rewrite_first", "needs_clarification"]
    reason: str
    focused_question: str = ""


@dataclass(frozen=True)
class EvidenceDecision:
    decision: Literal["answer", "insufficient", "needs_clarification"]
    reason: str
    focused_question: str = ""


@dataclass(frozen=True)
class SessionHistoryResult:
    user_id: str
    session_id: str
    messages: list[dict[str, Any]]


class OpenAIChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


@dataclass
class MedicalQAService:
    pipeline: RetrievalPipeline
    llm_client: SupportsChatClient
    default_options: RetrievalOptions
    cache_store: RAGCacheStore | None = None
    session_store: SessionMemoryStore | None = None
    max_history_turns: int = 6
    evidence_preview_limit: int = 3

    def ask(
        self,
        user_id: str,
        question: str,
        *,
        session_id: str | None = None,
        options: RetrievalOptions | None = None,
    ) -> ChatResult:
        started_total = time.perf_counter()
        normalized_user_id = user_id.strip()
        normalized_question = question.strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if not normalized_question:
            raise ValueError("question must not be empty")

        active_session_id = session_id.strip() if session_id else ""
        if not active_session_id:
            if self.session_store is not None and self.session_store.enabled:
                active_session_id = self.session_store.create_session(normalized_user_id)
            else:
                active_session_id = "stateless"

        history_messages: list[dict[str, str]] = []
        if (
            self.session_store is not None
            and self.session_store.enabled
            and active_session_id != "stateless"
        ):
            history_messages = self.session_store.load_recent_messages(
                normalized_user_id, active_session_id, self.max_history_turns
            )
        history_turns_used = len(history_messages) // 2

        opts = options or self.default_options
        cache_hit = False
        precheck = self._rule_gate(normalized_question, history_messages)
        if precheck is None:
            if self._should_run_precheck_llm(normalized_question, history_messages):
                precheck = self._llm_precheck_decide(normalized_question, history_messages)
            else:
                precheck = PrecheckDecision(
                    decision="retrieve_directly",
                    reason="question is simple enough to retrieve directly",
                )

        if precheck.decision == "needs_clarification":
            answer = precheck.focused_question.strip() or DEFAULT_CLARIFICATION_QUESTION
            if (
                self.session_store is not None
                and self.session_store.enabled
                and active_session_id != "stateless"
            ):
                self.session_store.append_message(
                    normalized_user_id,
                    active_session_id,
                    "user",
                    normalized_question,
                )
                self.session_store.append_message(
                    normalized_user_id,
                    active_session_id,
                    "assistant",
                    answer,
                )
            total_ms = int((time.perf_counter() - started_total) * 1000)
            return ChatResult(
                user_id=normalized_user_id,
                session_id=active_session_id,
                history_turns_used=history_turns_used,
                answer=answer,
                evidence_preview=[],
                query_variants=[],
                cache_hit=False,
                retrieve_ms=0,
                llm_ms=0,
                total_ms=total_ms,
            )

        retrieval_question = (
            precheck.focused_question.strip()
            if precheck.decision == "rewrite_first" and precheck.focused_question.strip()
            else normalized_question
        )

        started_retrieve = time.perf_counter()

        bundle: RetrievalBundle | None = None
        cache_key = ""
        if self.cache_store is not None and self.cache_store.enabled:
            normalized_cache_query = normalize_query(retrieval_question)
            cache_key = self.cache_store.build_cache_key(normalized_cache_query, opts)
            bundle = self.cache_store.get(cache_key)
            cache_hit = bundle is not None

        if bundle is None:
            bundle = self.pipeline.retrieve(retrieval_question, options=opts)
            if self.cache_store is not None and self.cache_store.enabled and cache_key:
                self.cache_store.set(cache_key, bundle)

        retrieve_ms = int((time.perf_counter() - started_retrieve) * 1000)
        evidence_decision = self._judge_evidence_action(
            normalized_question,
            bundle,
            history_messages,
        )

        started_llm = time.perf_counter()
        if evidence_decision.decision == "insufficient":
            answer = DEFAULT_INSUFFICIENT_ANSWER
            llm_ms = 0
        elif evidence_decision.decision == "needs_clarification":
            answer = evidence_decision.focused_question.strip() or DEFAULT_CLARIFICATION_QUESTION
            llm_ms = 0
        else:
            answer = self._generate_final_answer(normalized_question, bundle, history_messages)
            llm_ms = int((time.perf_counter() - started_llm) * 1000)

        if (
            self.session_store is not None
            and self.session_store.enabled
            and active_session_id != "stateless"
        ):
            self.session_store.append_message(
                normalized_user_id,
                active_session_id,
                "user",
                normalized_question,
            )
            self.session_store.append_message(
                normalized_user_id,
                active_session_id,
                "assistant",
                answer,
            )

        total_ms = int((time.perf_counter() - started_total) * 1000)
        return ChatResult(
            user_id=normalized_user_id,
            session_id=active_session_id,
            history_turns_used=history_turns_used,
            answer=answer,
            evidence_preview=self._build_evidence_preview(bundle),
            query_variants=list(bundle.query_variants),
            cache_hit=cache_hit,
            retrieve_ms=retrieve_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )

    def _rule_gate(
        self,
        question: str,
        history_messages: list[dict[str, str]],
    ) -> PrecheckDecision | None:
        normalized_question = question.strip()
        if not normalized_question:
            return None
        if history_messages:
            return None

        if len(normalized_question) <= 5 and not self._contains_medical_hint(normalized_question):
            return PrecheckDecision(
                decision="needs_clarification",
                reason="question is too short and lacks a clear medical target",
                focused_question=DEFAULT_CLARIFICATION_QUESTION,
            )

        if len(normalized_question) <= 12 and self._contains_binary_risk_phrase(
            normalized_question
        ) and not self._contains_medical_hint(normalized_question):
            return PrecheckDecision(
                decision="needs_clarification",
                reason="binary medical question lacks an explicit disease or symptom target",
                focused_question=DEFAULT_CLARIFICATION_QUESTION,
            )

        return None

    def _should_run_precheck_llm(
        self,
        question: str,
        history_messages: list[dict[str, str]],
    ) -> bool:
        del history_messages
        normalized_question = question.strip()
        if len(normalized_question) >= 40:
            return True
        punctuation_hits = sum(normalized_question.count(ch) for ch in "，。；;,.、")
        if punctuation_hits >= 2:
            return True
        return False

    def _llm_precheck_decide(
        self,
        question: str,
        history_messages: list[dict[str, str]],
    ) -> PrecheckDecision:
        history_text = self._format_history_messages(history_messages)
        prompt = PRECHECK_USER_PROMPT.format(
            question=question,
            history=history_text,
        )
        raw = self.llm_client.complete(
            [
                {"role": "system", "content": PRECHECK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = self._extract_json_object(raw)
        if parsed is None:
            return PrecheckDecision(
                decision="retrieve_directly",
                reason="failed to parse precheck decision; falling back to direct retrieval",
            )
        decision = str(parsed.get("decision", "")).strip()
        reason = str(parsed.get("reason", "")).strip() or "llm precheck decision"
        focused_question = str(parsed.get("focused_question", "")).strip()
        if decision == "rewrite_first":
            if not focused_question:
                return PrecheckDecision(
                    decision="retrieve_directly",
                    reason="rewrite_first returned without focused question; falling back to direct retrieval",
                )
            return PrecheckDecision(decision=decision, reason=reason, focused_question=focused_question)
        if decision == "needs_clarification":
            return PrecheckDecision(
                decision=decision,
                reason=reason,
                focused_question=focused_question or DEFAULT_CLARIFICATION_QUESTION,
            )
        return PrecheckDecision(decision="retrieve_directly", reason=reason)

    def _judge_evidence_action(
        self,
        question: str,
        bundle: RetrievalBundle,
        history_messages: list[dict[str, str]],
    ) -> EvidenceDecision:
        if not bundle.evidence_items and not bundle.graph_evidence_texts and not bundle.text_results:
            return EvidenceDecision(
                decision="insufficient",
                reason="no evidence retrieved",
            )

        evidence_text = format_retrieval_bundle(bundle)
        history_text = self._format_history_messages(history_messages)
        prompt = EVIDENCE_JUDGE_USER_PROMPT.format(
            question=question,
            history=history_text,
            evidence=evidence_text,
        )
        raw = self.llm_client.complete(
            [
                {"role": "system", "content": EVIDENCE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = self._extract_json_object(raw)
        if parsed is None:
            return EvidenceDecision(
                decision="insufficient",
                reason="failed to parse evidence decision",
            )
        decision = str(parsed.get("decision", "")).strip()
        reason = str(parsed.get("reason", "")).strip() or "llm evidence decision"
        focused_question = str(parsed.get("focused_question", "")).strip()
        if decision == "answer":
            return EvidenceDecision(decision="answer", reason=reason)
        if decision == "needs_clarification":
            return EvidenceDecision(
                decision="needs_clarification",
                reason=reason,
                focused_question=focused_question or DEFAULT_CLARIFICATION_QUESTION,
            )
        return EvidenceDecision(decision="insufficient", reason=reason)

    def _generate_final_answer(
        self,
        question: str,
        bundle: RetrievalBundle,
        history_messages: list[dict[str, str]],
    ) -> str:
        evidence_text = format_retrieval_bundle(bundle)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        messages.extend(history_messages)
        messages.append(
            {
                "role": "user",
                "content": f"{evidence_text}\n\n原始问题：{question}\n\n请严格遵循“回答约束”并直接给出最终答案。",
            }
        )
        return self.llm_client.complete(messages)

    def _format_history_messages(self, history_messages: list[dict[str, str]]) -> str:
        if not history_messages:
            return "无最近对话。"
        rows: list[str] = []
        for idx, message in enumerate(history_messages[-6:], 1):
            role = str(message.get("role", "unknown")).strip() or "unknown"
            content = str(message.get("content", "")).strip()
            rows.append(f"{idx}. [{role}] {content}")
        return "\n".join(rows)

    def _contains_binary_risk_phrase(self, question: str) -> bool:
        return any(token in question for token in BINARY_RISK_TOKENS)

    def _contains_medical_hint(self, question: str) -> bool:
        return any(token in question for token in MEDICAL_HINT_TOKENS)

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        raw = text.strip()
        candidates = [raw]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
        if fenced:
            candidates.insert(0, fenced.group(1))
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(raw[start : end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                return None
        return None

    def get_session_messages(
        self,
        user_id: str,
        session_id: str,
        *,
        max_turns: int | None = None,
    ) -> SessionHistoryResult:
        normalized_user_id = user_id.strip()
        normalized_session_id = session_id.strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")

        if self.session_store is None or not self.session_store.enabled:
            return SessionHistoryResult(
                user_id=normalized_user_id,
                session_id=normalized_session_id,
                messages=[],
            )

        messages = self.session_store.load_recent_message_items(
            normalized_user_id,
            normalized_session_id,
            max_turns=max_turns,
        )
        return SessionHistoryResult(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
            messages=messages,
        )

    def _build_evidence_preview(self, bundle: RetrievalBundle) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for item in bundle.evidence_items[: self.evidence_preview_limit]:
            rows.append(
                {
                    "source": item.source_type,
                    "score": item.final_score,
                    "text": item.text,
                }
            )

        if rows:
            return rows

        for item in bundle.graph_evidence_texts[: self.evidence_preview_limit]:
            rows.append(
                {
                    "source": "graph",
                    "score": item.score,
                    "text": item.text,
                }
            )

        remaining = self.evidence_preview_limit - len(rows)
        if remaining > 0:
            for item in bundle.text_results[:remaining]:
                rows.append(
                    {
                        "source": "text",
                        "score": item.final_score,
                        "text": item.text,
                    }
                )
        return rows


def build_default_service() -> MedicalQAService:
    cfg = load_config()
    api_cfg = load_api_config()
    llm_cfg = load_llm_config()

    embedding_provider = build_embedding_provider(cfg)
    rerank_provider = build_rerank_provider(
        model_name=os.environ.get("RERANK_MODEL", "").strip() or "BAAI/bge-reranker-v2-m3"
    )

    pipeline = RetrievalPipeline(
        query_rewriter=None,
        neo4j_retriever=build_neo4j_retriever(
            embed_query=embedding_provider.embed_query,
            config=cfg,
        ),
        vector_retriever=build_vector_retriever(
            index_path=api_cfg.vector_index_path,
            meta_path=api_cfg.vector_meta_path,
            embedding_provider=embedding_provider,
            rerank_provider=None,
        ),
        evidence_rerank_provider=rerank_provider,
    )
    options = RetrievalOptions(
        use_rewrite=False,
        use_graph=True,
        use_text=True,
        graph_top_k=api_cfg.graph_top_k,
        text_top_k=api_cfg.text_top_k,
        text_recall_k=api_cfg.text_recall_k,
        evidence_top_k=api_cfg.evidence_top_k,
    )
    llm_client = OpenAIChatClient(
        base_url=llm_cfg.llm_base_url,
        api_key=llm_cfg.llm_api_key,
        model=llm_cfg.llm_model,
        timeout=llm_cfg.request_timeout,
    )
    cache_store: RAGCacheStore | None = None
    if api_cfg.cache_enabled:
        try:
            redis_client = build_redis_client(api_cfg.redis_url)
            cache_store = RAGCacheStore(
                client=redis_client,
                enabled=True,
                ttl_s=api_cfg.rag_cache_ttl_seconds,
                key_version=api_cfg.rag_cache_key_version,
                corpus_version=api_cfg.rag_corpus_version,
            )
        except Exception:
            # Fallback to non-cache mode when Redis init fails.
            cache_store = RAGCacheStore(client=None, enabled=False)

    session_store: SessionMemoryStore | None = None
    if api_cfg.session_enabled:
        try:
            session_redis_client = build_session_redis_client(api_cfg.session_redis_url)
            session_store = SessionMemoryStore(
                client=session_redis_client,
                enabled=True,
                ttl_s=api_cfg.session_ttl_seconds,
                max_history_turns=api_cfg.max_history_turns,
            )
        except Exception:
            session_store = SessionMemoryStore(client=None, enabled=False)

    return MedicalQAService(
        pipeline=pipeline,
        llm_client=llm_client,
        default_options=options,
        cache_store=cache_store,
        session_store=session_store,
        max_history_turns=api_cfg.max_history_turns,
        evidence_preview_limit=api_cfg.evidence_preview_limit,
    )
