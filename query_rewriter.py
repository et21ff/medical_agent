"""问题改写模块。

这个模块位于整个检索链路的最前面，职责很单一：
把 1 个用户原始问题改写成 3 个互补的检索子问题，
供后续 Neo4j 检索模块分别召回，再做结果融合。

设计原则：
1. 只做“改写”，不负责检索、融合或最终回答。
2. 优先要求模型返回结构化 JSON，方便程序稳定解析。
3. 当模型输出异常、接口不兼容、或依赖缺失时，必须能优雅降级。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Sequence

from .config import AgentConfig, LLMConfig, load_llm_config

logger = logging.getLogger(__name__)


# 系统提示词负责约束模型的“角色”和“输出格式”。
# 这里强制要求返回 {"queries": [...]}，是为了尽量减少后续解析的不确定性。
REWRITE_SYSTEM_PROMPT = """
你是医疗问答检索改写器。你的任务是将一个用户问题改写为 3 个互补的中文检索子问题。

要求：
1. 只输出 JSON 对象，格式严格为：{"queries": ["...", "...", "..."]}。
2. 必须恰好 3 条，且每条都是完整问句。
3. 三条问题要互补：可分别偏向 症状/诊断、治疗/用药、鉴别/风险与就医建议。
4. 不要引入原问题中未出现的具体药名、检查结果或诊断结论。
5. 不要输出除 JSON 之外的任何文字。
""".strip()


@dataclass
class QueryRewriter:
    """将用户问题改写为多个检索子问题的核心执行器。

    字段说明：
    - client: 兼容 OpenAI SDK 调用方式的客户端实例
    - model: 改写阶段使用的模型名
    - max_queries: 期望输出的子问题数量，当前固定为 3
    """

    client: Any
    model: str
    max_queries: int = 3

    # 练手建议 1：
    # 可以先从这个主流程入口开始重写，理解“输入 -> 调模型 -> 解析 -> 校验 -> 降级”的完整闭环。
    # 如果你想降低难度，也可以先保留这个函数，只重写下面两个辅助函数。
    def rewrite(self, question: str) -> list[str]:
        """对外主入口。

        执行流程：
        1. 校验输入问题
        2. 组装用户提示词
        3. 调用 LLM 完成改写
        4. 解析并校验输出
        5. 任何异常都降级返回 [原问题]

        之所以采用“异常即降级”，是为了保证后续检索链路不中断。
        就算模型临时不可用，系统仍可以退回单路检索。
        """
        if (
            not question or not question.strip()
        ):  # 判断字符串 与去除 空格后字符串是否为空
            raise ValueError("question must not be empty")
        user_prompt = (
            "请将下面这个医疗问题改写成3个互补检索子问题 \n\n"
            f"原问题{question.strip()}\n"
        )
        try:
            response = self._call_rewrite_api(user_prompt=user_prompt)
            raw = _extract_response_text(response) or ""
            return self._validate_queries(raw=raw)
        except Exception as exc:
            logger.warning("问题改写失败，降级返回原问题：%s", exc)

            return [question.strip()]

    # 练手建议 4：
    # 这一段更偏工程接口适配，建议放到最后再自己重写。
    # 学习重点是：为什么要先尝试 response_format，再做普通文本模式降级。
    def _call_rewrite_api(self, user_prompt: str) -> Any:
        """调用模型接口。

        优先尝试结构化 JSON 模式，因为它更利于程序解析。
        如果当前 provider / model 不支持 `response_format`，
        则自动退化为普通文本模式再试一次。
        """
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # 优先使用结构化 JSON 输出；如果平台不支持，再退回普通文本。
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except Exception as structured_exc:
            logger.info(
                "Structured response_format failed, retrying plain text mode: %s",
                structured_exc,
            )
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
            )

    def _validate_queries(self, raw: str) -> list[str]:
        """校验模型输出是否满足“3 条有效改写问题”的约束。

        校验内容包括：
        - 顶层必须是 JSON 对象
        - `queries` 必须是列表
        - 每项必须是非空字符串
        - 自动去重
        - 最终数量必须等于 `max_queries`
        - 不能 3 条都和原问题完全一样
        """
        parsed = _parse_json_text(raw)
        queries = parsed.get("queries")
        if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)):
            raise ValueError("无效输出值，‘queries’一定得是个list")
        cleaned: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if not isinstance(query, str):
                continue
            text = query.strip()
            if not text or text in seen:
                continue
            cleaned.append(text)
            seen.add(text)

        if len(cleaned) != self.max_queries:
            raise ValueError(
                f"invalid rewrite output: expected {self.max_queries} queries, got {len(cleaned)}"
            )
        return cleaned


# 练手建议 3：
# 这个函数适合在 _validate_queries 熟悉后再重写。
# 学习重点是：真实模型输出为什么需要做 markdown 代码块剥离和 JSON 容错。
def _parse_json_text(raw: str) -> dict[str, Any]:
    """把模型返回内容解析为 JSON 对象。

    容错策略：
    1. 去掉 markdown 代码块包裹（```json ... ```）
    2. 优先直接 `json.loads`
    3. 如果返回前后混入了解释文本，则尝试截取最外层 `{...}` 再解析

    这样可以兼容不同 provider 的轻微输出偏差。
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if not text:
        raise ValueError("empty rewrite response")
    try:
        obj = json.loads(text)
    except JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("rewrite response must be a JSON object")
    return obj


def _extract_response_text(response: Any) -> str:
    """兼容不同 provider 的响应形态，抽取文本内容。"""
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    raise TypeError(f"unsupported rewrite response type: {type(response)!r}")


# 这部分主要是“构造组件”和“注入配置”，更偏装配逻辑。
# 不建议最开始就改这里，除非你已经比较熟悉前面的主流程和解析逻辑。
def build_query_rewriter(
    config: AgentConfig | LLMConfig | None = None,
) -> QueryRewriter:
    """根据配置构造 `QueryRewriter` 实例。

    支持三种输入：
    - `None`：直接从环境变量加载 LLM 配置
    - `AgentConfig`：从完整 agent 配置里抽取 LLM 部分
    - `LLMConfig`：直接使用传入的 LLM 配置

    这里延迟导入 `openai`，是为了避免：
    即便环境里没安装 OpenAI SDK，整个包也无法被导入。
    """
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openai package is required for query rewriting. Install it first."
        ) from exc

    if config is None:
        llm_cfg = load_llm_config()
    elif isinstance(config, AgentConfig):
        llm_cfg = LLMConfig(
            llm_base_url=config.llm_base_url,
            llm_api_key=config.llm_api_key,
            llm_model=config.llm_model,
            request_timeout=config.request_timeout,
        )
    else:
        llm_cfg = config

    client = OpenAI(
        api_key=llm_cfg.llm_api_key,
        base_url=llm_cfg.llm_base_url,
        timeout=llm_cfg.request_timeout,
    )
    return QueryRewriter(client=client, model=llm_cfg.llm_model)


# 这是对外暴露的简化调用入口，适合先会用，再回头理解内部实现。
def rewrite_queries(
    question: str, config: AgentConfig | LLMConfig | None = None
) -> list[str]:
    """函数式封装，便于外部模块直接调用。

    用法上它比手动创建 `QueryRewriter` 更简单：
    - 自动构造 client
    - 自动读取配置
    - 自动执行改写

    返回约定：
    - 成功时：3 条互补子问题
    - 失败时：包含原问题的单元素列表
    """
    rewriter = build_query_rewriter(config=config)
    return rewriter.rewrite(question)
