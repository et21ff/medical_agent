from __future__ import annotations

import os
from pathlib import Path

import pytest

from medical_agent.config import load_llm_config


def _load_dotenv_if_needed() -> None:
    """如果当前环境没注入配置，就从项目根目录 `.env` 里补一层。"""
    needed = {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"}
    if needed.issubset(os.environ.keys()):
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


def _has_langchain_llm_config() -> bool:
    _load_dotenv_if_needed()
    return bool(
        os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )


@pytest.mark.skipif(
    not _has_langchain_llm_config(),
    reason="Smoke test requires DeepSeek/OpenAI-compatible LLM config in environment or .env",
)
def test_langchain_deepseek_toolcall_smoke() -> None:
    """真实 smoke test：验证 DeepSeek 经由 LangChain 能否返回工具调用。"""
    langchain_openai = pytest.importorskip("langchain_openai")
    langchain_tools = pytest.importorskip("langchain_core.tools")

    @langchain_tools.tool
    def add_numbers(a: int, b: int) -> int:
        """返回两个整数之和。"""
        return a + b

    cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools([add_numbers])
    response = llm_with_tools.invoke(
        "你必须调用工具 add_numbers 来计算 23 + 19，先不要直接回答最终自然语言解释。"
    )

    print("model:", cfg.llm_model)
    print("base_url:", cfg.llm_base_url)
    print("response_type:", type(response))
    print("response_content:", response.content)
    print("tool_calls:", getattr(response, "tool_calls", None))

    assert hasattr(response, "tool_calls")
    assert response.tool_calls, "模型没有返回任何 tool call"
    assert response.tool_calls[0]["name"] == "add_numbers"

    args = response.tool_calls[0]["args"]
    assert args["a"] == 23
    assert args["b"] == 19


@pytest.mark.skipif(
    not _has_langchain_llm_config(),
    reason="Smoke test requires DeepSeek/OpenAI-compatible LLM config in environment or .env",
)
def test_langchain_deepseek_toolcall_roundtrip_smoke() -> None:
    """真实 smoke test：验证工具调用后，把结果回传给模型并得到最终回答。"""
    langchain_openai = pytest.importorskip("langchain_openai")
    langchain_tools = pytest.importorskip("langchain_core.tools")
    langchain_messages = pytest.importorskip("langchain_core.messages")

    @langchain_tools.tool
    def add_numbers(a: int, b: int) -> int:
        """返回两个整数之和。"""
        return a + b

    cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
        temperature=0.0,
    )

    llm_with_tools = llm.bind_tools([add_numbers])
    user_input = "你必须调用工具 add_numbers 来计算 23 + 19，然后再告诉我结果。"
    first_response = llm_with_tools.invoke(user_input)

    assert first_response.tool_calls, "第一轮没有返回 tool call"
    tool_call = first_response.tool_calls[0]
    args = tool_call["args"]
    tool_result = add_numbers.invoke(args)

    tool_message = langchain_messages.ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],
    )

    second_response = llm_with_tools.invoke(
        [
            ("human", user_input),
            first_response,
            tool_message,
        ]
    )

    print("first_response_content:", first_response.content)
    print("first_tool_calls:", first_response.tool_calls)
    print("tool_result:", tool_result)
    print("second_response_content:", second_response.content)

    assert tool_result == 42
    assert isinstance(second_response.content, str)
    assert second_response.content.strip()
