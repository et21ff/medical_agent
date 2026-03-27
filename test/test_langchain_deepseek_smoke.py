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
def test_langchain_deepseek_invoke_smoke() -> None:
    """真实 smoke test：验证 LangChain 能否通过 DeepSeek 完成一次基本调用。"""
    langchain_openai = pytest.importorskip("langchain_openai")

    cfg = load_llm_config()
    llm = langchain_openai.ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
        temperature=0.2,
    )

    response = llm.invoke("请只回复四个字：联调成功")

    print("model:", cfg.llm_model)
    print("base_url:", cfg.llm_base_url)
    print("response_type:", type(response))
    print("response_content:", response.content)

    assert isinstance(response.content, str)
    assert response.content.strip()
