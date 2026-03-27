"""Inspect the structure of an OpenAI-compatible chat completion response.

Usage:
    /root/pytorch-env/bin/python -m medical_agent.examples.inspect_response_structure
    /root/pytorch-env/bin/python -m medical_agent.examples.inspect_response_structure --prompt "Hello"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 兼容“直接执行脚本”场景：
# 当使用 `python /abs/path/to/script.py` 运行时，Python 只会把脚本所在目录
# 放进 sys.path，这样项目根目录不在导入路径里，`import medical_agent` 就会失败。
# 这里主动把仓库根目录加入 sys.path，保证两种运行方式都可用。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_agent.config import load_llm_config


def _safe_dump(obj: Any) -> str:
    """Best-effort dump for SDK objects."""
    if hasattr(obj, "model_dump"):
        try:
            return json.dumps(obj.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            pass
    try:
        return repr(obj)
    except Exception as exc:
        return f"<unprintable object: {exc}>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a real OpenAI-compatible response structure layer by layer."
    )
    parser.add_argument("--prompt", default="Hello", help="User prompt to send")
    parser.add_argument(
        "--system",
        default="You are a helpful assistant",
        help="System prompt to send",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "openai package is required. Install it in the active environment first."
        ) from exc

    cfg = load_llm_config()
    client = OpenAI(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        timeout=cfg.request_timeout,
    )

    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        stream=False,
    )

    print("=== type(response) ===")
    print(type(response))
    print()

    print("=== response top-level dump ===")
    print(_safe_dump(response))
    print()

    print("=== response.choices ===")
    print(response.choices)
    print()

    print("=== response.choices[0] ===")
    first_choice = response.choices[0]
    print(first_choice)
    print()

    print("=== response.choices[0].message ===")
    print(first_choice.message)
    print()

    print("=== response.choices[0].message.content ===")
    print(first_choice.message.content)


if __name__ == "__main__":
    main()
