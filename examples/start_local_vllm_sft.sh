#!/usr/bin/env bash
set -euo pipefail

# Start a local OpenAI-compatible vLLM server for medical_agent.
#
# Example:
#   bash medical_agent/examples/start_local_vllm_sft.sh
#
# You can override any variable inline:
#   PORT=8001 API_KEY=local-key bash medical_agent/examples/start_local_vllm_sft.sh

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-/root/llm_learning/LLaMA-Factory/saves/qwen2_5-7b/lora/medical_rag_sft_attn_r16_e5_lowmem}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-7b-medical-sft}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-local-medical-agent}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM_BIN="${VLLM_BIN:-}"
PY_BIN="${PY_BIN:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ENABLE_LORA="${ENABLE_LORA:-0}"

if [[ -z "${VLLM_BIN}" ]]; then
  if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
  elif [[ -x "/root/pytorch-env/bin/vllm" ]]; then
    VLLM_BIN="/root/pytorch-env/bin/vllm"
  else
    echo "[error] vllm executable not found. Activate your venv or set VLLM_BIN=/abs/path/to/vllm." >&2
    exit 1
  fi
fi

if [[ -z "${PY_BIN}" ]]; then
  CANDIDATE_PY="$(dirname "${VLLM_BIN}")/python"
  if [[ -x "${CANDIDATE_PY}" ]]; then
    PY_BIN="${CANDIDATE_PY}"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  else
    echo "[error] python executable not found. Set PY_BIN=/abs/path/to/python." >&2
    exit 1
  fi
fi

if ! "${PY_BIN}" - <<'PY'
import sys

try:
    import transformers
except Exception as exc:
    print(f"[error] cannot import transformers: {exc}", file=sys.stderr)
    raise SystemExit(1)

version_text = getattr(transformers, "__version__", "0.0.0")
major = int((version_text.split(".")[0] or "0"))
if major >= 5:
    print(
        "[error] transformers>=5 detected "
        f"({version_text}). vLLM 0.11.0 may fail with tokenizer API removals.\n"
        "[fix] run: /root/pytorch-env/bin/python -m pip install --upgrade "
        "'transformers<5' 'tokenizers<0.23'",
        file=sys.stderr,
    )
    raise SystemExit(2)

print(f"[check] transformers={version_text} OK")
PY
then
  exit 1
fi

echo "[vllm] base model: ${BASE_MODEL}"
echo "[vllm] lora path: ${LORA_PATH}"
echo "[vllm] served model: ${SERVED_MODEL_NAME}"
echo "[vllm] listening: http://${HOST}:${PORT}/v1"
echo "[vllm] binary: ${VLLM_BIN}"
echo "[vllm] python: ${PY_BIN}"
echo "[vllm] tool parser: ${TOOL_CALL_PARSER}"
echo "[vllm] enable lora: ${ENABLE_LORA}"

EXTRA_TOOL_FLAGS=()
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  EXTRA_TOOL_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

EXTRA_LORA_FLAGS=()
if [[ "${ENABLE_LORA}" == "1" ]]; then
  EXTRA_LORA_FLAGS+=(--enable-lora --lora-modules "medical_sft=${LORA_PATH}")
fi

exec "${VLLM_BIN}" serve "${BASE_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  "${EXTRA_LORA_FLAGS[@]}" \
  "${EXTRA_TOOL_FLAGS[@]}"
