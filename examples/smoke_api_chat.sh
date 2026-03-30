#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
QUESTION="${QUESTION:-关心和理解艾滋病病毒感染者，最需掌握的生活技能是}"

echo "[smoke] base url: ${BASE_URL}"
echo "[smoke] check healthz..."
curl -fsS "${BASE_URL}/healthz" >/tmp/medical_agent_healthz.json
cat /tmp/medical_agent_healthz.json
echo

echo "[smoke] check chat..."
HTTP_CODE="$(curl -sS -o /tmp/medical_agent_chat.json -w "%{http_code}" \
  -X POST "${BASE_URL}/chat" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"${QUESTION}\"}")"

if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "[error] /chat returned HTTP ${HTTP_CODE}"
  cat /tmp/medical_agent_chat.json
  echo
  exit 1
fi

cat /tmp/medical_agent_chat.json
echo

echo "[smoke] validate response shape..."
/root/pytorch-env/bin/python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/medical_agent_chat.json").read_text(encoding="utf-8"))
required = ["answer", "evidence_preview", "query_variants", "request_id", "latency_ms"]
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(f"[error] missing keys: {missing}")

answer = str(data.get("answer", "")).strip()
if not answer:
    raise SystemExit("[error] answer is empty")

bad_markers = ["通常情况下", "可以采取以下措施", "建议如下：", "预防措施包括"]
if ("证据不足" in answer or "未检索到直接证据" in answer) and any(m in answer for m in bad_markers):
    raise SystemExit("[error] answer violates constraint: evidence-insufficient but still gives generic suggestions")

print("[ok] response structure and constraint checks passed")
PY

echo "[done] smoke passed"
