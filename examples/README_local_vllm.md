# Connect `medical_agent` to local vLLM SFT backend

This repo's `medical_agent` already uses an OpenAI-compatible API client.
To switch from external provider to local SFT backend, do:

## 1) Start local vLLM server (with LoRA)

```bash
bash medical_agent/examples/start_local_vllm_sft.sh
```

Defaults used by the script:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: `/root/llm_learning/LLaMA-Factory/saves/qwen2_5-7b/lora/medical_rag_sft_good`
- Served model name: `qwen2.5-7b-medical-sft`
- Endpoint: `http://0.0.0.0:8000/v1`
- API key: `local-medical-agent`

## Troubleshooting: tokenizer crash on startup

If you see an error like:

`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`

it usually means your environment has `transformers>=5`, while your current
`vllm==0.11.0` runtime expects pre-v5 tokenizer APIs.

Fix in the same venv where `vllm` is installed:

```bash
/root/pytorch-env/bin/python -m pip install --upgrade "transformers<5" "tokenizers<0.23"
```

Then restart:

```bash
bash medical_agent/examples/start_local_vllm_sft.sh
```

## 2) Point `medical_agent` to local backend

Use either generic `LLM_*` variables or `VLLM_*` aliases.

```bash
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="local-medical-agent"
export VLLM_MODEL="qwen2.5-7b-medical-sft"
```

`medical_agent.config.load_llm_config()` now supports `VLLM_*` aliases and resolves keys by priority:

1. `LLM_*`
2. `VLLM_*`
3. `DEEPSEEK_*`

## 3) Smoke test

```bash
python -m medical_agent.examples.inspect_response_structure --prompt "你好，请简要介绍你的功能。"
```

If the output contains normal `choices[0].message.content`, `medical_agent` has switched to local backend successfully.

## 4) Start FastAPI service

After `.env` is configured (Neo4j/embedding/LLM), start the single-endpoint API:

```bash
cd /root/llm_learning
/root/pytorch-env/bin/python -m medical_agent.api_app
```

Default endpoint:

- `POST http://127.0.0.1:8080/chat`
- `GET  http://127.0.0.1:8080/healthz`
- `GET  http://127.0.0.1:8080/readyz`

Quick check:

```bash
curl -s http://127.0.0.1:8080/healthz
```

Chat example:

```bash
curl -s http://127.0.0.1:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "关心和理解艾滋病病毒感染者，最需掌握的生活技能是",
    "retrieval_options": {
      "graph_top_k": 3,
      "text_top_k": 5
    }
  }'
```

## 5) One-command API acceptance smoke

Use the bundled script to verify `/healthz` + `/chat` and basic response constraints:

```bash
bash medical_agent/examples/smoke_api_chat.sh
```

Custom endpoint or custom question:

```bash
BASE_URL=http://127.0.0.1:8080 \
QUESTION="如何预防结膜炎" \
bash medical_agent/examples/smoke_api_chat.sh
```

## 6) Enable Redis precise-key RAG cache (V1)

The `/chat` pipeline supports exact-key cache for retrieval results:

```bash
export CACHE_ENABLED=true
export CACHE_BACKEND=redis
export REDIS_URL=redis://127.0.0.1:6379/0
export RAG_CACHE_TTL_SECONDS=1800
export RAG_CACHE_KEY_VERSION=v1
export RAG_CORPUS_VERSION=exam_v1
```

If Redis is unavailable, the service will fall back to direct retrieval automatically.
