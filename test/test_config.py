import os
import unittest
from unittest.mock import patch

from medical_agent.config import ConfigError, load_api_config, load_config, load_llm_config


class LoadLLMConfigTests(unittest.TestCase):
    def test_load_llm_config_uses_deepseek_defaults(self) -> None:
        env = {"DEEPSEEK_API_KEY": "sk-test"}

        cfg = load_llm_config(env)

        self.assertEqual(cfg.llm_api_key, "sk-test")
        self.assertEqual(cfg.llm_base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.llm_model, "deepseek-chat")
        self.assertEqual(cfg.request_timeout, 30.0)

    def test_load_llm_config_prefers_generic_keys(self) -> None:
        env = {
            "LLM_API_KEY": "sk-generic",
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "LLM_BASE_URL": "https://generic.example.com/v1",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "generic-model",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "LLM_REQUEST_TIMEOUT": "45",
        }

        cfg = load_llm_config(env)

        self.assertEqual(cfg.llm_api_key, "sk-generic")
        self.assertEqual(cfg.llm_base_url, "https://generic.example.com/v1")
        self.assertEqual(cfg.llm_model, "generic-model")
        self.assertEqual(cfg.request_timeout, 45.0)

    def test_load_llm_config_supports_vllm_aliases(self) -> None:
        env = {
            "VLLM_API_KEY": "local-key",
            "VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
            "VLLM_MODEL": "qwen2.5-7b-medical-sft",
            "DEEPSEEK_API_KEY": "sk-deepseek",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "deepseek-chat",
        }

        cfg = load_llm_config(env)

        self.assertEqual(cfg.llm_api_key, "local-key")
        self.assertEqual(cfg.llm_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(cfg.llm_model, "qwen2.5-7b-medical-sft")
        self.assertEqual(cfg.request_timeout, 30.0)

    def test_load_llm_config_requires_api_key(self) -> None:
        with self.assertRaises(ConfigError):
            load_llm_config({})

    def test_load_llm_config_rejects_invalid_timeout(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "sk-test",
            "LLM_REQUEST_TIMEOUT": "abc",
        }

        with self.assertRaises(ConfigError):
            load_llm_config(env)


class LoadConfigTests(unittest.TestCase):
    def test_load_config_reads_complete_agent_config(self) -> None:
        env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "EMBED_MODEL": "shibing624/text2vec-base-chinese",
            "DEEPSEEK_API_KEY": "sk-test",
        }

        cfg = load_config(env)

        self.assertEqual(cfg.neo4j_uri, "bolt://localhost:7687")
        self.assertEqual(cfg.neo4j_user, "neo4j")
        self.assertEqual(cfg.neo4j_password, "password")
        self.assertEqual(cfg.embed_model, "shibing624/text2vec-base-chinese")
        self.assertEqual(cfg.llm_api_key, "sk-test")
        self.assertEqual(cfg.llm_base_url, "https://api.deepseek.com")
        self.assertEqual(cfg.llm_model, "deepseek-chat")

    def test_load_config_requires_neo4j_and_embedding_fields(self) -> None:
        env = {"DEEPSEEK_API_KEY": "sk-test"}

        with self.assertRaises(ConfigError):
            load_config(env)

    def test_load_config_uses_os_environ_by_default(self) -> None:
        fake_env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "EMBED_MODEL": "text2vec",
            "LLM_API_KEY": "sk-live",
            "LLM_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-chat",
            "LLM_REQUEST_TIMEOUT": "20",
        }

        with patch.dict(os.environ, fake_env, clear=True):
            cfg = load_config()

        self.assertEqual(cfg.neo4j_uri, "bolt://localhost:7687")
        self.assertEqual(cfg.llm_api_key, "sk-live")
        self.assertEqual(cfg.request_timeout, 20.0)


class LoadAPIConfigTests(unittest.TestCase):
    def test_load_api_config_uses_defaults(self) -> None:
        cfg = load_api_config({})
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertEqual(cfg.port, 8080)
        self.assertFalse(cfg.debug)
        self.assertEqual(cfg.evidence_preview_limit, 3)
        self.assertEqual(
            cfg.vector_index_path,
            "/root/llm_learning/data/EXAM/exam_rag_faiss.index",
        )
        self.assertEqual(
            cfg.vector_meta_path,
            "/root/llm_learning/data/EXAM/exam_rag_meta.jsonl",
        )
        self.assertEqual(cfg.graph_top_k, 3)
        self.assertEqual(cfg.text_top_k, 5)
        self.assertEqual(cfg.text_recall_k, 20)
        self.assertEqual(cfg.evidence_top_k, 5)

    def test_load_api_config_reads_overrides(self) -> None:
        env = {
            "API_HOST": "127.0.0.1",
            "API_PORT": "9000",
            "API_DEBUG": "true",
            "EVIDENCE_PREVIEW_LIMIT": "6",
            "VECTOR_INDEX_PATH": "/tmp/a.index",
            "VECTOR_META_PATH": "/tmp/a.jsonl",
            "GRAPH_TOP_K": "4",
            "TEXT_TOP_K": "8",
            "TEXT_RECALL_K": "30",
            "EVIDENCE_TOP_K": "7",
        }
        cfg = load_api_config(env)
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 9000)
        self.assertTrue(cfg.debug)
        self.assertEqual(cfg.evidence_preview_limit, 6)
        self.assertEqual(cfg.vector_index_path, "/tmp/a.index")
        self.assertEqual(cfg.vector_meta_path, "/tmp/a.jsonl")
        self.assertEqual(cfg.graph_top_k, 4)
        self.assertEqual(cfg.text_top_k, 8)
        self.assertEqual(cfg.text_recall_k, 30)
        self.assertEqual(cfg.evidence_top_k, 7)

    def test_load_api_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ConfigError):
            load_api_config({"API_PORT": "abc"})
        with self.assertRaises(ConfigError):
            load_api_config({"API_DEBUG": "maybe"})
        with self.assertRaises(ConfigError):
            load_api_config({"GRAPH_TOP_K": "0"})


if __name__ == "__main__":
    unittest.main()
