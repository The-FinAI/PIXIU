"""Tests for MiniMax LLM provider integration in PIXIU."""
import os
import sys
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from types import ModuleType

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock lm_eval and its submodules (project-specific submodule dependency)
_lm_eval_mock = MagicMock()
_lm_eval_mock.base.BaseLM = type("BaseLM", (), {
    "__init__": lambda self: None,
    "cache_hook": MagicMock(),
})
sys.modules["lm_eval"] = _lm_eval_mock
sys.modules["lm_eval.base"] = _lm_eval_mock.base
sys.modules["lm_eval.utils"] = _lm_eval_mock.utils
sys.modules["lm_eval.metrics"] = _lm_eval_mock.metrics
sys.modules["lm_eval.models"] = _lm_eval_mock.models
sys.modules["lm_eval.tasks"] = _lm_eval_mock.tasks


class TestMiniMaxLMConfig(unittest.TestCase):
    """Test MiniMaxLM class configuration and attributes."""

    def test_minimax_models_dict(self):
        from minimax_lm import MINIMAX_MODELS
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MINIMAX_MODELS)

    def test_minimax_context_windows(self):
        from minimax_lm import MINIMAX_MODELS
        for model, ctx_len in MINIMAX_MODELS.items():
            self.assertEqual(ctx_len, 204800, f"{model} should have 204K context")

    def test_minimax_api_base_url(self):
        from minimax_lm import MiniMaxLM
        self.assertEqual(
            MiniMaxLM.API_BASE_URL,
            "https://api.minimax.io/v1/chat/completions",
        )

    def test_minimax_api_key_env(self):
        from minimax_lm import MiniMaxLM
        self.assertEqual(MiniMaxLM.API_KEY_ENV, "MINIMAX_API_KEY")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_lm_init(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertEqual(lm.model, "MiniMax-M2.7")
        self.assertIn("Bearer test-key-123", lm.headers["Authorization"])

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_max_length(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertEqual(lm.max_length, 204800)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_max_length_highspeed(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7-highspeed")
        self.assertEqual(lm.max_length, 204800)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_temperature_clamping_zero(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertGreater(lm._get_temperature(0.0), 0.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_temperature_clamping_negative(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertGreater(lm._get_temperature(-0.5), 0.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_temperature_clamping_high(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertLessEqual(lm._get_temperature(1.5), 1.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_temperature_valid(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertEqual(lm._get_temperature(0.5), 0.5)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_temperature_boundary(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        lm = MiniMaxLM("MiniMax-M2.7")
        self.assertEqual(lm._get_temperature(1.0), 1.0)

    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_missing_api_key_raises(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(KeyError):
                MiniMaxLM("MiniMax-M2.7")


class TestChatLMConfig(unittest.TestCase):
    """Test ChatLM base class configuration."""

    def test_chatl_api_base_url(self):
        from chatlm import ChatLM
        self.assertEqual(
            ChatLM.API_BASE_URL,
            "https://api.openai.com/v1/chat/completions",
        )

    def test_chatl_api_key_env(self):
        from chatlm import ChatLM
        self.assertEqual(ChatLM.API_KEY_ENV, "OPENAI_API_SECRET_KEY")

    @patch.dict(os.environ, {"OPENAI_API_SECRET_KEY": "sk-test"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_chatl_default_temperature(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from chatlm import ChatLM
        lm = ChatLM("gpt-4")
        self.assertEqual(lm._get_temperature(0.0), 0.0)
        self.assertEqual(lm._get_temperature(0.5), 0.5)

    @patch.dict(os.environ, {"OPENAI_API_SECRET_KEY": "sk-test"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_chatl_inherits_base_lm(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from chatlm import ChatLM
        lm = ChatLM("gpt-4")
        self.assertEqual(lm.model, "gpt-4")

    def test_minimax_is_subclass_of_chatlm(self):
        from chatlm import ChatLM
        from minimax_lm import MiniMaxLM
        self.assertTrue(issubclass(MiniMaxLM, ChatLM))


class TestEvaluatorRouting(unittest.TestCase):
    """Test model routing in evaluator.py."""

    def test_minimax_models_in_evaluator(self):
        from minimax_lm import MINIMAX_MODELS
        self.assertIn("MiniMax-M2.7", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.7-highspeed", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MINIMAX_MODELS)

    def test_evaluator_has_minimax_models(self):
        from minimax_lm import MINIMAX_MODELS
        self.assertTrue(len(MINIMAX_MODELS) >= 4)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_model_creates_minimax_lm(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from minimax_lm import MiniMaxLM, MINIMAX_MODELS
        for model_name in MINIMAX_MODELS:
            lm = MiniMaxLM(model_name)
            self.assertIsInstance(lm, MiniMaxLM)
            self.assertEqual(lm.model, model_name)

    @patch.dict(os.environ, {"OPENAI_API_SECRET_KEY": "sk-test"})
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_gpt_model_creates_chatlm(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock()
        from chatlm import ChatLM
        from minimax_lm import MiniMaxLM
        lm = ChatLM("gpt-4")
        self.assertIsInstance(lm, ChatLM)
        self.assertNotIsInstance(lm, MiniMaxLM)


class TestMiniMaxFActScore(unittest.TestCase):
    """Test MiniMax integration in FActScore package."""

    def test_minimax_lm_import(self):
        try:
            from factscore_package.minimax_lm import MiniMaxModel
        except ImportError:
            self.fail("Should be able to import MiniMaxModel from factscore_package")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-factscore"})
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_model_init(self, mock_cache):
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel("MiniMax-M2.7", cache_file="/tmp/test_cache")
        self.assertEqual(model.model_name, "MiniMax-M2.7")
        self.assertEqual(model.temp, 0.7)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-factscore"})
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_model_default_model(self, mock_cache):
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel(cache_file="/tmp/test_cache")
        self.assertEqual(model.model_name, "MiniMax-M2.7")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-api-key"})
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_model_api_base(self, mock_cache):
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel("MiniMax-M2.7", cache_file="/tmp/test_cache")
        self.assertEqual(model.client.base_url.host, "api.minimax.io")

    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_openai_model_accepts_api_base(self, mock_cache):
        from factscore_package.openai_lm import OpenAIModel
        model = OpenAIModel(
            "test-model",
            cache_file="/tmp/test_cache",
            key="test-key",
            api_base="https://custom.api.com/v1",
        )
        self.assertEqual(model.client.base_url.host, "custom.api.com")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"})
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_model_inherits_openai(self, mock_cache):
        from factscore_package.openai_lm import OpenAIModel
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel(cache_file="/tmp/test_cache")
        self.assertIsInstance(model, OpenAIModel)


class TestOaCompletion(unittest.TestCase):
    """Test the oa_completion async function."""

    @patch("httpx.AsyncClient")
    def test_oa_completion_uses_provided_url(self, mock_client_cls):
        from chatlm import oa_completion
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        result = asyncio.run(oa_completion(
            url="https://api.minimax.io/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.5,
        ))
        self.assertEqual(result, ["test response"])
        call_args = mock_client.post.call_args
        self.assertEqual(call_args.kwargs["url"], "https://api.minimax.io/v1/chat/completions")

    @patch("httpx.AsyncClient")
    def test_oa_completion_sends_model_name(self, mock_client_cls):
        from chatlm import oa_completion
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        asyncio.run(oa_completion(
            url="https://api.minimax.io/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            model="MiniMax-M2.7-highspeed",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=50,
            temperature=0.01,
        ))
        call_args = mock_client.post.call_args
        sent_json = call_args.kwargs["json"]
        self.assertEqual(sent_json["model"], "MiniMax-M2.7-highspeed")
        self.assertEqual(sent_json["temperature"], 0.01)


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests (require MINIMAX_API_KEY)."""

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_minimax_greedy_until_live(self, mock_tokenizer):
        """Test MiniMaxLM.greedy_until with live API."""
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        mock_tok.decode.return_value = "test"
        mock_tok.eos_token_id = 2
        mock_tokenizer.return_value = mock_tok

        from minimax_lm import MiniMaxLM
        from chatlm import oa_completion
        lm = MiniMaxLM("MiniMax-M2.7")
        result = asyncio.run(oa_completion(
            url=MiniMaxLM.API_BASE_URL,
            headers=lm.headers,
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "What is 1+1? Answer briefly."}],
            max_tokens=32,
            temperature=lm._get_temperature(0.0),
        ))
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertTrue(len(result[0]) > 0)

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_factscore_live(self, mock_cache):
        """Test MiniMaxModel._generate with live API."""
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel("MiniMax-M2.7", cache_file="/tmp/test_cache")
        output, response = model._generate("What is 2+2?", max_output_length=32)
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration test",
    )
    @patch("factscore_package.lm.LM.load_cache", return_value={})
    def test_minimax_highspeed_model_live(self, mock_cache):
        """Test MiniMax-M2.7-highspeed model."""
        from factscore_package.minimax_lm import MiniMaxModel
        model = MiniMaxModel("MiniMax-M2.7-highspeed", cache_file="/tmp/test_cache")
        output, response = model._generate("Say hello", max_output_length=16)
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0)


if __name__ == "__main__":
    unittest.main()
