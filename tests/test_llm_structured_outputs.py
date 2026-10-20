"""Verify wire-format requests without requiring SDKs or a running model."""

import importlib.util
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch


with patch.dict(sys.modules):
    sys.modules["openai"] = types.ModuleType("openai")
    logger = types.ModuleType("logger")
    logger.log_llm_call, logger.log_problematic_request = Mock(), Mock()
    sys.modules["logger"] = logger
    spec = importlib.util.spec_from_file_location("structured_llm_under_test", Path(__file__).resolve().parents[1] / "llm.py")
    LLM = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = LLM
    spec.loader.exec_module(LLM)


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["type"], "properties": {"type": {"type": "string", "enum": ["UPDATE"]}},
}


class StructuredOutputRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"type":"UPDATE"}'))],
            usage=None,
        )

    def call(self, **options):
        return LLM.timed_llm_call(self.client, "vllm", "fake-model", "JSON request", "curator", "budget_test", **options)

    def test_schema_is_forwarded_and_recorded_without_mutating_the_caller(self):
        original = deepcopy(SCHEMA)
        _, info = self.call(response_schema=SCHEMA)
        payload = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "ace_response", "strict": True, "schema": SCHEMA},
        })
        self.assertEqual(info["response_format"], "json_schema")
        self.assertEqual(SCHEMA, original)

    def test_legacy_text_requests_are_unchanged(self):
        self.call()
        self.assertNotIn("response_format", self.client.chat.completions.create.call_args.kwargs)

    def test_legacy_json_mode_remains_syntax_only(self):
        self.call(use_json_mode=True)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_schema_takes_precedence_over_json_object_mode(self):
        self.call(response_schema=SCHEMA, use_json_mode=True)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs["response_format"]["type"], "json_schema")

    def test_unsupported_schema_errors_are_not_retried_or_downgraded(self):
        failure = RuntimeError("HTTP 400: unsupported response_format schema")
        self.client.chat.completions.create.side_effect = failure
        with patch.object(LLM.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                self.call(response_schema=SCHEMA)
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        sleep.assert_not_called()

    def test_native_sglang_cannot_silently_ignore_a_schema(self):
        with self.assertRaisesRegex(ValueError, "native SGLang"):
            LLM.timed_llm_call(self.client, "sglang", "fake-model", "JSON request", "curator", "budget_test", response_schema=SCHEMA)
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
