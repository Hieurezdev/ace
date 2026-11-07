"""Behavior tests for the opt-in idea4 memory, without provider or GPU access."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ace/core/amore_memory.py"
spec = importlib.util.spec_from_file_location("amore_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AmoreMemory = module.AmoreMemory
ENCOUNTER = {"task_id": "train-1", "question": "Compute margin", "trajectory": "answer 5", "feedback": "expected 10"}


class AmoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "memory.json"
        self.call = Mock(side_effect=self.respond)
        self.insights = [{"text": "Check arithmetic.", "evidence_quote": "expected 10"}]

    def respond(self, stage, prompt):
        if stage == "splitter":
            return '{"should_split":false,"confidence":0.9}'
        if stage == "compressor":
            return '{"title":"Margins","summary":"Compute financial margins."}'
        if stage == "hypothesizer":
            return "Margins require checking."
        return json.dumps({"insights": self.insights})

    def memory(self, **kwargs):
        return AmoreMemory(self.path, self.call, lambda texts: [[1., 0.] for _ in texts], **kwargs)

    def test_admission_preserves_original_evidence_and_retrieval_never_calls_llm(self):
        memory = self.memory(max_buffer_tasks=1)
        memory.observe(ENCOUNTER, "Playbook")
        self.assertEqual(memory.cases[0]["raw_conversation"], [ENCOUNTER])
        resolver = json.loads(self.call.call_args.args[1])
        self.assertEqual(resolver["raw_conversation"][0]["feedback"], "expected 10")
        self.call.reset_mock()
        self.assertIn("Check arithmetic.", memory.context("margin"))
        self.call.assert_not_called()
        self.assertNotIn("expected 10", memory.context("margin"))

    def test_empty_insights_are_not_admitted(self):
        self.insights = []
        memory = self.memory(max_buffer_tasks=1)
        memory.observe(ENCOUNTER, "")
        self.assertEqual(memory.cases, [])
        self.assertEqual(memory.buffer, [])

    def test_invalid_evidence_cannot_commit_and_pending_data_survives_resume(self):
        self.insights = [{"text": "Invented.", "evidence_quote": "not in trajectory"}]
        memory = self.memory(max_buffer_tasks=1)
        with self.assertRaises(ValueError):
            memory.observe(ENCOUNTER, "")
        resumed = self.memory()
        self.assertEqual(resumed.buffer, [ENCOUNTER])
        self.assertEqual(resumed.cases, [])

    def test_eval_is_read_only_and_does_not_use_pending_training_evidence(self):
        memory = self.memory()
        memory.observe(ENCOUNTER, "")
        before = self.path.read_bytes()
        evaluation = self.memory(read_only=True)
        self.call.reset_mock()
        self.assertEqual(evaluation.context("query"), "")
        with self.assertRaises(RuntimeError):
            evaluation.observe(ENCOUNTER, "")
        with self.assertRaises(RuntimeError):
            evaluation.flush("")
        self.call.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_eval_memory_fails_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            self.memory(read_only=True)

    def test_boundary_checks_previous_tasks_before_inserting_new_question(self):
        memory = self.memory()
        memory.observe(ENCOUNTER, "")
        original = self.respond
        self.call.side_effect = lambda stage, prompt: ('{"should_split":true,"confidence":0.9}' if stage == "splitter" else original(stage, prompt))
        second = dict(ENCOUNTER, task_id="train-2", question="New task")
        memory.observe(second, "")
        split = json.loads(self.call.call_args_list[0].args[1])
        self.assertEqual(split["previous_questions"], ["Compute margin"])
        self.assertEqual(memory.cases[0]["raw_conversation"], [ENCOUNTER])
        self.assertEqual(memory.buffer, [second])

    def test_invalid_parameters_rejected(self):
        for key in ("top_k", "consolidation_frequency", "max_buffer_tasks"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.memory(**{key: 0})

    def test_bad_boundary_schema_fails_without_changing_the_buffer(self):
        memory = self.memory()
        memory.observe(ENCOUNTER, "")
        self.call.side_effect = lambda stage, prompt: '{"should_split":"false","confidence":1}'
        with self.assertRaises(ValueError):
            memory.observe(dict(ENCOUNTER, task_id="new"), "")
        self.assertEqual(self.memory().buffer, [ENCOUNTER])


if __name__ == "__main__":
    unittest.main()
