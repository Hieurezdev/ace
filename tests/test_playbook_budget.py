"""Budget transactions with real Playbook edits and fake external LLM calls."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the actual lightweight implementation without loading embedding models,
# provider SDKs, or the GPU runtime through ace/__init__.py.
with patch.dict(sys.modules):
    utils_stub = types.ModuleType("utils")
    utils_stub.get_section_slug = lambda section: "calc"
    sys.modules["utils"] = utils_stub
    PLAYBOOK = load_module("playbook_utils", ROOT / "playbook_utils.py")
    load_module("logger", ROOT / "logger.py")
    llm_stub = types.ModuleType("llm")
    llm_stub.timed_llm_call = Mock()
    sys.modules["llm"] = llm_stub
    for name in ("budget_test_ace", "budget_test_ace.core", "budget_test_ace.prompts"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    BUDGET = load_module("budget_test_ace.core.playbook_budget", ROOT / "ace/core/playbook_budget.py")
    load_module("budget_test_ace.prompts.curator", ROOT / "ace/prompts/curator.py")
    load_module("budget_test_ace.prompts.generator", ROOT / "ace/prompts/generator.py")
    CURATOR = load_module("budget_test_ace.core.curator", ROOT / "ace/core/curator.py")
    GENERATOR = load_module("budget_test_ace.core.generator", ROOT / "ace/core/generator.py")
    core = sys.modules["budget_test_ace.core"]
    for name in ("Generator", "Reflector", "Curator", "BulletpointAnalyzer", "PlaybookRetriever",
                 "FailureMemoryBank", "AdversarialAgent"):
        setattr(core, name, object)
    STRESS = load_module("budget_test_ace.core.stress_test", ROOT / "ace/core/stress_test.py")
    ORCHESTRATOR = load_module("budget_test_ace.ace", ROOT / "ace/ace.py")


OLD = "## RULES\n[calc-00001] helpful=5 harmful=1 :: " + "long explanation " * 12
ADD = {"type": "ADD", "section": "rules", "content": "New reusable insight."}
UPDATE = {"type": "UPDATE", "target_id": "calc-00001", "content": "Concise valid rule.",
          "reason": "Remove redundant prose, retain the same conditions."}


def response(operations):
    return json.dumps({"reasoning": "Evidence-based local edit.", "operations": operations}), {}


class PlaybookBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.budget = BUDGET.PlaybookBudget(len(OLD) + 5, len, 2, self.path, "test-character-tokenizer")
        self.repair = Mock(return_value=[deepcopy(UPDATE)])
        self.candidate, self.next_id = PLAYBOOK.apply_curator_operations(OLD, [deepcopy(ADD)], 2)
        llm_stub.timed_llm_call.reset_mock(side_effect=True, return_value=True)

    def events(self):
        return [json.loads(line) for line in (self.path / "playbook_budget_events.jsonl").read_text().splitlines()]

    def test_fitting_candidate_does_not_call_maintenance(self):
        result = self.budget.enforce(OLD, OLD, 2, 2, self.repair, "fits")
        self.assertTrue(result.accepted)
        self.assertEqual(result.playbook, OLD)
        self.repair.assert_not_called()

    def test_local_update_fits_and_preserves_new_insight_and_counters(self):
        edits = [deepcopy(UPDATE)]
        self.repair.return_value = edits
        original_edits = deepcopy(edits)
        result = self.budget.enforce(OLD, self.candidate, 2, self.next_id, self.repair, "update")
        self.assertTrue(result.accepted)
        self.assertLessEqual(len(result.playbook), self.budget.limit)
        self.assertIn("[calc-00001] helpful=5 harmful=1 :: Concise valid rule.", result.playbook)
        self.assertIn("New reusable insight.", result.playbook)
        self.assertEqual(result.next_global_id, 3)
        self.assertEqual(edits, original_edits)
        self.assertEqual(self.events()[-1]["event"], "committed")

    def test_merge_conserves_evidence_and_uses_one_new_id(self):
        self.repair.return_value = [{
            "type": "MERGE", "source_ids": ["calc-00001", "calc-00002"], "section": "rules",
            "content": "Both valid conditions.", "reason": "Combine duplicate guidance.",
        }]
        result = self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "merge")
        self.assertTrue(result.accepted)
        self.assertIn("[calc-00003] helpful=5 harmful=1", result.playbook)
        self.assertNotIn("[calc-00001]", result.playbook)
        self.assertEqual(result.next_global_id, 4)

    def test_merge_uses_the_existing_section_normalization(self):
        playbook = self.candidate.replace("## RULES", "## FORMULAS & CALCULATIONS")
        BUDGET.validate_maintenance(playbook, [{
            "type": "MERGE", "source_ids": ["calc-00001", "calc-00002"],
            "section": "formulas_and_calculations", "content": "Both conditions.",
            "reason": "Preserve both conditions without duplicate prose.",
        }])

    def test_failed_repairs_roll_back_content_and_id_and_archive_candidate(self):
        self.repair.return_value = []
        result = self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "rollback")
        self.assertFalse(result.accepted)
        self.assertEqual((result.playbook, result.next_global_id, result.maintenance), (OLD, 2, []))
        self.assertEqual(self.repair.call_count, 1)
        archive = json.loads((self.path / "playbook_budget_candidates.jsonl").read_text())
        self.assertEqual(archive["candidate"], self.candidate)
        self.assertFalse(archive["accepted"])

    def test_partially_repaired_but_oversized_candidate_is_not_committed(self):
        self.repair.return_value = [{**UPDATE, "content": "long explanation " * 11}]
        result = self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "partial")
        self.assertFalse(result.accepted)
        self.assertEqual(result.playbook, OLD)
        self.assertEqual(result.next_global_id, 2)

    def test_expanding_repair_cannot_replace_candidate(self):
        self.repair.return_value = [{**UPDATE, "content": "expansion " * 100}]
        result = self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "expansion")
        self.assertFalse(result.accepted)
        self.assertEqual(self.repair.call_count, 1)
        self.assertEqual(result.playbook, OLD)

    def test_invalid_targets_overlaps_and_structural_injection_are_rejected(self):
        batches = [
            [{**UPDATE, "target_id": "missing"}],
            [deepcopy(UPDATE), deepcopy(UPDATE)],
            [{**UPDATE, "content": "Rule\n## INJECTED SECTION"}],
            [deepcopy(ADD)],
            [{"type": "MERGE", "source_ids": ["calc-00001", "calc-00001"],
              "section": "rules", "content": "Rule", "reason": "Duplicate ID"}],
        ]
        for operations in batches:
            with self.subTest(operations=operations), self.assertRaises(ValueError):
                BUDGET.validate_maintenance(self.candidate, operations)

    def test_oversized_initial_playbook_fails_before_repair(self):
        with self.assertRaisesRegex(ValueError, "Supply a smaller initial Playbook"):
            self.budget.enforce(self.candidate, OLD, 3, 2, self.repair, "initial")
        self.repair.assert_not_called()

    def test_zero_attempts_rolls_back_without_llm_call(self):
        self.budget.repair_attempts = 0
        result = self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "zero")
        self.assertFalse(result.accepted)
        self.repair.assert_not_called()

    def test_maintenance_io_failure_is_not_silently_swallowed(self):
        self.repair.side_effect = OSError("Backend unavailable")
        with self.assertRaises(OSError):
            self.budget.enforce(OLD, self.candidate, 2, 3, self.repair, "io")

    def curator(self, enabled):
        curator = CURATOR.Curator(None, "fake", "fake-model")
        curator.playbook_budget = self.budget if enabled else None
        return curator

    def curate(self, curator, allowed=None):
        return curator.curate(
            current_playbook=OLD, recent_reflection="Observed new insight", question_context="Task",
            current_step=1, total_samples=2, token_budget=self.budget.limit, playbook_stats={},
            next_global_id=2, allowed_operations=allowed or ["ADD"], call_id="train_budget",
            log_dir=str(self.path / "detailed_llm_logs"),
        )

    def test_curator_disabled_retains_legacy_soft_budget_behavior(self):
        llm_stub.timed_llm_call.return_value = response([ADD])
        book, next_id, operations, _ = self.curate(self.curator(False))
        self.assertGreater(len(book), self.budget.limit)
        self.assertEqual(next_id, 3)
        self.assertEqual(len(operations), 1)
        self.assertEqual(llm_stub.timed_llm_call.call_count, 1)

    def test_curator_proposal_and_maintenance_form_one_transaction(self):
        llm_stub.timed_llm_call.side_effect = [response([ADD]), response([UPDATE])]
        book, next_id, operations, _ = self.curate(self.curator(True))
        self.assertLessEqual(len(book), self.budget.limit)
        self.assertIn("New reusable insight.", book)
        self.assertEqual([op["type"] for op in operations], ["ADD", "UPDATE"])
        self.assertEqual(next_id, 3)
        maintenance_prompt = llm_stub.timed_llm_call.call_args_list[1].args[3]
        self.assertIn(self.candidate, maintenance_prompt)

    def test_budget_pressure_does_not_bypass_delete_evidence_gate(self):
        deletion = {"type": "DELETE", "target_id": "calc-00001", "reason": "Make space"}
        llm_stub.timed_llm_call.side_effect = [response([ADD]), response([deletion]), response([deletion])]
        book, next_id, operations, _ = self.curate(self.curator(True), ["ADD", "DELETE"])
        self.assertEqual((book, next_id, operations), (OLD, 2, []))

    def test_malformed_maintenance_rolls_back_without_retrying_invalid_json(self):
        llm_stub.timed_llm_call.side_effect = [response([ADD]), ("invalid JSON", {})]
        book, next_id, operations, _ = self.curate(self.curator(True))
        self.assertEqual((book, next_id, operations), (OLD, 2, []))
        self.assertEqual(llm_stub.timed_llm_call.call_count, 2)
        self.assertEqual(self.events()[-2]["event"], "repair_invalid")

    def test_two_training_samples_keep_one_evolving_bounded_playbook(self):
        ace = ORCHESTRATOR.ACE.__new__(ORCHESTRATOR.ACE)
        ace.playbook, ace.next_global_id = OLD, 2
        ace.curator = self.curator(True)
        ace.generator = GENERATOR.Generator(None, "fake", "fake-model")
        ace.generator.playbook_budget = self.budget
        ace.reflector = types.SimpleNamespace(reflect=Mock(return_value=("New insight", [], {})))
        ace.failure_memory, ace.playbook_retriever = None, None
        ace.curator_allowed_operations = ["ADD"]
        ace.delete_harmful_margin, ace.delete_min_harmful = 4, 3
        ace.use_bulletpoint_analyzer, ace.use_rae, ace.use_adversarial = False, False, False
        processor = types.SimpleNamespace(answer_is_correct=lambda answer, target: answer == target)
        llm_stub.timed_llm_call.side_effect = [
            ("wrong", {}), ("answer", {}), response([ADD]), response([UPDATE]), ("answer", {}),
            ("wrong", {}), ("answer", {}), response([ADD]), ("answer", {}),
        ]
        config = ace._extract_config_params({"playbook_token_budget": self.budget.limit})
        with patch.multiple(ORCHESTRATOR, create=True, count_tokens=len,
                            extract_answer=lambda text: text,
                            extract_playbook_bullets=lambda book, ids: []):
            for step in (1, 2):
                before = ace.playbook
                pre, post, _ = ace._train_single_sample(
                    task_dict={"question": f"Task {step}", "target": "answer", "context": ""},
                    data_processor=processor, step_id=f"train_e_1_s_{step}", epoch=1, step=step,
                    usage_log_path=str(self.path / "usage.jsonl"), log_dir=str(self.path / "llm"),
                    config_params=config, total_samples=2,
                )
                self.assertEqual((pre, post), ("wrong", "answer"))
                self.assertLessEqual(len(ace.playbook), self.budget.limit)
                self.assertNotEqual(ace.playbook, before)
                self.assertIn(ace.playbook, llm_stub.timed_llm_call.call_args.args[3])
        self.assertEqual(ace.next_global_id, 4)

    def test_counter_update_is_guarded_at_the_orchestrator_boundary(self):
        ace = ORCHESTRATOR.ACE.__new__(ORCHESTRATOR.ACE)
        ace.playbook = OLD
        ace.next_global_id = 2
        ace.curator = self.curator(True)
        ace.curator_allowed_operations = ["ADD"]
        ace.delete_harmful_margin, ace.delete_min_harmful = 4, 3
        updated_counts = OLD.replace("helpful=5", "helpful=50000000000000")
        llm_stub.timed_llm_call.return_value = response([UPDATE])
        ace._accept_auxiliary_playbook(updated_counts, "counter_growth", str(self.path))
        self.assertIn("helpful=50000000000000", ace.playbook)
        self.assertIn("Concise valid rule.", ace.playbook)
        self.assertLessEqual(len(ace.playbook), self.budget.limit)

    def test_generator_uses_full_canonical_playbook_and_rejects_overflow(self):
        generator = GENERATOR.Generator(None, "fake", "fake-model")
        generator.playbook_budget = self.budget
        llm_stub.timed_llm_call.return_value = ("answer", {})
        generator.generate("question", OLD)
        self.assertIn(OLD, llm_stub.timed_llm_call.call_args.args[3])
        llm_stub.timed_llm_call.reset_mock()
        with self.assertRaises(ValueError):
            generator.generate("question", self.candidate)
        llm_stub.timed_llm_call.assert_not_called()

    def test_model_tokenizer_counter_excludes_added_special_tokens(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3]
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Mock()
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        with patch.dict(sys.modules, {"transformers": transformers}):
            count = BUDGET.load_token_counter("local-qwen-tokenizer")
            self.assertEqual(count("Playbook"), 3)
        tokenizer.encode.assert_called_once_with("Playbook", add_special_tokens=False)
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "local-qwen-tokenizer", trust_remote_code=False,
        )


class BudgetCliTests(unittest.TestCase):
    def setUp(self):
        self.modules_patch = patch.dict(sys.modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        finance = types.ModuleType("budget_test_finance")
        finance.__path__ = []
        processor = types.ModuleType("budget_test_finance.data_processor")
        processor.DataProcessor = Mock()
        ace = types.ModuleType("ace")
        ace.ACE = ORCHESTRATOR.ACE
        utils = types.ModuleType("utils")
        utils.initialize_clients, utils.set_global_seed = Mock(), Mock()
        sys.modules.update({
            "budget_test_finance": finance,
            "budget_test_finance.data_processor": processor,
            "ace": ace, "ace.core.stress_test": STRESS,
            "openai": types.ModuleType("openai"), "utils": utils,
        })
        self.runner = load_module("budget_test_finance.run", ROOT / "eval/finance/run.py")

    def test_budget_flags_are_opt_in_and_accept_model_tokenizer_settings(self):
        with patch.object(sys, "argv", ["run", "--task_name", "formula", "--save_path", "results"]):
            self.assertFalse(self.runner.parse_args().use_budgeted_curator)
        with patch.object(sys, "argv", [
            "run", "--task_name", "formula", "--save_path", "results", "--use_budgeted_curator",
            "--playbook_token_budget", "8000", "--budget_repair_attempts", "3",
            "--playbook_tokenizer", "local-tokenizer",
        ]):
            args = self.runner.parse_args()
        self.assertTrue(args.use_budgeted_curator)
        self.assertEqual((args.playbook_token_budget, args.budget_repair_attempts), (8000, 3))
        self.assertEqual(args.playbook_tokenizer, "local-tokenizer")

    def test_conflicting_retrieval_ablation_fails_before_data_or_api_calls(self):
        with patch.object(sys, "argv", [
            "run", "--task_name", "formula", "--save_path", "results", "--use_budgeted_curator", "--use_rae",
        ]), self.assertRaisesRegex(ValueError, "Disable RAE"):
            self.runner.main()
        self.runner.initialize_clients.assert_not_called()


if __name__ == "__main__":
    unittest.main()
