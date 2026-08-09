import unittest
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "ace" / "core" / "failure_memory.py"
SPEC = importlib.util.spec_from_file_location("failure_memory_under_test", MODULE_PATH)
failure_memory_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(failure_memory_module)
FailureMemoryBank = failure_memory_module.FailureMemoryBank
VERIFIED_FAILURE_TYPES = failure_memory_module.VERIFIED_FAILURE_TYPES
DIAGNOSTIC_ONLY_FAILURE_TYPES = failure_memory_module.DIAGNOSTIC_ONLY_FAILURE_TYPES


def fake_encoder(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append(
            [
                lowered.count("npv") + lowered.count("discount"),
                lowered.count("interest") + lowered.count("rate"),
                lowered.count("cash") + lowered.count("flow"),
                1.0,
            ]
        )
    return np.asarray(vectors, dtype=np.float32)


class VerifiedFailureMemoryTests(unittest.TestCase):
    def make_bank(self):
        return FailureMemoryBank(
            encoder=fake_encoder,
            embedding_dim=4,
            mode="verified",
            min_verifier_confidence=0.8,
            min_retrieval_score=0.0,
            top_k=2,
        )

    def test_rejects_unverified_failure(self):
        bank = self.make_bank()
        failure_id = bank.add_verified(
            question="Calculate NPV",
            predicted_answer="10",
            ground_truth="20",
            error_identification="wrong discounting",
            root_cause="rate conversion",
            key_insight="convert percent",
            verification={"verified": True, "confidence": 0.5},
            evidence=["answer mismatch"],
        )
        self.assertIsNone(failure_id)
        self.assertEqual(bank.size, 0)

    def test_general_verified_taxonomy_is_learning_focused(self):
        self.assertEqual(
            VERIFIED_FAILURE_TYPES,
            {
                "PLAYBOOK_GAP",
                "PLAYBOOK_MISAPPLICATION",
                "REASONING_ERROR",
                "CALCULATION_ERROR",
                "RETRIEVAL_ERROR",
                "VERIFICATION_ERROR",
                "INSTRUCTION_FOLLOWING_ERROR",
            },
        )
        self.assertTrue(
            VERIFIED_FAILURE_TYPES.isdisjoint(DIAGNOSTIC_ONLY_FAILURE_TYPES)
        )

    def test_accepts_general_verified_failure_type(self):
        bank = self.make_bank()
        failure_id = bank.add_verified(
            question="Calculate NPV",
            predicted_answer="100",
            ground_truth="90",
            error_identification="wrong arithmetic",
            root_cause="discount factor calculation",
            key_insight="check every period",
            verification={"verified": True, "confidence": 1.0},
            evidence=["ground_truth=90", "observed_answer=100"],
            failure_type="CALCULATION_ERROR",
        )
        self.assertIsNotNone(failure_id)
        self.assertEqual(bank.entries[0]["failure_type"], "CALCULATION_ERROR")

    def test_rejects_diagnostic_only_failure_from_active_memory(self):
        bank = self.make_bank()
        failure_id = bank.add_verified(
            question="Calculate NPV",
            predicted_answer="timeout",
            ground_truth="90",
            error_identification="server timeout",
            root_cause="environment unavailable",
            key_insight="retry later",
            verification={"verified": True, "confidence": 1.0},
            evidence=["HTTP timeout"],
            failure_type="ENVIRONMENT_ERROR",
        )
        self.assertIsNone(failure_id)

    def test_multistage_retrieval_and_curator_feedback(self):
        bank = self.make_bank()
        npv_id = bank.add_verified(
            question="Calculate NPV of cash flows",
            predicted_answer="100",
            ground_truth="90",
            error_identification="discount omitted",
            root_cause="discount rate was not converted",
            key_insight="convert percent before NPV",
            verification={"verified": True, "confidence": 1.0},
            evidence=["ground_truth=90", "observed_answer=100"],
            source="finance",
        )
        bank.add_verified(
            question="Calculate simple interest",
            predicted_answer="5",
            ground_truth="10",
            error_identification="period omitted",
            root_cause="interest period mismatch",
            key_insight="normalize periods",
            verification={"verified": True, "confidence": 1.0},
            evidence=["ground_truth=10", "observed_answer=5"],
            source="finance",
        )

        results = bank.retrieve("NPV discount cash flow", top_k=1)
        self.assertEqual(results[0]["failure_id"], npv_id)
        self.assertIn("retrieval_score", results[0])

        operations = [{"type": "ADD", "section": "FORMULAS", "content": "Convert rates"}]
        self.assertTrue(bank.record_curator_result(npv_id, operations, applied=True))
        entry = next(item for item in bank.entries if item["failure_id"] == npv_id)
        self.assertEqual(entry["status"], "curated")
        self.assertEqual(entry["curator_operations"], operations)

    def test_legacy_add_remains_available(self):
        bank = FailureMemoryBank(encoder=fake_encoder, embedding_dim=4, mode="legacy")
        failure_id = bank.add("NPV", "1", "2")
        self.assertIsNotNone(failure_id)
        self.assertEqual(bank.size, 1)

    def test_logs_full_verified_lifecycle(self):
        bank = self.make_bank()
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "detailed_llm_logs"
            bank.set_log_dir(str(log_dir), task_name="formula")
            failure_id = bank.add_verified(
                question="Calculate NPV",
                predicted_answer="100",
                ground_truth="90",
                error_identification="discount omitted",
                root_cause="rate conversion",
                key_insight="normalize rate",
                verification={"verified": True, "confidence": 1.0},
                evidence=["ground_truth=90", "observed_answer=100"],
                source="finance",
            )
            bank.retrieve("NPV discount", top_k=1)
            bank.record_curator_result(
                failure_id,
                [{"type": "ADD", "section": "FORMULAS", "content": "Normalize rate"}],
                applied=True,
            )

            event_path = log_dir / "failure_memory_events.jsonl"
            events = [json.loads(line)["event"] for line in event_path.read_text().splitlines()]
            self.assertIn("verification_gate", events)
            self.assertIn("semantic_candidates", events)
            self.assertIn("candidates_reranked", events)
            self.assertIn("curator_result_attached", events)

            snapshot_path = Path(directory) / "failure_memory_v2.jsonl"
            snapshot = [json.loads(line) for line in snapshot_path.read_text().splitlines()]
            self.assertEqual(snapshot[0]["failure_id"], failure_id)
            self.assertTrue(snapshot[0]["curator_applied"])


if __name__ == "__main__":
    unittest.main()
