import unittest

from ace.core.adversarial_agent import AdversarialAgent


class AdversarialSelectorTests(unittest.TestCase):
    def test_selector_rejects_unverified_candidates(self):
        candidates = [
            {
                "candidate_id": "c1",
                "verified": False,
                "verifier_confidence": 0.99,
                "learning_value": 1.0,
                "novelty": 1.0,
                "ambiguity": 0.0,
            }
        ]
        self.assertIsNone(AdversarialAgent.select_attack(candidates))

    def test_selector_prefers_verified_learning_value_and_novelty(self):
        candidates = [
            {
                "candidate_id": "c1",
                "verified": True,
                "verifier_confidence": 0.90,
                "learning_value": 0.20,
                "novelty": 0.20,
                "ambiguity": 0.10,
            },
            {
                "candidate_id": "c2",
                "verified": True,
                "verifier_confidence": 0.90,
                "learning_value": 0.90,
                "novelty": 0.80,
                "ambiguity": 0.10,
            },
        ]
        selected = AdversarialAgent.select_attack(candidates)
        self.assertEqual(selected["candidate_id"], "c2")
        self.assertGreater(selected["selection_score"], 0)

    def test_scores_are_clamped(self):
        self.assertEqual(AdversarialAgent._score(3), 1.0)
        self.assertEqual(AdversarialAgent._score(-2), 0.0)
        self.assertEqual(AdversarialAgent._score("bad", 0.4), 0.4)

    def test_string_false_is_not_treated_as_true(self):
        self.assertFalse(AdversarialAgent._strict_bool("false"))
        self.assertTrue(AdversarialAgent._strict_bool("true"))

    def test_full_pipeline_returns_only_verified_candidate(self):
        agent = AdversarialAgent(None, "fake", "fake-model", num_candidates=2)
        responses = iter([
            ({"vulnerabilities": [{
                "id": "v1", "type": "edge", "description": "missing zero case",
                "severity": 0.9, "testability": 1.0,
            }]}, {}),
            ({"candidates": [
                {
                    "candidate_id": "c1", "vulnerability_id": "v1",
                    "question": "q1", "context": "ctx", "target": "a1",
                    "target_derivation": "d1", "novelty": 1.0,
                    "learning_value": 1.0,
                },
                {
                    "candidate_id": "c2", "vulnerability_id": "v1",
                    "question": "q2", "context": "ctx", "target": "a2",
                    "target_derivation": "d2", "novelty": 0.1,
                    "learning_value": 0.1,
                },
            ]}, {}),
            ({"verifications": [
                {
                    "candidate_id": "c1", "valid": "false",
                    "independent_target": "a1", "target_matches": "true",
                    "confidence": 0.99, "ambiguity": 0.0,
                },
                {
                    "candidate_id": "c2", "valid": "true",
                    "independent_target": "a2", "target_matches": "true",
                    "confidence": 0.90, "ambiguity": 0.1,
                },
            ]}, {}),
        ])
        agent._call_json = lambda *args, **kwargs: next(responses)

        selected, info = agent.generate_attack(
            playbook="playbook", task_name="task", recent_question="q",
            recent_context="ctx", recent_target="target", log_dir=None,
        )

        self.assertEqual(selected["candidate_id"], "c2")
        self.assertTrue(selected["verified"])
        self.assertEqual(info["pipeline"], "mine-generate-verify-select")


if __name__ == "__main__":
    unittest.main()
