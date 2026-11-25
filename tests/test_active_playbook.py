import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("active_playbook", Path(__file__).resolve().parents[1] / "ace/core/active_playbook.py")
active = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = active
spec.loader.exec_module(active)


class ActivePlaybookTests(unittest.TestCase):
    def test_ranks_disagreement_before_grading_and_retains_success_probe(self):
        candidates = [active.Candidate("base", 1), active.Candidate("edit", 2, (0,))]
        cases = [
            active.ProbeCase("agree", "", False),
            active.ProbeCase("target", "", False),
            active.ProbeCase("guard", "", True),
        ]
        predictions = {"agree": [10, 10], "target": [10, 20], "guard": [3, 3]}
        labels = {"agree": 10, "target": 20, "guard": 3}
        rolled = []
        graded = []

        def rollout(case, candidate):
            rolled.append((case.key, candidate.playbook))
            return predictions[case.key][candidates.index(candidate)]

        def grade(case, answer):
            self.assertEqual(len(rolled), 6)
            graded.append(case.key)
            return answer == labels[case.key]

        decision = active.select_edit(
            candidates, cases, rollout, lambda answer: answer, grade,
            policy="disagreement", checker_cases=2, seed=42,
        )
        self.assertEqual(decision.selected, 1)
        self.assertEqual(decision.checked_cases, ("guard", "target"))
        self.assertEqual(graded, ["guard", "guard", "target", "target"])
        self.assertEqual(decision.checker_calls, 4)

    def test_holds_if_candidate_repairs_a_case_but_breaks_a_prior_success(self):
        candidates = [active.Candidate("base", 1), active.Candidate("edit", 2, (0,))]
        cases = [active.ProbeCase("bad", "", False), active.ProbeCase("good", "", True)]
        decision = active.select_edit(
            candidates, cases, lambda case, candidate: case.prior_success == (candidate.playbook == "base"),
            lambda success: success, lambda _, result: result,
            policy="random", checker_cases=2, seed=11,
        )
        self.assertEqual(decision.selected, 0)
        self.assertEqual(decision.candidate_scores[1], (1, 1))

    def test_holds_when_no_candidate_fits(self):
        versions = active.candidate_versions("base", 9, 2, 3, lambda _: ("too long", 99), lambda text: False)
        self.assertEqual(versions, [active.Candidate("base", 9)])

    def test_pool_retains_trigger_and_prior_success(self):
        cases = [active.ProbeCase("s", "discounting", True), active.ProbeCase("f", "margin", False),
                 active.ProbeCase("t", "margin", False)]
        self.assertEqual([case.key for case in active.probe_pool(cases, 2)], ["t", "s"])


if __name__ == "__main__":
    unittest.main()
