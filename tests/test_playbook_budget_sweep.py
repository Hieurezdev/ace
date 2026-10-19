"""Exercise the sweep CLI without invoking a model or writing experiment data."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_playbook_budget_sweep.sh"


class PlaybookBudgetSweepTests(unittest.TestCase):
    def run_sweep(self, budgets: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "unused_results"
            environment = {
                **os.environ,
                "TASKS": "formula finer_0.5",
                "TOKEN_BUDGETS": budgets,
                "RESULTS_ROOT": str(results),
            }
            result = subprocess.run(
                ["bash", str(SCRIPT), "--dry-run", *arguments],
                cwd=directory, env=environment, capture_output=True, text=True,
                check=False, timeout=10,
            )
            self.assertFalse(results.exists(), "Dry runs must not create experiment files")
            return result

    def test_sweep_emits_one_distinct_run_per_dataset_and_budget(self):
        result = self.run_sweep("2000 8000", "--max_train_samples", "50")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("uv ")]
        self.assertEqual(len(commands), 4)
        combinations = set()
        destinations = set()
        for command in commands:
            self.assertEqual(command[:6], ["uv", "run", "python", "-m", "eval.finance.run", "--mode"])
            self.assertIn("--use_budgeted_curator", command)
            self.assertEqual(command[command.index("--max_train_samples") + 1], "50")
            self.assertEqual(command[command.index("--seed") + 1], "42")
            task = command[command.index("--task_name") + 1]
            budget = command[command.index("--playbook_token_budget") + 1]
            destination = command[command.index("--save_path") + 1]
            combinations.add((task, budget))
            destinations.add(destination)
            self.assertTrue(destination.endswith(f"/{task}/budget_{budget}"))
        self.assertEqual(combinations, {("formula", "2000"), ("formula", "8000"),
                                        ("finer_0.5", "2000"), ("finer_0.5", "8000")})
        self.assertEqual(len(destinations), 4)

    def test_invalid_budget_fails_before_emitting_any_runs(self):
        for budgets in ("2000 0", "-1", "abc"):
            with self.subTest(budgets=budgets):
                result = self.run_sweep(budgets)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Invalid token budget", result.stderr)
                self.assertNotIn("uv run", result.stdout)

    def test_dry_run_without_forwarded_arguments_works_with_system_bash(self):
        result = self.run_sweep("4000")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(line.startswith("uv ") for line in result.stdout.splitlines()), 2)

    def test_budget_override_cannot_mislabel_sweep_results(self):
        result = self.run_sweep("2000", "--playbook_token_budget=9999")
        self.assertEqual(result.returncode, 2)
        self.assertIn("TOKEN_BUDGETS", result.stderr)


if __name__ == "__main__":
    unittest.main()
