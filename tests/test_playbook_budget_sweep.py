"""Exercise the sweep CLI without invoking a model or writing experiment data."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_playbook_budget_sweep.sh"
FULL_OPERATIONS_SCRIPT = SCRIPT.with_name("run_playbook_budget_sweep_full_operations.sh")


class PlaybookBudgetSweepTests(unittest.TestCase):
    def run_sweep(self, budgets: str, *arguments: str, script: Path = SCRIPT) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "unused_results"
            environment = {
                **os.environ,
                "TASKS": "formula finer_0.5",
                "TOKEN_BUDGETS": budgets,
                "RESULTS_ROOT": str(results),
            }
            result = subprocess.run(
                ["bash", str(script), "--dry-run", *arguments],
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

    def test_full_operations_sweep_preserves_budgets_and_enables_only_requested_components(self):
        result = self.run_sweep("4000 8000 10000", "--max_train_samples", "50", script=FULL_OPERATIONS_SCRIPT)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("uv ")]
        self.assertEqual(len(commands), 6)
        for command in commands:
            self.assertIn("--use_budgeted_curator", command)
            self.assertIn("--use_lifecycle_curator", command)
            self.assertIn("--use_dbscan_merge_candidates", command)
            self.assertEqual(command[command.index("--curator_dbscan_similarity_threshold") + 1], "0.9")
            self.assertEqual(command[command.index("--prune_unused_interval") + 1], "50")
            self.assertEqual(command[command.index("--max_train_samples") + 1], "50")
            self.assertEqual(command[command.index("--seed") + 1], "42")
            for disabled in ("--use_rae", "--use_verified_failure_memory", "--use_failure_memory",
                             "--use_adversarial", "--use_bulletpoint_analyzer", "--use_dbscan_merge"):
                self.assertNotIn(disabled, command)
        self.assertEqual(
            {command[command.index("--playbook_token_budget") + 1] for command in commands},
            {"4000", "8000", "10000"},
        )

    def test_full_operations_default_output_does_not_mix_with_budget_only_results(self):
        environment = {**os.environ, "TASKS": "formula", "TOKEN_BUDGETS": "4000"}
        environment.pop("RESULTS_ROOT", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(FULL_OPERATIONS_SCRIPT), "--dry-run"],
                cwd=directory, env=environment, capture_output=True, text=True,
                check=False, timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(shlex.split(line) for line in result.stdout.splitlines() if line.startswith("uv "))
        self.assertEqual(command[command.index("--save_path") + 1],
                         "results/playbook_budget_sweep_full_operations/formula/budget_4000")


if __name__ == "__main__":
    unittest.main()
