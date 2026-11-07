"""Exercise the real Generator with fake LLM/embedding boundaries."""

import ast
import json
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "ace/core/generator.py"
tree = ast.parse(source.read_text())
definitions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Generator"], type_ignores=[])
namespace = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                 json=json, re=re, PlaybookBudget=object,
                 GENERATOR_PROMPT="Playbook: {}\nReflection: {}\nQuestion: {}\nContext: {}")
exec(compile(definitions, str(source), "exec"), namespace)


class GeneratorTests(unittest.TestCase):
    def test_flag_off_keeps_original_context_and_flag_on_adds_only_retrieved_lessons(self):
        call = Mock(return_value=('{"answer":"10"}', {}))
        namespace["timed_llm_call"] = call
        generator = namespace["Generator"](None, "vllm", "fake")
        generator.generate("Question", "Playbook", context="Original")
        self.assertTrue(call.call_args.args[3].endswith("Context: Original"))
        memory = Mock()
        memory.context.return_value = "\nRetrieved training lessons"
        generator.amore_memory = memory
        generator.generate("Question", "Playbook", context="Original")
        self.assertIn("Original\nRetrieved training lessons", call.call_args.args[3])
        memory.context.assert_called_once_with("Question")
        memory.observe.assert_not_called()

    def test_playbook_budget_is_checked_before_memory_or_the_llm(self):
        generator = namespace["Generator"](None, "vllm", "fake")
        generator.playbook_budget = Mock()
        generator.playbook_budget.require_within_budget.side_effect = ValueError("overflow")
        generator.amore_memory = Mock()
        with self.assertRaises(ValueError):
            generator.generate("Q", "oversized")
        generator.amore_memory.context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
