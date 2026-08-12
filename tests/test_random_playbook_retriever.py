import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "ace/core/playbook_retriever.py"
try:
    SPEC = importlib.util.spec_from_file_location("playbook_retriever", MODULE_PATH)
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
    IMPORT_ERROR = None
except ModuleNotFoundError as error:
    MODULE = None
    IMPORT_ERROR = error


class RandomPlaybookRetrieverTest(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, "NumPy is not installed in this lightweight test environment")
    def test_random_mode_is_deterministic_and_top_k_bounded(self):
        playbook = "\n".join([
            "## Strategies",
            "[s-00001] helpful=0 harmful=0 :: alpha",
            "[s-00002] helpful=0 harmful=0 :: beta",
            "[s-00003] helpful=0 harmful=0 :: gamma",
            "[s-00004] helpful=0 harmful=0 :: delta",
        ])
        retriever = MODULE.PlaybookRetriever(
            top_k=2, retrieval_mode="random", random_seed=42
        )
        retriever.update_index(playbook)

        first = retriever.retrieve("same query")
        second = retriever.retrieve("same query")

        self.assertEqual(first, second)
        self.assertEqual(sum(line.startswith("[") for line in first.splitlines()), 2)
        self.assertTrue(retriever.is_available)


if __name__ == "__main__":
    unittest.main()
