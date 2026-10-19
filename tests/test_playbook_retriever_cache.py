import unittest
from unittest.mock import MagicMock
import numpy as np
from ace.core.playbook_retriever import PlaybookRetriever


class PlaybookRetrieverCacheTest(unittest.TestCase):
    def test_update_index_caches_embeddings(self):
        # Create a retriever
        retriever = PlaybookRetriever(
            top_k=2,
            retrieval_mode="semantic",
        )

        # Mock _encode to return fake embeddings of correct shape (dim=1024)
        mock_embeddings = lambda texts: np.random.randn(len(texts), 1024).astype(np.float32)
        retriever._encode = MagicMock(side_effect=mock_embeddings)

        playbook_v1 = "\n".join([
            "## Section 1",
            "[s-00001] helpful=0 harmful=0 :: alpha",
            "[s-00002] helpful=0 harmful=0 :: beta",
        ])

        # 1. Update index first time
        retriever.update_index(playbook_v1)

        # Should call _encode once with the two contents
        retriever._encode.assert_called_once_with(["alpha", "beta"])
        self.assertEqual(len(retriever._embedding_cache), 2)
        self.assertIn("alpha", retriever._embedding_cache)
        self.assertIn("beta", retriever._embedding_cache)

        # Reset mock
        retriever._encode.reset_mock()

        # 2. Update index second time with one modified/added bullet and one identical bullet
        playbook_v2 = "\n".join([
            "## Section 1",
            "[s-00001] helpful=0 harmful=0 :: alpha",
            "[s-00003] helpful=0 harmful=0 :: gamma",
        ])
        retriever.update_index(playbook_v2)

        # Should call _encode only with the new bullet ["gamma"]
        retriever._encode.assert_called_once_with(["gamma"])
        self.assertEqual(len(retriever._embedding_cache), 3)
        self.assertIn("alpha", retriever._embedding_cache)
        self.assertIn("beta", retriever._embedding_cache)
        self.assertIn("gamma", retriever._embedding_cache)


if __name__ == "__main__":
    unittest.main()
