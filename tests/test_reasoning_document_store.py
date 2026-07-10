import tempfile
import unittest
from pathlib import Path

from reasoning.document_store import DocumentStore


class DocumentStoreTests(unittest.TestCase):
    def test_small_demo_corpus_uses_direct_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "documents"
            docs_dir.mkdir(parents=True, exist_ok=True)
            (docs_dir / "spec.txt").write_text("Install scaffold. Verify ladder. Wear hardhat.")
            (docs_dir / "safety.md").write_text("Procedure: inspect tools before use.")

            store = DocumentStore(
                {
                    "directory": str(docs_dir),
                    "index_path": str(Path(tmpdir) / "index.json"),
                    "direct_doc_limit": 6,
                    "direct_char_limit": 10000,
                }
            )
            store.sync()

            self.assertEqual(store.strategy(), "direct_context")
            hits = store.retrieve("What is the tool procedure?", top_k=2)
            self.assertGreaterEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
