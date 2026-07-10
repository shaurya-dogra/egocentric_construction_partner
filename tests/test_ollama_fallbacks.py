import unittest

from integration.vlm_hook import VLMHook
from reasoning.document_store import DocumentStore


class OllamaFallbackTests(unittest.TestCase):
    def test_chat_endpoint_candidates_include_generate_fallback(self) -> None:
        urls = VLMHook._build_chat_url_candidates("http://localhost:11434/api/chat")
        self.assertEqual(
            urls,
            [
                "http://localhost:11434/api/chat",
                "http://localhost:11434/api/generate",
                "http://localhost:11434/v1/chat/completions",
            ],
        )

    def test_embed_endpoint_candidates_include_legacy_fallback(self) -> None:
        urls = DocumentStore._build_embed_url_candidates("http://localhost:11434/api/embed")
        self.assertEqual(
            urls,
            [
                "http://localhost:11434/api/embed",
                "http://localhost:11434/api/embeddings",
                "http://localhost:11434/v1/embeddings",
            ],
        )


if __name__ == "__main__":
    unittest.main()
