"""Local construction document ingestion and retrieval."""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reasoning.models import RetrievalHit

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional until deps are installed
    PdfReader = None

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".json", ".yaml", ".yml"}


@dataclass
class _Chunk:
    document_id: str
    title: str
    content: str
    chunk_index: int
    source_path: str
    embedding: Optional[list[float]] = None


class DocumentStore:
    """Hackathon-scale document store with direct-context and local embedding modes."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.docs_dir = Path(config.get("directory", "data/documents"))
        self.index_path = Path(config.get("index_path", "data/indexes/documents_index.json"))
        self.api_url = config.get("embed_api_url", "http://localhost:11434/api/embed")
        self._embed_urls = self._build_embed_url_candidates(self.api_url)
        self.embedding_model = config.get("embedding_model", "nomic-embed-text")
        self.force_embeddings = bool(config.get("force_embeddings", False))
        self.direct_doc_limit = int(config.get("direct_doc_limit", 6))
        self.direct_char_limit = int(config.get("direct_char_limit", 48000))
        self.chunk_size = int(config.get("chunk_size", 1200))
        self.chunk_overlap = int(config.get("chunk_overlap", 150))
        self.top_k = int(config.get("top_k", 4))
        self._chunks: list[_Chunk] = []
        self._strategy = "direct_context"
        self._loaded = False

    def sync(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = [
            path for path in self.docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        ]
        self._chunks = []
        for path in sorted(files):
            text = self._extract_text(path)
            if not text.strip():
                continue
            self._chunks.extend(self._chunk_document(path, text))
        self._strategy = self._choose_strategy()
        if self._strategy == "embedding_index":
            self._ensure_index()
        self._loaded = True

    def strategy(self) -> str:
        self._ensure_loaded()
        return self._strategy

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalHit]:
        self._ensure_loaded()
        top_k = top_k or self.top_k
        if not self._chunks:
            return []
        if self._strategy == "direct_context":
            return [
                RetrievalHit(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=1.0,
                    chunk_index=chunk.chunk_index,
                    source_path=chunk.source_path,
                )
                for chunk in self._chunks[:top_k]
            ]
        query_embedding = self._embed(query)
        if not query_embedding:
            logger.warning("Embedding failed for query; falling back to first chunks.")
            return self.retrieve_direct(top_k)
        scored: list[tuple[float, _Chunk]] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            scored.append((self._cosine_similarity(query_embedding, chunk.embedding), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                document_id=chunk.document_id,
                title=chunk.title,
                content=chunk.content,
                score=score,
                chunk_index=chunk.chunk_index,
                source_path=chunk.source_path,
            )
            for score, chunk in scored[:top_k]
        ]

    def retrieve_direct(self, top_k: Optional[int] = None) -> list[RetrievalHit]:
        self._ensure_loaded()
        top_k = top_k or self.top_k
        return [
            RetrievalHit(
                document_id=chunk.document_id,
                title=chunk.title,
                content=chunk.content,
                score=1.0,
                chunk_index=chunk.chunk_index,
                source_path=chunk.source_path,
            )
            for chunk in self._chunks[:top_k]
        ]

    def build_prompt_context(self, query: str, top_k: Optional[int] = None) -> dict[str, object]:
        hits = self.retrieve(query, top_k=top_k)
        return {
            "strategy": self._strategy,
            "hits": [hit.to_dict() for hit in hits],
        }

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.sync()

    def _choose_strategy(self) -> str:
        total_chars = sum(len(chunk.content) for chunk in self._chunks)
        doc_count = len({chunk.document_id for chunk in self._chunks})
        if self.force_embeddings:
            return "embedding_index"
        if doc_count <= self.direct_doc_limit and total_chars <= self.direct_char_limit:
            return "direct_context"
        return "embedding_index"

    def _ensure_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = self._load_existing_index()
        if serialized and serialized.get("embedding_model") == self.embedding_model:
            index_map = {
                (item["document_id"], item["chunk_index"]): item["embedding"]
                for item in serialized.get("chunks", [])
            }
            for chunk in self._chunks:
                chunk.embedding = index_map.get((chunk.document_id, chunk.chunk_index))
            if all(chunk.embedding for chunk in self._chunks):
                return

        for chunk in self._chunks:
            chunk.embedding = self._embed(chunk.content)
        payload = {
            "embedding_model": self.embedding_model,
            "chunks": [
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "source_path": chunk.source_path,
                    "embedding": chunk.embedding,
                }
                for chunk in self._chunks
            ],
        }
        self.index_path.write_text(json.dumps(payload, indent=2))

    def _load_existing_index(self) -> Optional[dict]:
        if not self.index_path.exists():
            return None
        try:
            return json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return None

    def _embed(self, text: str) -> Optional[list[float]]:
        last_error: Optional[Exception] = None
        for url in self._embed_urls:
            if url.endswith("/api/embeddings"):
                payload = {"model": self.embedding_model, "prompt": text[:8000]}
            elif url.endswith("/v1/embeddings"):
                payload = {"model": self.embedding_model, "input": text[:8000]}
            else:
                payload = {"model": self.embedding_model, "input": text[:8000]}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    logger.warning("Embedding endpoint %s returned 404; trying fallback.", url)
                    continue
                logger.warning("Embedding request failed: %s", exc)
                return None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                logger.warning("Embedding request failed: %s", exc)
                return None
            embeddings = data.get("embeddings") or []
            if embeddings and isinstance(embeddings[0], list):
                return [float(v) for v in embeddings[0]]
            openai_data = data.get("data") or []
            if openai_data and isinstance(openai_data[0], dict) and openai_data[0].get("embedding"):
                return [float(v) for v in openai_data[0]["embedding"]]
            embedding = data.get("embedding")
            if embedding:
                return [float(v) for v in embedding]
        if last_error is not None:
            logger.warning("Embedding request failed after fallbacks: %s", last_error)
        return None

    def _chunk_document(self, path: Path, text: str) -> list[_Chunk]:
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            return []
        chunks: list[_Chunk] = []
        start = 0
        index = 0
        while start < len(clean_text):
            end = min(len(clean_text), start + self.chunk_size)
            chunk_text = clean_text[start:end]
            chunks.append(
                _Chunk(
                    document_id=path.stem,
                    title=path.name,
                    content=chunk_text,
                    chunk_index=index,
                    source_path=str(path),
                )
            )
            if end == len(clean_text):
                break
            start = max(end - self.chunk_overlap, start + 1)
            index += 1
        return chunks

    @staticmethod
    def _extract_text(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".json", ".yaml", ".yml"}:
            return path.read_text(errors="ignore")
        if suffix == ".pdf":
            if PdfReader is None:
                logger.warning("Skipping PDF extraction for %s because pypdf is unavailable.", path)
                return ""
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return ""

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_embed_url_candidates(configured_url: str) -> list[str]:
        parsed = urllib.parse.urlparse(configured_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path.rstrip("/")
        candidates: list[str] = []
        if configured_url:
            candidates.append(configured_url)
        if path.endswith("/api/embed"):
            candidates.extend([base + "/api/embeddings", base + "/v1/embeddings"])
        elif path.endswith("/api/embeddings"):
            candidates.extend([base + "/api/embed", base + "/v1/embeddings"])
        elif path.endswith("/v1/embeddings"):
            candidates.extend([base + "/api/embed", base + "/api/embeddings"])
        else:
            candidates.extend([base + "/api/embed", base + "/api/embeddings", base + "/v1/embeddings"])
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped
