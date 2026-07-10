"""Blueprint and floor-plan asset loading for Tier 2 reasoning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from reasoning.models import BlueprintContext

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional until deps are installed
    PdfReader = None

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class BlueprintStore:
    """Loads uploaded blueprint assets and exposes them as prompt context."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.directory = Path(config.get("directory", "data/blueprints"))
        self.render_directory = Path(config.get("render_directory", "data/blueprints/rendered"))
        self.max_pdf_pages = int(config.get("max_pdf_pages", 2))
        self._contexts: list[BlueprintContext] = []
        self._loaded = False

    def sync(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.render_directory.mkdir(parents=True, exist_ok=True)
        contexts: list[BlueprintContext] = []
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                contexts.append(
                    BlueprintContext(
                        source_path=str(path),
                        asset_type="image",
                        summary=f"Blueprint image asset: {path.name}",
                        image_paths=[str(path)],
                    )
                )
            elif suffix == ".pdf":
                extracted = self._extract_pdf_text(path)
                image_paths = self._render_pdf_pages(path)
                contexts.append(
                    BlueprintContext(
                        source_path=str(path),
                        asset_type="pdf",
                        summary=f"Blueprint PDF asset: {path.name}",
                        extracted_text=extracted[:8000],
                        image_paths=image_paths,
                    )
                )
        self._contexts = contexts
        self._loaded = True

    def load_context(self) -> list[BlueprintContext]:
        if not self._loaded:
            self.sync()
        return self._contexts

    def prompt_context(self) -> list[dict]:
        return [context.to_dict() for context in self.load_context()]

    def image_paths(self, limit: int = 2) -> list[str]:
        paths: list[str] = []
        for context in self.load_context():
            for image_path in context.image_paths:
                if len(paths) >= limit:
                    return paths
                paths.append(image_path)
        return paths

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        if PdfReader is None:
            logger.warning("Skipping blueprint PDF text extraction for %s because pypdf is unavailable.", path)
            return ""
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            logger.warning("Failed to parse blueprint PDF %s: %s", path, exc)
            return ""
        return "\n".join(page.extract_text() or "" for page in reader.pages[:2])

    def _render_pdf_pages(self, path: Path) -> list[str]:
        if fitz is None:
            return []
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.warning("Failed to render blueprint PDF %s: %s", path, exc)
            return []
        rendered: list[str] = []
        for page_index in range(min(self.max_pdf_pages, len(doc))):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            output_path = self.render_directory / f"{path.stem}_page_{page_index + 1}.png"
            pix.save(str(output_path))
            rendered.append(str(output_path))
        return rendered
