from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        return [
            Page(path.name, i + 1, (p.extract_text() or "").strip())
            for i, p in enumerate(reader.pages)
        ]
    if suffix in {".txt", ".md"}:
        return [Page(path.name, 1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported file type: {suffix}")