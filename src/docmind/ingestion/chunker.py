import re
from dataclasses import dataclass

from docmind.ingestion.loader import Page


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    index: int
    text: str


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return pieces


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", page.text)]
        paragraphs = [p for p in paragraphs if p]

        texts: list[str] = []
        buffer = ""
        for para in paragraphs:
            if len(para) > size:
                if buffer:
                    texts.append(buffer)
                    buffer = ""
                texts.extend(_split_long(para, size, overlap))
            elif buffer and len(buffer) + 1 + len(para) > size:
                texts.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer} {para}".strip()
        if buffer:
            texts.append(buffer)

        for text in texts:
            chunks.append(Chunk(page.source, page.page, len(chunks), text))
    return chunks