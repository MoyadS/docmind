import pytest

from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import Page


def test_chunks_overlap_and_keep_source():
    pages = [Page("a.txt", 1, "x" * 100)]
    chunks = chunk_pages(pages, size=40, overlap=10)
    assert len(chunks) == 3
    assert all(c.source == "a.txt" and c.page == 1 for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_pages([Page("a.txt", 1, "hi")], size=10, overlap=10)

def test_paragraphs_stay_separate_when_they_do_not_fit_together():
    text = ("a" * 300) + "\n\n" + ("b" * 300)
    chunks = chunk_pages([Page("a.txt", 1, text)], size=400, overlap=50)
    assert [c.text[0] for c in chunks] == ["a", "b"]