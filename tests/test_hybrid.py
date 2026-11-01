from docmind.ingestion.chunker import Chunk
from docmind.retrieval.hybrid import HybridStore, tokenize
from docmind.retrieval.store import VectorStore


class ConstantEmbedder:
    dim = 2

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_tokenize_normalizes_arabic():
    assert tokenize("الجامع الأموي") == ["جامع", "اموي"]


def test_keyword_match_wins_when_vectors_cannot_tell_chunks_apart():
    store = HybridStore(VectorStore(ConstantEmbedder()))
    store.add(
        "s1",
        [
            Chunk("a.txt", 1, 0, "apples and oranges"),
            Chunk("a.txt", 1, 1, "bananas and grapes"),
            Chunk("a.txt", 1, 2, "the zebra crossing"),
        ],
    )
    hits = store.search("s1", "zebra", k=1)
    assert hits[0].chunk.text == "the zebra crossing"


def test_unknown_session_returns_nothing():
    assert HybridStore(VectorStore(ConstantEmbedder())).search("nope", "x") == []