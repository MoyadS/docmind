from docmind.ingestion.chunker import Chunk
from docmind.retrieval.store import VectorStore


class FakeEmbedder:
    dim = 2

    def embed(self, texts):
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]


def test_search_returns_best_match_and_isolates_sessions():
    store = VectorStore(FakeEmbedder())
    store.add("s1", [Chunk("a.txt", 1, 0, "cat food"), Chunk("b.txt", 1, 1, "stock market")])

    hits = store.search("s1", "cat", k=1)
    assert hits[0].chunk.source == "a.txt"

    assert store.search("other-session", "cat") == []