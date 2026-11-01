import math
import re
from collections import Counter

from docmind.ingestion.chunker import Chunk
from docmind.retrieval.store import Hit, VectorStore

_WORD = re.compile(r"\w+")
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
_ALEF = re.compile(r"[أإآ]")


def tokenize(text: str) -> list[str]:
    text = _ALEF.sub("ا", _DIACRITICS.sub("", text.lower()))
    tokens = []
    for token in _WORD.findall(text):
        if len(token) > 3 and token.startswith("ال"):
            token = token[2:]
        tokens.append(token)
    return tokens


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self._docs = docs
        self._k1 = k1
        self._b = b
        self._avg_len = (sum(len(d) for d in docs) / len(docs)) or 1.0
        self._tf = [Counter(d) for d in docs]
        doc_freq = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in doc_freq.items()}

    def scores(self, query: list[str]) -> list[float]:
        result = []
        for tf, doc in zip(self._tf, self._docs):
            norm = 1 - self._b + self._b * len(doc) / self._avg_len
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self._k1 + 1) / (freq + self._k1 * norm)
            result.append(score)
        return result


class HybridStore:
    """Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion."""

    def __init__(self, vector_store: VectorStore, rrf_k: int = 60) -> None:
        self._vectors = vector_store
        self._rrf_k = rrf_k
        self._chunks: dict[str, list[Chunk]] = {}
        self._bm25: dict[str, BM25] = {}

    def add(self, session_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._vectors.add(session_id, chunks)
        combined = self._chunks.get(session_id, []) + list(chunks)
        self._chunks[session_id] = combined
        self._bm25[session_id] = BM25([tokenize(c.text) for c in combined])

    def search(self, session_id: str, query: str, k: int = 5) -> list[Hit]:
        if session_id not in self._chunks:
            return []
        pool = max(k * 3, 10)
        chunks = self._chunks[session_id]
        keyword_scores = self._bm25[session_id].scores(tokenize(query))
        keyword_order = sorted(range(len(chunks)), key=lambda i: keyword_scores[i], reverse=True)

        fused: dict[Chunk, float] = {}
        vector_hits = self._vectors.search(session_id, query, k=pool)
        for rank, hit in enumerate(vector_hits, start=1):
            fused[hit.chunk] = fused.get(hit.chunk, 0.0) + 1 / (self._rrf_k + rank)
        keyword_hits = [i for i in keyword_order[:pool] if keyword_scores[i] > 0]
        for rank, i in enumerate(keyword_hits, start=1):
            fused[chunks[i]] = fused.get(chunks[i], 0.0) + 1 / (self._rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(score, chunk) for chunk, score in ranked]

    def delete_session(self, session_id: str) -> None:
        self._vectors.delete_session(session_id)
        self._chunks.pop(session_id, None)
        self._bm25.pop(session_id, None)