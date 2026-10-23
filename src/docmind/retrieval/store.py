import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from docmind.ingestion.chunker import Chunk


@dataclass(frozen=True)
class Hit:
    score: float
    chunk: Chunk


class VectorStore:
    """Vector store with one isolated collection per visitor session."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._client = QdrantClient(":memory:")

    def add(self, session_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        if not self._client.collection_exists(session_id):
            self._client.create_collection(
                session_id,
                vectors_config=VectorParams(size=self._embedder.dim, distance=Distance.COSINE),
            )
        vectors = self._embedder.embed([c.text for c in chunks])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"source": c.source, "page": c.page, "index": c.index, "text": c.text},
            )
            for c, vector in zip(chunks, vectors)
        ]
        self._client.upsert(session_id, points=points)

    def search(self, session_id: str, query: str, k: int = 5) -> list[Hit]:
        if not self._client.collection_exists(session_id):
            return []
        vector = self._embedder.embed([query])[0]
        result = self._client.query_points(session_id, query=vector, limit=k)
        return [Hit(point.score, Chunk(**point.payload)) for point in result.points]

    def delete_session(self, session_id: str) -> None:
        if self._client.collection_exists(session_id):
            self._client.delete_collection(session_id)