from fastembed import TextEmbedding

from docmind.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self._model = TextEmbedding(model_name or settings.embedding_model)
        self.dim = len(self.embed(["dim probe"])[0])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]