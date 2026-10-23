from docmind.ingestion.chunker import Chunk
from docmind.rag import answer
from docmind.retrieval.store import VectorStore


class FakeEmbedder:
    dim = 2

    def embed(self, texts):
        return [[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts]


class FakeLLM:
    def __init__(self):
        self.last_user = ""

    def complete(self, system, user):
        self.last_user = user
        return "fake answer [1]"


def test_answer_passes_retrieved_context_and_returns_sources():
    store = VectorStore(FakeEmbedder())
    store.add("s1", [Chunk("a.txt", 3, 0, "cat food facts")])
    llm = FakeLLM()

    result = answer("cat?", "s1", store, llm)

    assert result.text == "fake answer [1]"
    assert "cat food facts" in llm.last_user
    assert result.sources[0].chunk.page == 3


def test_answer_without_documents_does_not_call_llm():
    llm = FakeLLM()
    result = answer("anything", "empty", VectorStore(FakeEmbedder()), llm)
    assert llm.last_user == ""
    assert result.sources == []