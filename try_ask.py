from pathlib import Path

from docmind.config import settings
from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import load_file
from docmind.llm.client import ClaudeLLM
from docmind.rag import answer
from docmind.retrieval.embedder import Embedder
from docmind.retrieval.store import VectorStore

store = VectorStore(Embedder())
llm = ClaudeLLM()
chunks = chunk_pages(load_file(Path("sample.txt")), settings.chunk_size, settings.chunk_overlap)
store.add("demo", chunks)

while True:
    question = input("\nسؤالك (أو exit): ")
    if question == "exit":
        break
    result = answer(question, "demo", store, llm)
    print("\n" + result.text)
    print("\nالمصادر:")
    for i, hit in enumerate(result.sources, start=1):
        print(f"  [{i}] {hit.chunk.source} p.{hit.chunk.page} (score {hit.score:.2f})")