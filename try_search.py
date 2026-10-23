from pathlib import Path

from docmind.config import settings
from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import load_file
from docmind.retrieval.embedder import Embedder
from docmind.retrieval.store import VectorStore

store = VectorStore(Embedder())
pages = load_file(Path("sample.txt"))
chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
store.add("demo", chunks)

while True:
    question = input("\nسؤالك (أو exit): ")
    if question == "exit":
        break
    for hit in store.search("demo", question, k=3):
        print(f"[{hit.score:.2f}] {hit.chunk.source} p.{hit.chunk.page}: {hit.chunk.text[:150]}...")