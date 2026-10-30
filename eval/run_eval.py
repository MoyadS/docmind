import json
from pathlib import Path

from docmind.config import settings
from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import load_file
from docmind.retrieval.embedder import Embedder
from docmind.retrieval.store import VectorStore

ROOT = Path(__file__).parent
K = 3

store = VectorStore(Embedder())
pages = load_file(ROOT / "corpus.txt")
store.add("eval", chunk_pages(pages, settings.chunk_size, settings.chunk_overlap))

questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
ranks = []
for item in questions:
    hits = store.search("eval", item["question"], k=K)
    rank = next(
        (i for i, h in enumerate(hits, start=1) if item["expected"] in h.chunk.text),
        None,
    )
    ranks.append(rank)
    mark = f"rank {rank}" if rank else "MISS"
    print(f"[{mark:>6}] {item['question']}")

total = len(ranks)
hit1 = sum(1 for r in ranks if r == 1) / total
hit3 = sum(1 for r in ranks if r) / total
mrr = sum(1 / r for r in ranks if r) / total

print(f"\nQuestions: {total}")
print(f"Hit@1: {hit1:.0%}   Hit@{K}: {hit3:.0%}   MRR: {mrr:.2f}")