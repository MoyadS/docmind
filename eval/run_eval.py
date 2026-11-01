import json
import sys
from pathlib import Path

from docmind.config import settings
from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import load_file
from docmind.retrieval.embedder import Embedder
from docmind.retrieval.hybrid import HybridStore
from docmind.retrieval.store import VectorStore

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
K = 3

vector_store = VectorStore(Embedder())
hybrid_store = HybridStore(vector_store)
pages = load_file(ROOT / "corpus.txt")
hybrid_store.add("eval", chunk_pages(pages, settings.chunk_size, settings.chunk_overlap))

questions_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "questions.json"
questions = json.loads(questions_file.read_text(encoding="utf-8"))


def evaluate(name, store):
    print(f"\n=== {name} ===")
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
    print(f"Hit@1: {hit1:.0%}   Hit@{K}: {hit3:.0%}   MRR: {mrr:.2f}")


evaluate("Vector only", vector_store)
evaluate("Hybrid (vector + BM25)", hybrid_store)