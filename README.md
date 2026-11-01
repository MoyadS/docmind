# DocMind

![CI](https://github.com/MoyadS/docmind/actions/workflows/ci.yml/badge.svg)

Ask questions about your own documents in **Arabic or English**. Upload a PDF, TXT or Markdown file and get the most relevant passages back, ranked by meaning, with the source file and page for each result.

## Features

- **Semantic search**, not keyword matching: "how do I make a good coffee shot?" finds the paragraph about espresso even though the words differ.
- **Multilingual**: works with Arabic and English documents and questions.
- **Source references**: every result shows the file, page and similarity score.
- **Isolated sessions**: each visitor gets a private in-memory collection, and nothing is stored on disk.
- **Upload limits**: file type and size are validated.
- **Pluggable LLM layer**: `llm/` and `rag.py` can turn the retrieved passages into a written answer with any model.

## How it works

```
Document -> load -> chunk (paragraph-aware) -> embed -> Qdrant (per-session collection)
Question -> embed -> vector search -> top passages with sources
```

| Layer | Tech |
|---|---|
| API | FastAPI |
| Embeddings | fastembed (multilingual MiniLM, runs locally, no API key) |
| Vector store | Qdrant (in-memory) |
| PDF parsing | pypdf |
| Tests / lint | pytest, ruff |
| Packaging | uv |

## Quick start

```bash
git clone https://github.com/MoyadS/docmind.git
cd docmind
uv sync
uv run uvicorn docmind.api.main:app --reload
```

Open http://127.0.0.1:8000, upload a file (try `sample.txt`) and ask a question.
The first run downloads the embedding model (about 250 MB) once.

Interactive API docs are available at http://127.0.0.1:8000/docs.

## API

| Endpoint | Description |
|---|---|
| `POST /upload` | Upload a file, returns a `session_id` and the number of chunks |
| `POST /search` | Body: `{"session_id": "...", "question": "...", "k": 4}`, returns ranked passages |

## Tests

```bash
uv run pytest
```

## Project structure

```
src/docmind/
├── config.py        # settings (env-driven)
├── ingestion/       # file loading and chunking
├── retrieval/       # embeddings and vector store
├── llm/             # LLM client interface (optional answer generation)
├── rag.py           # retrieval + generation
└── api/             # FastAPI app and web UI
```

## Evaluation

Retrieval quality is measured with `eval/run_eval.py` on a small labeled set:
14 questions (English and Arabic) over a 15-passage corpus with deliberately similar topics
(Python / Java / JavaScript, Damascus / Aleppo, coffee / tea).

| Setup | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| MiniLM, 150-char chunks | 71% | 93% | 0.82 |
| MiniLM, 250-char chunks | 79% | 100% | 0.89 |
| MiniLM, 400-char chunks (default) | 86% | 100% | 0.92 |
| MiniLM, 600-char chunks | 86% | 100% | 0.92 |
| mpnet-base (1.1 GB), 400-char chunks | 86% | 100% | 0.92 |
| MiniLM + BM25 hybrid (RRF), 400-char chunks | 100% | 100% | 1.00 |

Hybrid search (vector similarity + BM25 keyword matching, merged with Reciprocal Rank Fusion) fixed both Arabic misses. The question set was written after seeing the baseline failures and shares many words with the passages, which favors keyword matching, so a held-out set of paraphrased questions is the next check.

A larger multilingual model (mpnet-base, 5x bigger) gave identical aggregate scores: it fixed one Arabic question and broke another. With 14 questions the difference is within noise, so the lighter MiniLM stays the default.
The test set is small, so treat these numbers as a regression baseline rather than a benchmark.

```bash
uv run python eval/run_eval.py
```
```

## Roadmap

- [x] Retrieval evaluation (Hit@k and MRR on a labeled question set)
- [x] Hybrid search (BM25 + vectors, Reciprocal Rank Fusion)
- [x] CI (lint and tests on every push)
- [x] Compared a larger embedding model (mpnet-base): no measurable gain, MiniLM stays the default
- [ ] Held-out evaluation set with paraphrased questions
- [ ] Reranking
- [ ] Written answers with citations using a free-tier LLM
- [ ] Docker image and public demo