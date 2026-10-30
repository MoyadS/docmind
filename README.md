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

| Baseline (multilingual MiniLM, 400-char chunks) | 86% | 100% | 0.92 |

Both misses are Arabic questions, so the next step is a stronger multilingual embedding model.
The test set is small, so treat these numbers as a regression baseline rather than a benchmark.

```bash
uv run python eval/run_eval.py
```

## Roadmap

- [ ] Retrieval evaluation (hit rate and MRR on a labeled question set)
- [ ] Hybrid search (BM25 + vectors) and reranking
- [ ] Stronger multilingual embedding model
- [ ] Written answers with citations using a free-tier LLM
- [ ] Docker image, CI and public demo