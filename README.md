# DocMind

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

## Roadmap

- [ ] Retrieval evaluation (hit rate and MRR on a labeled question set)
- [ ] Hybrid search (BM25 + vectors) and reranking
- [ ] Stronger multilingual embedding model
- [ ] Written answers with citations using a free-tier LLM
- [ ] Docker image, CI and public demo