import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from docmind.config import settings
from docmind.ingestion.chunker import chunk_pages
from docmind.ingestion.loader import load_file
from docmind.retrieval.embedder import Embedder
from docmind.retrieval.store import VectorStore

app = FastAPI(title="DocMind")
store = VectorStore(Embedder())
ALLOWED_TYPES = {".pdf", ".txt", ".md"}
STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    session_id: str
    question: str
    k: int = 4


@app.post("/upload")
def upload(file: UploadFile, session_id: str | None = None):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED_TYPES:
        raise HTTPException(400, "Only PDF, TXT and MD files are supported")

    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        pages = load_file(path)

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "No readable text found in the file")

    session_id = session_id or str(uuid.uuid4())
    store.add(session_id, chunks)
    return {"session_id": session_id, "chunks": len(chunks)}


@app.post("/search")
def search(body: SearchRequest):
    hits = store.search(body.session_id, body.question, k=body.k)
    return {
        "results": [
            {
                "score": round(hit.score, 3),
                "source": hit.chunk.source,
                "page": hit.chunk.page,
                "text": hit.chunk.text,
            }
            for hit in hits
        ]
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")