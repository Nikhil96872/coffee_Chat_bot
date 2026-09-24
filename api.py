"""
FastAPI backend for the coffee document assistant.

The server owns the Qdrant client and the embedding models for its whole lifetime
and loads them once at startup, which is both faster per request and the reason
this design avoids the embedded-Qdrant single-process limit: exactly one process
ever touches the storage, and every browser tab talks to it over HTTP.

Answers stream back as Server-Sent Events so text appears as it is generated
rather than after a five-second wait:

    event: sources   the retrieved passages, sent first so the UI can show them
    event: token     one chunk of the answer
    event: done      generation finished
    event: error     something failed mid-stream
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chunking import PAPER_SOURCES
from rag import MODEL, answer_stream, get_client
from retrieval import COLLECTION, Retriever

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_MESSAGES = 12

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once, before the first request arrives."""
    print("loading retrieval models...", flush=True)
    try:
        retriever = Retriever()
    except RuntimeError as exc:
        if "already accessed by another instance" not in str(exc):
            raise
        raise SystemExit(
            "\n  Another program is already using qdrant_data.\n"
            "  Qdrant's embedded storage allows one process at a time.\n\n"
            "  Close the other terminal running api.py or rag.py, then try again.\n"
            "  To find it:  Get-Process python | Format-Table Id, StartTime\n"
        ) from None
    retriever.search("coffee")          # force the lazy models to load now
    state["retriever"] = retriever
    state["client"] = get_client()
    print(f"ready - answering with {MODEL}", flush=True)
    yield
    retriever.close()


app = FastAPI(title="Coffee Documents Assistant", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/health")
async def health() -> dict:
    retriever = state.get("retriever")
    points = 0
    if retriever is not None:
        points = retriever.client.get_collection(COLLECTION).points_count
    return {"status": "ok", "model": MODEL, "indexed_chunks": points}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    retriever: Retriever | None = state.get("retriever")
    client = state.get("client")
    if retriever is None or client is None:
        raise HTTPException(status_code=503, detail="Server is still starting up")

    history = [m.model_dump() for m in request.history[-MAX_HISTORY_MESSAGES:]]

    async def events() -> AsyncIterator[str]:
        try:
            # Retrieval and the Groq call are blocking, so run them off the event
            # loop to keep the server responsive to other requests.
            mode, hits, token_iter = await asyncio.to_thread(
                answer_stream, client, retriever, request.question, history,
                request.top_k,
            )
            yield _sse("sources", {
                # "chat": answered from the conversation, so the UI shows the
                # previous answer's sources, which its citations still refer to.
                "mode": mode,
                "sources": [
                    {
                        "n": i,
                        "citation": h.citation,
                        "source": h.source,
                        "document": ("Research papers" if h.source in PAPER_SOURCES
                                     else "Handbook"),
                        "page_start": h.page_range[0],
                        "page_end": h.page_range[1],
                        "heading": h.heading,
                        "text": h.text,
                        "score": round(h.rerank_score, 2),
                        "paper": h.paper and {
                            key: h.paper[key]
                            for key in ("number", "title", "citation", "text")
                        },
                    }
                    for i, h in enumerate(hits, 1)
                ]
            })

            iterator = iter(token_iter)
            sentinel = object()
            while True:
                piece = await asyncio.to_thread(next, iterator, sentinel)
                if piece is sentinel:
                    break
                yield _sse("token", {"t": piece})

            yield _sse("done", {})
        except Exception as exc:                       # surface it in the UI
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/pdf/{name}")
async def pdf(name: str) -> FileResponse:
    """Serve a source PDF, so a citation can open it at its page (#page=N)."""
    allowed = {p.name: p for p in ROOT.glob("*.pdf")}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="No such document")
    return FileResponse(allowed[name], media_type="application/pdf")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
