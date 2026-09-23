"""
Hybrid retrieval over the Qdrant index.

Two stages, for two different reasons:

  search  - dense and BM25 are queried separately and their rankings fused with
            Reciprocal Rank Fusion. Fusing ranks rather than scores means the two
            systems' incomparable score scales never have to be reconciled.

  rerank  - a cross-encoder re-scores the shortlist. It reads the question and a
            passage together, so it judges relevance far better than the vectors
            can - but it cannot be precomputed, so it only ever sees the handful
            of candidates that survived the first stage.

A hit from the research compendium carries the paper it belongs to. A paper
matched through several of its chunks is returned once, and when it is short
enough the model reads the whole abstract rather than the one chunk that matched.
"""

from __future__ import annotations

import atexit
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).parent
QDRANT_PATH = ROOT / "qdrant_data"
PAPERS_FILE = ROOT / "ocr_text" / "papers.jsonl"

COLLECTION = "coffee"
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

CANDIDATES = 40      # per retriever, before fusion
SHORTLIST = 20       # fused, handed to the cross-encoder
TOP_K = 5            # passed to the LLM
# Papers up to this long go to the LLM whole (about 93% of them); a longer one
# is represented by its matching chunk, to keep the prompt within token limits.
PAPER_CONTEXT_CHARS = 2500
# The handbook gives practice, the compendium gives trial results, and for many
# questions both have something to say - but whichever has more on the topic
# tends to fill every slot. A document's best passage is guaranteed a place if
# it scores at least this; the cross-encoder's scores are logits, and below
# about -2 its matches stop being about the question.
MIN_SOURCE_SCORE = -2.0

# BGE was trained with an instruction prefix on the query side only. Using it
# lifts retrieval quality measurably; passages are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Hit:
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    heading: str
    text: str
    fusion_score: float
    rerank_score: float = 0.0
    paper: dict | None = None         # a record from papers.jsonl

    @property
    def page_range(self) -> tuple[int, int]:
        if self.paper:
            return self.paper["page_start"], self.paper["page_end"]
        return self.page_start, self.page_end

    @property
    def citation(self) -> str:
        start, end = self.page_range
        pages = f"p{start}" if start == end else f"pp{start}-{end}"
        return f"{self.source} {pages}"

    @property
    def passage(self) -> str:
        """What the model reads: the whole paper when it is short enough."""
        if self.paper and len(self.paper["text"]) <= PAPER_CONTEXT_CHARS:
            return self.paper["text"]
        return self.text


class Retriever:
    """Loads the models lazily so importing this module stays cheap."""

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        if not QDRANT_PATH.exists():
            raise SystemExit(f"{QDRANT_PATH} not found - run build_index.py first")
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.papers = _load_papers()
        # Qdrant's local client releases its file lock from __del__, which on
        # Windows can run after msvcrt has already been torn down and prints a
        # spurious traceback. Closing at exit instead keeps shutdown clean.
        atexit.register(self.close)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    @cached_property
    def dense(self) -> TextEmbedding:
        return TextEmbedding(DENSE_MODEL)

    @cached_property
    def sparse(self) -> SparseTextEmbedding:
        return SparseTextEmbedding(SPARSE_MODEL)

    @cached_property
    def reranker(self) -> TextCrossEncoder:
        return TextCrossEncoder(RERANK_MODEL)

    def search(self, question: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.top_k

        dense_vec = next(iter(self.dense.query_embed(QUERY_PREFIX + question)))
        sparse_vec = next(iter(self.sparse.query_embed(question)))

        response = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=dense_vec.tolist(), using="dense", limit=CANDIDATES,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using="bm25", limit=CANDIDATES,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=SHORTLIST,
            with_payload=True,
        )

        hits = [
            Hit(
                chunk_id=p.payload["chunk_id"],
                source=p.payload["source"],
                page_start=p.payload["page_start"],
                page_end=p.payload["page_end"],
                heading=p.payload["heading"],
                text=p.payload["text"],
                fusion_score=p.score,
                paper=self.papers.get(p.payload.get("paper_id")),
            )
            for p in response.points
        ]
        if not hits:
            return []

        scores = list(self.reranker.rerank(question, [h.text for h in hits]))
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        hits.sort(key=lambda h: h.rerank_score, reverse=True)

        # Several chunks of one paper should surface as that paper, once, at the
        # rank of its best chunk - leaving room in top_k for other papers.
        seen: set[tuple[str, int]] = set()
        unique: list[Hit] = []
        for hit in hits:
            key = ("paper", hit.paper["paper_id"]) if hit.paper else ("chunk", hit.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(hit)
        return _include_every_source(unique, top_k)


def _include_every_source(ranked: list[Hit], top_k: int) -> list[Hit]:
    """Take the top_k hits, making room for each document's best relevant hit.

    A missing document's best hit replaces the weakest hit from a document that
    has more than one, so no document is pushed out in the process.
    """
    top = ranked[:top_k]
    for hit in ranked[top_k:]:
        if hit.rerank_score < MIN_SOURCE_SCORE:
            break                                  # ranked, so the rest are worse
        if any(h.source == hit.source for h in top):
            continue
        counts: dict[str, int] = {}
        for h in top:
            counts[h.source] = counts.get(h.source, 0) + 1
        weakest = next((h for h in reversed(top) if counts[h.source] > 1), None)
        if weakest is None:
            break
        top.remove(weakest)
        top.append(hit)
    return top


def _load_papers() -> dict[int, dict]:
    if not PAPERS_FILE.exists():         # an index built before papers existed
        return {}
    with PAPERS_FILE.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return {r["paper_id"]: r for r in records}


def format_context(hits: list[Hit]) -> str:
    """Number the passages so the model can cite them by index."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        header = f"[{i}] {hit.citation}"
        if hit.paper:
            header += f" - Research paper {hit.paper['number']}: {hit.paper['title']}"
            if hit.paper["citation"]:
                header += f" ({hit.paper['citation']})"
        elif hit.heading:
            header += f" - {hit.heading}"
        blocks.append(f"{header}\n{hit.passage}")
    return "\n\n".join(blocks)
