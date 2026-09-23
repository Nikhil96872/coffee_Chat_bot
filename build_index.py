"""
Chunk the OCR'd pages, embed them, and load them into Qdrant.

Two vectors are stored per chunk:

  dense  - BGE sentence embeddings, which match on meaning ("what rots coffee
           leaves in the rains?" finds text about humidity and monsoon disease)
  bm25   - sparse keyword weights, which match exact domain tokens that dense
           models blur together (Hemileia vastatrix, S.795, cultivar codes)

Searching both and fusing the rankings retrieves noticeably more than either
alone on a technical corpus like this one.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from chunking import chunk_all

ROOT = Path(__file__).parent
PAGES_FILE = ROOT / "ocr_text" / "pages.jsonl"
CHUNKS_FILE = ROOT / "ocr_text" / "chunks.jsonl"
PAPERS_FILE = ROOT / "ocr_text" / "papers.jsonl"
QDRANT_PATH = ROOT / "qdrant_data"

COLLECTION = "coffee"
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 768
BATCH = 128


def load_pages() -> list[dict]:
    if not PAGES_FILE.exists():
        raise SystemExit(f"{PAGES_FILE} not found - run run_ocr.py first")
    records = []
    with PAGES_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    started = time.perf_counter()

    pages = load_pages()
    print(f"loaded {len(pages)} OCR'd pages")

    chunks, papers = chunk_all(pages)
    print(f"built {len(chunks)} chunks "
          f"(avg {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars) "
          f"from {len(papers)} research papers and the handbook")

    with CHUNKS_FILE.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_payload(), ensure_ascii=False) + "\n")
    # The full papers stay out of Qdrant: only chunks are searched, and the
    # retriever looks a hit's paper up here by the paper_id in its payload.
    with PAPERS_FILE.open("w", encoding="utf-8") as fh:
        for p in papers:
            fh.write(json.dumps(p.to_record(), ensure_ascii=False) + "\n")

    print(f"loading models ({DENSE_MODEL}, {SPARSE_MODEL}) - first run downloads them")
    dense_model = TextEmbedding(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    # A fresh build every time: the collection is cheap to rebuild and this
    # avoids stale chunks lingering when the chunker or OCR changes.
    client = QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            # IDF weighting is applied by Qdrant at query time, which is what
            # makes this a real BM25 rather than raw term frequencies.
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

    texts = [c.for_embedding() for c in chunks]
    embedded = 0
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        batch_texts = texts[start:start + BATCH]

        dense = list(dense_model.embed(batch_texts))
        sparse = list(sparse_model.embed(batch_texts))

        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        "dense": dv.tolist(),
                        "bm25": models.SparseVector(
                            indices=sv.indices.tolist(), values=sv.values.tolist(),
                        ),
                    },
                    payload=chunk.to_payload(),
                )
                for chunk, dv, sv in zip(batch, dense, sparse)
            ],
        )
        embedded += len(batch)
        print(f"  indexed {embedded}/{len(chunks)}", flush=True)

    info = client.get_collection(COLLECTION)
    print(f"\ncollection '{COLLECTION}': {info.points_count} points")
    print(f"-> {QDRANT_PATH}")
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
