"""Smoke-test retrieval quality without involving the LLM.

Checks that hybrid search finds the right pages for questions whose answers we
already know are in the corpus, including ones phrased nothing like the source
text (which tests the dense side) and ones hinging on an exact term (the BM25
side).
"""

from __future__ import annotations

import time

from retrieval import Retriever

QUESTIONS = [
    "what causes coffee leaves to rot during the monsoon?",
    "how long should coffee seeds be soaked to speed up germination?",
    "what is the rainfall variability in the Chikkamagaluru growing zone?",
    "Hemileia vastatrix",
    "how should cherry coffee be dried in the sun?",
    "effect of shade trees on coffee yield",
    "what is the coffee bean beetle and how is it controlled?",
    "S.795 arabica variety characteristics",
    "effluent treatment for wet processing of coffee",
    "transpiration loss from coffee leaves",
]


def main() -> None:
    retriever = Retriever(top_k=3)
    retriever.search("warm up")          # load models before timing

    for question in QUESTIONS:
        started = time.perf_counter()
        hits = retriever.search(question)
        elapsed = (time.perf_counter() - started) * 1000
        print("=" * 78)
        print(f"Q: {question}    ({elapsed:.0f} ms)")
        for i, hit in enumerate(hits, 1):
            snippet = " ".join(hit.text.split())[:150]
            print(f"  [{i}] {hit.citation:<32} rerank {hit.rerank_score:+6.2f}")
            print(f"      {hit.heading[:70]}")
            print(f"      {snippet}...")


if __name__ == "__main__":
    main()
