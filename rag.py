"""
Answer generation: retrieve passages, then have the LLM answer strictly from them.

The prompt is deliberately strict about grounding. The whole point of a RAG bot
over these documents is that it reports what the Coffee Board actually published,
so an answer drawn from the model's own training would be a failure even if it
happened to be correct.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq

from retrieval import Hit, Retriever, format_context

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# Groq's catalogue changes over time; run `client.models.list()` to see what the
# account can currently reach. gpt-oss-120b is the strongest general model there.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """\
You answer questions about coffee cultivation using only excerpts from two Indian \
Coffee Board documents: a grower's handbook and a compendium of research abstracts.

Rules:
1. Answer ONLY from the numbered passages given to you. Never add facts from your \
own knowledge, even if you are confident they are correct.
2. Cite the passage numbers you used in square brackets after the relevant claim, \
like [1] or [2][4].
3. If the passages do not contain the answer, say "I couldn't find this in the \
documents." and, if useful, name what the documents do cover nearby. Do not guess.
4. The passages come from OCR of scanned pages, so occasional characters are wrong \
(for example "nemotode" for "nematode"). Read through obvious scanning errors, but \
never invent a fact to fill a gap.
5. Prefer the specific over the general: give the actual varieties, dosages, \
temperatures, percentages and conditions as the documents state them.
6. Be concise. Use short paragraphs, or a list when the source material is a list.
7. When a passage is a research paper, attribute its findings to the paper's \
authors and year as given in its header, e.g. "Muthappa and Nataraj (1979) found \
that ... [2]".
"""

CONDENSE_PROMPT = """\
Rewrite the user's latest question into a standalone search query, resolving any \
pronouns or references to earlier turns. Reply with the query only, nothing else.

Conversation so far:
{history}

Latest question: {question}
Standalone query:"""


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is not set.\n"
            "Create a file named .env next to this script containing:\n"
            "    GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com"
        )
    return Groq(api_key=key)


def condense_question(client: Groq, history: list[dict], question: str) -> str:
    """Turn a follow-up like "what about robusta?" into a searchable question.

    Retrieval sees one query with no memory, so an unresolved follow-up would
    search for the wrong thing entirely.
    """
    if not history:
        return question
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in history[-MAX_HISTORY_TURNS:]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": CONDENSE_PROMPT.format(history=transcript, question=question),
        }],
        temperature=0.0,
        max_tokens=120,
    )
    condensed = (response.choices[0].message.content or "").strip()
    return condensed or question


def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY_TURNS:]
    messages.append({
        "role": "user",
        "content": (
            f"Passages:\n\n{format_context(hits)}\n\n"
            f"Question: {question}"
        ),
    })
    return messages


def answer_stream(
    client: Groq,
    retriever: Retriever,
    question: str,
    history: list[dict] | None = None,
    top_k: int | None = None,
) -> tuple[list[Hit], Iterator[str]]:
    """Retrieve, then stream the grounded answer.

    Returns the hits immediately so a UI can show sources while the text streams.
    """
    history = history or []
    search_query = condense_question(client, history, question)
    hits = retriever.search(search_query, top_k=top_k)

    if not hits:
        def empty() -> Iterator[str]:
            yield "I couldn't find this in the documents."
        return [], empty()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(question, hits, history),
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    def tokens() -> Iterator[str]:
        for chunk in stream:
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    return hits, tokens()


def answer(client: Groq, retriever: Retriever, question: str,
           history: list[dict] | None = None) -> tuple[list[Hit], str]:
    hits, stream = answer_stream(client, retriever, question, history)
    return hits, "".join(stream)


if __name__ == "__main__":
    import sys

    # The Windows console defaults to cp1252, which cannot encode characters the
    # model routinely emits (narrow no-break spaces, dashes, quotes).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    retriever = Retriever()
    client = get_client()
    history: list[dict] = []

    print("Coffee RAG. Ask a question, or Ctrl-C to quit.\n")
    try:
        while True:
            question = input("you > ").strip()
            if not question:
                continue
            hits, stream = answer_stream(client, retriever, question, history)
            print("\nbot > ", end="", flush=True)
            parts = []
            for piece in stream:
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            reply = "".join(parts)
            print("\n\nsources:")
            for i, hit in enumerate(hits, 1):
                label = hit.heading[:60]
                if hit.paper:
                    label = f"paper {hit.paper['number']}: {hit.paper['title'][:50]}"
                    if hit.paper["citation"]:
                        label += f" ({hit.paper['citation']})"
                print(f"  [{i}] {hit.citation}  (rerank {hit.rerank_score:+.2f})"
                      f"  {label}")
            print()
            history += [{"role": "user", "content": question},
                        {"role": "assistant", "content": reply}]
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
