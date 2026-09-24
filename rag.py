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
MAX_HISTORY_TURNS = 10          # messages, i.e. the last five exchanges
# gpt-oss sometimes cites as 【1】 despite the prompt; the UI only links [1].
CITATION_BRACKETS = str.maketrans("【】", "[]")

SYSTEM_PROMPT = """\
You answer questions about coffee cultivation using only excerpts from two Indian \
Coffee Board documents: a grower's handbook and a compendium of research abstracts.

Rules:
1. Answer ONLY from the numbered passages given to you. Never add facts from your \
own knowledge, even if you are confident they are correct.
2. Cite the passage numbers you used in square brackets after the relevant claim, \
like [1] or [2][4]. Use only this form, never 【1†L1-L4】 or line references.
3. The passages were selected because they relate to the question, so always give \
the user something useful from them. If they answer the question, answer it. If \
they do not answer the specific point asked, open with one sentence in this form: \
"The documents do not specifically address <the point asked>, but they do contain \
related information that may help." Then present the most relevant related facts \
from the passages, with citations, and say briefly how each bears on the question. \
Include only passages with a genuine bearing on it; leave out ones that merely \
share a word, and do not draw conclusions the passages do not state. Keep it to \
the opening sentence and a few bullet points, with no table and no closing \
summary. Never reply "I couldn't find this in the documents." Do not guess or fill the gap \
from your own knowledge.
4. The passages come from OCR of scanned pages, so occasional characters are wrong \
(for example "nemotode" for "nematode"). Read through obvious scanning errors, but \
never invent a fact to fill a gap.
5. Prefer the specific over the general: give the actual varieties, dosages, \
temperatures, percentages and conditions as the documents state them.
6. Be concise. Use short paragraphs, or a list when the source material is a list.
7. When a passage is a research paper, attribute its findings to the paper's \
authors and year as given in its header, e.g. "Muthappa and Nataraj (1979) found \
that ... [2]".
8. Earlier turns of the conversation are context for follow-up questions. Follow \
any format the user asks for; use a markdown table for tables, schedules and \
calendar views.
9. If the message is a greeting, thanks or small talk rather than a question \
(e.g. "hi", "thank you"), ignore the passages and rules 2-3: reply in one or two \
warm sentences and offer to help with questions about coffee cultivation.
"""

# Follow-ups that only reshape or recall an earlier answer ("only the months",
# "what did you tell me?") are answered from the conversation. Sending them to
# retrieval would fetch unrelated passages, and the grounding rules would then
# make the model reply that the documents say nothing about it.
CHAT_PROMPT = """\
You are continuing a conversation about coffee cultivation. Your earlier answers \
in this conversation were written from excerpts of two Indian Coffee Board \
documents.

Answer the user's latest message using only what is already in the conversation:
1. Do not add facts that were not stated earlier in the conversation.
2. When you reuse a fact, keep its citation marker, such as [1], exactly as it \
appeared in your earlier answer. Never invent new citation numbers.
3. Follow any format the user asks for; use a markdown table for tables, \
schedules and calendar views.
4. If the message needs information that is not in the conversation, say so in \
one sentence and suggest asking it as a new question.
5. Be concise.
"""

# Questions the documents have nothing on (retrieval found no relevant passage)
# still get a courteous reply rather than a one-line refusal.
OFF_TOPIC_PROMPT = """\
You are the assistant for a coffee cultivation knowledge base built from two \
Indian Coffee Board documents: a grower's handbook and a compendium of research \
abstracts. It covers topics such as planting, varieties, shade, nutrition, \
irrigation, pests and diseases, harvesting and processing.

The user's message is outside what these documents cover. Reply in a friendly, \
professional tone in two short sentences:
1. State plainly that the topic is outside what this assistant covers, e.g. \
"Upcoming Hindi events are outside what I can help with - I answer questions \
about coffee cultivation from the Coffee Board documents."
2. Suggest two or three coffee topics they could ask about.
Do not answer the question itself or offer facts about it from your own knowledge. \
Start directly with the statement: no filler such as "Thank you for your \
question", "Thanks for reaching out", "I appreciate", "Unfortunately" or "Sorry".
If the message is a greeting, thanks or small talk, respond naturally and offer help \
with coffee questions."""

ROUTE_PROMPT = """\
You route the latest message in a chat about Indian Coffee Board documents on \
coffee cultivation. Reply with exactly one line, in one of these two forms:

CHAT
SEARCH: <standalone search query>

Choose CHAT when the message can be answered fully from the conversation so far \
without looking anything up: reformatting, shortening or translating an earlier \
answer ("list only the months", "show it as a table"), repeating or summarising \
what was already said ("what did you tell me?"), or explaining a point already made.

Choose SEARCH when the message needs information that is not already in the \
conversation, including follow-ups about a new aspect ("how is it controlled?", \
"what about robusta?"). Write the query so it stands alone, resolving pronouns and \
references to earlier turns. When unsure, choose SEARCH.

Conversation so far:
{history}

Latest message: {question}"""


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


def route_question(client: Groq, history: list[dict], question: str) -> tuple[str, str]:
    """Decide how to answer: ("search", standalone query) or ("chat", "").

    Retrieval sees one query with no memory, so a follow-up like "what about
    robusta?" is rewritten to stand alone before searching. A follow-up that
    only reshapes or recalls an earlier answer needs no search at all.
    """
    if not history:
        return "search", question
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:600]}" for m in history[-MAX_HISTORY_TURNS:]
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": ROUTE_PROMPT.format(history=transcript, question=question),
        }],
        temperature=0.0,
        # gpt-oss reasons before it replies, and the reasoning counts against
        # max_tokens: with a tight limit it can run out before writing the
        # one-line answer. A one-word decision needs little reasoning.
        max_tokens=1024,
        reasoning_effort="low",
    )
    reply = (response.choices[0].message.content or "").replace("\x00", "").strip()
    first = reply.splitlines()[0].strip() if reply else ""
    if first.upper().startswith("CHAT"):
        return "chat", ""
    if first.upper().startswith("SEARCH"):
        query = first.split(":", 1)[1].strip() if ":" in first else ""
        return "search", query or question
    return "search", question                  # unparseable: searching is safe


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
) -> tuple[str, list[Hit], Iterator[str]]:
    """Route, retrieve if needed, then stream the answer.

    Returns the mode ("search" or "chat") and the hits immediately, so a UI can
    show sources while the text streams. In chat mode there are no new hits:
    the answer reuses the previous answer's sources.
    """
    history = history or []
    mode, search_query = route_question(client, history, question)

    if mode == "chat":
        messages = [{"role": "system", "content": CHAT_PROMPT}]
        messages += history[-MAX_HISTORY_TURNS:]
        messages.append({"role": "user", "content": question})
        hits: list[Hit] = []
    else:
        hits = retriever.search(search_query, top_k=top_k)
        if hits:
            messages = build_messages(question, hits, history)
        else:
            # Nothing cleared the relevance floor: the question is off-topic
            # (e.g. movies), not merely unanswered by the documents.
            messages = [{"role": "system", "content": OFF_TOPIC_PROMPT},
                        {"role": "user", "content": question}]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    def tokens() -> Iterator[str]:
        for chunk in stream:
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece.translate(CITATION_BRACKETS)

    return mode, hits, tokens()


def answer(client: Groq, retriever: Retriever, question: str,
           history: list[dict] | None = None) -> tuple[str, list[Hit], str]:
    mode, hits, stream = answer_stream(client, retriever, question, history)
    return mode, hits, "".join(stream)


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
            mode, hits, stream = answer_stream(client, retriever, question, history)
            print("\nbot > ", end="", flush=True)
            parts = []
            for piece in stream:
                parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            reply = "".join(parts)
            print("\n\n(answered from the conversation, sources as above)" if mode == "chat"
                  else "\n\nsources:")
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
