"""
Turn OCR'd pages into retrieval chunks.

Chunks are built from sentences rather than fixed character windows, so a chunk
never begins or ends mid-sentence - a truncated sentence embeds poorly and reads
badly when shown as a citation.

Size is driven by the embedding model: BGE encodes at most 512 tokens and silently
truncates beyond that, so chunks are kept well inside that ceiling. Consecutive
chunks overlap by a couple of sentences so a fact stated across a chunk boundary
is still wholly present in one of them.

The research compendium is different in kind from the handbook: it is a run of
short, self-contained abstracts, each opening with a numbered title and closing
with its (Authors, Year) citation. There, every chunk is kept inside a single
paper and tagged with it, so a retrieved passage can be shown as the paper it
came from rather than as an anonymous run of text that may straddle two papers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from itertools import groupby

# ~4 chars per token, so 1400 chars is ~350 tokens - comfortably under BGE's 512
# limit even when the heading and source line are prepended for embedding.
MAX_CHARS = 1400
OVERLAP_CHARS = 250
MIN_CHARS = 120          # below this a chunk is noise, not content

# Documents made of research-paper abstracts, chunked paper by paper.
PAPER_SOURCES = {"merged_ocr.pdf"}


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    heading: str
    text: str
    paper_id: int | None = None

    def to_payload(self) -> dict:
        return asdict(self)

    def for_embedding(self) -> str:
        """Prefix the heading so the chunk carries its own topic context.

        Body text often refers to "the disease" or "this treatment" without
        naming it; the heading is frequently the only place the actual subject
        appears, so it belongs in the embedded string.
        """
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class Paper:
    """One research abstract: its numbered title, citation and full text."""

    paper_id: int
    source: str
    number: str                   # "3.2.3"
    title: str
    page_start: int
    page_end: int
    citation: str = ""            # "Muthappa and Nataraj, 1979"
    text: str = ""

    def to_record(self) -> dict:
        return asdict(self)


# "3.2.3. Control of brown eye spot" or "4.12. Bearing nature of Arabica". A
# two-level number must keep its trailing dot, or "15.5 kg" would open a paper.
_PAPER_HEADING = re.compile(
    r"^(\d{1,2}\.\d{1,2}\.\d{1,2})\.?\s+([A-Za-z].*)$"
    r"|^(\d{1,2}\.\d{1,2})\.\s+([A-Z].*)$"
)
# Some chapters number papers with a single level ("5. Embryo culture of ...",
# OCR'd as "4, Comparison" too). That also looks like a list item, so it only
# opens a paper once the previous one has closed with its citation.
_PAPER_HEADING_SHORT = re.compile(r"^(\d{1,3})[.,]\s+([A-Z][A-Za-z].{6,})$")
# "(Muthappa and Nataraj, 1979)" - the paper's closing citation.
_CITATION = re.compile(r"\(([A-Z][^()]{2,150}?),?\s+((?:1[89]|20)\d\d)[a-z]?\)")
# The compendium's running page banner, which is not a heading of anything.
_RUNNING_HEADER = re.compile(r"COMPENDIUM OF COFFEE RESEARCH|Abstracts of Research Papers", re.I)


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def _last_citation(text: str) -> str:
    """The last (Authors, Year) in a paper is its own citation."""
    found = _CITATION.findall(text)
    if not found:
        return ""
    authors, year = found[-1]
    return f"{' '.join(authors.split())}, {year}"


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
# "1.3. Analysis of rainfall" is a heading; "15.5 kg of parchment" is not. The
# trailing dot and the following capital are what separate a section number from
# an ordinary decimal at the start of a line.
_SECTION_HEADING = re.compile(r"^\d{1,2}\.\d{1,2}\.\s+[A-Z]")


def is_heading(line: str) -> bool:
    """Numbered section headings and short all-caps banners."""
    line = line.strip()
    if not 4 <= len(line) <= 90:
        return False
    if _SECTION_HEADING.match(line):
        return True

    letters = [c for c in line if c.isalpha()]
    if len(letters) < 8 or not all(c.isupper() for c in letters):
        return False
    # Reject all-caps OCR debris such as "HISTO]" or "OK OK OK OK" - a real
    # banner heading contains at least one substantial word.
    words = re.findall(r"[A-Za-z]+", line)
    return bool(words) and max(len(w) for w in words) >= 3


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if not para:
            continue
        for sentence in _SENTENCE_END.split(para):
            sentence = sentence.strip()
            if sentence:
                parts.append(sentence)
    return parts


def _hard_split(sentence: str) -> list[str]:
    """Break a sentence longer than one chunk on word boundaries."""
    words, out, cur = sentence.split(), [], ""
    for word in words:
        if cur and len(cur) + len(word) + 1 > MAX_CHARS:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_pages(
    pages: list[dict],
    start_id: int = 0,
    paper_start_id: int = 0,
) -> tuple[list[Chunk], list[Paper]]:
    """Chunk one document's pages, which must already be in page order.

    Chunks may span a page break: a sentence interrupted by a page boundary is
    still one thought, and `page_start`/`page_end` record the range so a citation
    can point at every page the text came from. In a paper collection they never
    span two papers, though, since that would attribute one paper's findings to
    another.
    """
    source = pages[0]["source"]
    by_paper = source in PAPER_SOURCES
    # (page_no, sentence, heading, paper_id)
    units: list[tuple[int, str, str, int | None]] = []
    papers: list[Paper] = []
    paper_lines: dict[int, list[str]] = {}
    current: Paper | None = None
    current_cited = False             # has the current paper reached its citation?
    heading = ""

    for page in pages:
        page_no = page["page_no"]
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]
        k = 0
        while k < len(lines):
            line = lines[k]
            k += 1
            if by_paper:
                if _RUNNING_HEADER.search(line):
                    continue
                m = _PAPER_HEADING.match(line)
                if not m and (current is None or current_cited):
                    m = _PAPER_HEADING_SHORT.match(line)
                if m:
                    number = next(g for g in m.groups()[0::2] if g)
                    title = next(g for g in m.groups()[1::2] if g)
                    if _is_all_caps(title):        # a category, "2.6. CHEMICAL CONTROL"
                        current, heading = None, line
                        continue
                    # A long title wraps; its tail is a short line in lower case.
                    while (k < len(lines) and len(lines[k]) <= 70
                           and lines[k][0].islower() and not title.endswith(".")):
                        title = f"{title} {lines[k]}"
                        k += 1
                    current = Paper(
                        paper_id=paper_start_id + len(papers), source=source,
                        number=number, title=title,
                        page_start=page_no, page_end=page_no,
                    )
                    papers.append(current)
                    paper_lines[current.paper_id] = []
                    current_cited = False
                    heading = f"{number} {title}"
                    continue
            if is_heading(line):
                # Inside a paper, an all-caps line is the running section banner
                # at the top of the page ("BROWN EYE SPOT"), not a new topic.
                if current is None:
                    heading = line
                continue
            paper_id = current.paper_id if current else None
            if current:
                current.page_end = page_no
                paper_lines[current.paper_id].append(line)
                # A citation can wrap: "(Raghuramulu," / "1989).".
                tail = " ".join(paper_lines[current.paper_id][-2:])
                current_cited = current_cited or bool(_CITATION.search(tail))
            for sentence in _sentences(line):
                for piece in ([sentence] if len(sentence) <= MAX_CHARS
                              else _hard_split(sentence)):
                    units.append((page_no, piece, heading, paper_id))

    for paper in papers:
        paper.text = " ".join(" ".join(paper_lines[paper.paper_id]).split())
        paper.citation = _last_citation(paper.text)

    chunks: list[Chunk] = []
    for paper_id, group in groupby(units, key=lambda u: u[3]):
        chunks += _pack(list(group), source, start_id + len(chunks), paper_id,
                        keep_short=not chunks)
    return chunks, papers


def _pack(
    units: list[tuple[int, str, str, int | None]],
    source: str,
    start_id: int,
    paper_id: int | None,
    keep_short: bool,
) -> list[Chunk]:
    """Pack consecutive sentences into overlapping chunks of at most MAX_CHARS."""
    chunks: list[Chunk] = []
    i, next_id = 0, start_id
    while i < len(units):
        taken, size = [], 0
        j = i
        while j < len(units) and size + len(units[j][1]) + 1 <= MAX_CHARS:
            taken.append(units[j])
            size += len(units[j][1]) + 1
            j += 1
        if not taken:                           # single oversized unit
            taken, j = [units[i]], i + 1

        text = " ".join(u[1] for u in taken)
        if len(text) >= MIN_CHARS or (keep_short and not chunks):
            chunks.append(Chunk(
                chunk_id=next_id,
                source=source,
                page_start=min(u[0] for u in taken),
                page_end=max(u[0] for u in taken),
                heading=taken[0][2],
                text=text,
                paper_id=paper_id,
            ))
            next_id += 1

        if j >= len(units):
            break
        # Step back far enough to overlap the next chunk by ~OVERLAP_CHARS.
        back, overlap = j, 0
        while back > i + 1 and overlap < OVERLAP_CHARS:
            back -= 1
            overlap += len(units[back][1]) + 1
        i = back
    return chunks


def chunk_all(records: list[dict]) -> tuple[list[Chunk], list[Paper]]:
    """Chunk every document found in the OCR output."""
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("text", "").strip():
            by_source.setdefault(rec["source"], []).append(rec)

    chunks: list[Chunk] = []
    papers: list[Paper] = []
    for source in sorted(by_source):
        pages = sorted(by_source[source], key=lambda r: r["page_no"])
        new_chunks, new_papers = chunk_pages(
            pages, start_id=len(chunks), paper_start_id=len(papers),
        )
        chunks += new_chunks
        papers += new_papers
    # A heading misread in the scan can open a "paper" with no usable text.
    cited = {c.paper_id for c in chunks}
    return chunks, [p for p in papers if p.paper_id in cited]
