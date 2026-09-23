"""
OCR pipeline for the scanned, multi-column coffee PDFs.

The text layer already embedded in these files was produced by an OCR engine that
read straight across both columns, splicing unrelated sentences together mid-line:

    "An analysis has been carried out to study faiths on"
     |__ left column ____________________| |_ right col _|

That damage is not recoverable by post-processing, so this module re-OCRs from the
page images instead:

    render 300 DPI -> deskew -> Tesseract layout analysis -> reading-order text

Tesseract's automatic page segmentation (PSM 3) detects columns itself and returns
blocks in reading order, which handles headers, figures and irregular layouts far
more robustly than a hand-rolled projection. It does need a straight page, so
deskew comes first.

Because splicing is the specific failure we are trying to eliminate, every page is
scored for it afterwards (see `splice_score`). If a page still looks spliced and we
can find a confident column gutter, we re-OCR it by cutting the page at the gutter
and reading each side separately, then keep whichever result scores better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytesseract

# The winget package installs here and does not add itself to PATH.
_TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if _TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_TESSERACT_EXE)

RENDER_DPI = 300
INK_THRESHOLD = 160          # grey level below which a pixel counts as text

# Scans are hand-placed on a flatbed; a few degrees, occasionally more.
COARSE_SKEW_DEG = 8.0
COARSE_STEP_DEG = 0.5
FINE_STEP_DEG = 0.1

# A gutter must be this much emptier than the columns beside it to be believed.
GUTTER_MAX_RATIO = 0.15

PSM_AUTO = "--oem 1 --psm 3"          # automatic layout analysis
PSM_BLOCK = "--oem 1 --psm 6"         # one uniform block, for a single column


@dataclass
class PageResult:
    """One OCR'd page, with the geometry and quality signals we inferred."""

    source: str
    page_no: int                      # 1-indexed, matches the PDF viewer
    text: str
    skew_deg: float
    splice_score: int
    method: str                       # "auto" or "column-split"
    n_blocks: int = 0
    blocks: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ----------------------------------------------------------------- rendering


def render_page(doc: pymupdf.Document, index: int, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render one page to a greyscale numpy array."""
    pix = doc[index].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _ink_mask(grey: np.ndarray) -> np.ndarray:
    """1 where there is text, 0 where there is paper."""
    return (grey < INK_THRESHOLD).astype(np.uint8)


# ------------------------------------------------------------------- deskew


def _rotate(img: np.ndarray, angle: float, fill: int) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def _line_sharpness(ink: np.ndarray, angle: float) -> float:
    """How crisply separated the text lines are at this rotation."""
    rotated = _rotate(ink, angle, fill=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(((profile[1:] - profile[:-1]) ** 2).sum())


def estimate_skew(ink: np.ndarray) -> float:
    """Return the rotation in degrees that would straighten the text.

    Text lines are horizontal, so on a straight page the row-by-row ink profile
    alternates sharply between dense (a line of text) and empty (the gap between
    lines). Tilting smears those peaks together, so the angle that maximises the
    profile's sharpness is the angle that straightens the page.

    Searched coarsely over a wide range, then refined around the winner - which
    covers badly-placed scans without paying for a fine sweep of the whole range.
    Estimation runs on a quarter-scale copy, ample at this precision and ~16x
    cheaper.
    """
    h, w = ink.shape
    small = cv2.resize(ink, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

    coarse = np.arange(-COARSE_SKEW_DEG, COARSE_SKEW_DEG + COARSE_STEP_DEG, COARSE_STEP_DEG)
    best = max(coarse, key=lambda a: _line_sharpness(small, float(a)))

    fine = np.arange(best - COARSE_STEP_DEG, best + COARSE_STEP_DEG + FINE_STEP_DEG, FINE_STEP_DEG)
    best = max(fine, key=lambda a: _line_sharpness(small, float(a)))
    return round(float(best), 2)


# ------------------------------------------------------- splice detection


# Two numbered section headings on one line means two columns were read across,
# e.g. "1.3. of rainfall 1,5. Influence of rainfall 3". This corpus numbers its
# sections throughout, which makes it a precise signal for exactly our bug.
_SECTION_NO = re.compile(r"\b\d{1,2}\.\d{1,2}\.")


def splice_score(text: str) -> int:
    """Count lines that appear to contain text from two columns at once."""
    return sum(1 for line in text.split("\n") if len(_SECTION_NO.findall(line)) >= 2)


# --------------------------------------------------------- column detection


def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="same")


def find_gutter(ink: np.ndarray) -> int | None:
    """Locate the x of the vertical gap between two columns, or None.

    Only a gutter far emptier than the text either side of it is believed; this
    is used as a fallback, so a false positive is more costly than a miss.
    """
    h, w = ink.shape
    body = ink[int(h * 0.12):int(h * 0.95), :]
    profile = _smooth(body.sum(axis=0, dtype=np.float64))

    lo, hi = int(w * 0.40), int(w * 0.60)
    band = profile[lo:hi]
    if band.size == 0:
        return None

    gutter_x = lo + int(band.argmin())
    left = float(np.median(profile[int(w * 0.10):int(w * 0.35)]))
    right = float(np.median(profile[int(w * 0.65):int(w * 0.90)]))
    column_ink = (left + right) / 2.0
    if column_ink <= 0 or float(band.min()) / column_ink > GUTTER_MAX_RATIO:
        return None
    if ink[:, :gutter_x].sum() == 0 or ink[:, gutter_x:].sum() == 0:
        return None
    return gutter_x


# ---------------------------------------------------------------------- OCR


def _blocks_in_reading_order(grey: np.ndarray) -> list[str]:
    """OCR with layout analysis, rebuilding text in Tesseract's block order."""
    data = pytesseract.image_to_data(
        grey, lang="eng", config=PSM_AUTO, output_type=pytesseract.Output.DICT,
    )
    blocks: dict[int, dict[tuple[int, int], list[str]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        block = data["block_num"][i]
        line = (data["par_num"][i], data["line_num"][i])
        blocks.setdefault(block, {}).setdefault(line, []).append(word)

    out = []
    for block in sorted(blocks):
        lines = [" ".join(words) for _, words in sorted(blocks[block].items())]
        out.append("\n".join(lines))
    return out


def _ocr_plain(image: np.ndarray, config: str = PSM_BLOCK) -> str:
    if image.size == 0 or min(image.shape[:2]) < 10:
        return ""
    return pytesseract.image_to_string(image, lang="eng", config=config)


def _column_split_blocks(grey: np.ndarray, gutter_x: int) -> list[str]:
    """Cut the page at the gutter and read each side as its own column."""
    pad = 8
    return [
        _ocr_plain(grey[:, : gutter_x - pad]),
        _ocr_plain(grey[:, gutter_x + pad :]),
    ]


def ocr_page(
    doc: pymupdf.Document,
    index: int,
    source: str,
    dpi: int = RENDER_DPI,
) -> PageResult:
    """Deskew and OCR a single page, retrying differently if it looks spliced."""
    grey = render_page(doc, index, dpi=dpi)

    skew = estimate_skew(_ink_mask(grey))
    if abs(skew) >= FINE_STEP_DEG:
        grey = _rotate(grey, skew, fill=255)

    blocks = [clean_text(b) for b in _blocks_in_reading_order(grey)]
    blocks = [b for b in blocks if b.strip()]
    text = "\n\n".join(blocks)
    score = splice_score(text)
    method = "auto"

    # Layout analysis failed to separate the columns - fall back to cutting the
    # page physically, and keep the attempt with less splicing.
    if score > 0:
        gutter_x = find_gutter(_ink_mask(grey))
        if gutter_x is not None:
            alt = [clean_text(b) for b in _column_split_blocks(grey, gutter_x)]
            alt = [b for b in alt if b.strip()]
            alt_text = "\n\n".join(alt)
            if splice_score(alt_text) < score:
                blocks, text, method = alt, alt_text, "column-split"
                score = splice_score(alt_text)

    return PageResult(
        source=source,
        page_no=index + 1,
        text=text,
        skew_deg=skew,
        splice_score=score,
        method=method,
        n_blocks=len(blocks),
        blocks=blocks,
    )


# ------------------------------------------------------------------ cleanup


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.!?:;\)\]])\n(?=[a-z])")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# Lines carrying no letters or digits at all - scan speckle and table rules.
_SPECKLE_LINE = re.compile(r"^[^A-Za-z0-9]*$")


def clean_text(raw: str) -> str:
    """Repair line-level OCR artefacts without touching the wording."""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufffd", "").replace("|", " ")
    # Rejoin words split across a line break by hyphenation.
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = "\n".join(
        line for line in text.split("\n") if not _SPECKLE_LINE.match(line.strip())
    )
    # Unwrap mid-sentence line breaks so chunks contain whole sentences.
    text = _SOFT_BREAK.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
