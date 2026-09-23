"""Quantify the improvement: splicing in the embedded text layer vs our re-OCR."""

from __future__ import annotations

import random
import time

import pymupdf

from ocr_pipeline import ocr_page, splice_score

SAMPLE_PER_DOC = 14
SEED = 7


def main() -> None:
    rng = random.Random(SEED)
    totals = {"old": 0, "new": 0, "pages": 0, "old_pages": 0, "new_pages": 0}
    started = time.perf_counter()

    for path in ("merged_ocr.pdf", "chapter 1_merged.pdf"):
        doc = pymupdf.open(path)
        pages = sorted(rng.sample(range(doc.page_count), SAMPLE_PER_DOC))
        print("=" * 78)
        print(f"{path}   sampling {len(pages)} of {doc.page_count} pages")
        print(f"  {'page':>5} {'skew':>6} {'method':>13} {'blocks':>7} "
              f"{'splice OLD':>11} {'NEW':>5} {'chars':>7}")
        for idx in pages:
            old = doc[idx].get_text()
            r = ocr_page(doc, idx, source=path)
            old_s, new_s = splice_score(old), r.splice_score
            flag = "  <-- still spliced" if new_s > 0 else ""
            print(f"  {idx + 1:>5} {r.skew_deg:>+6.2f} {r.method:>13} {r.n_blocks:>7} "
                  f"{old_s:>11} {new_s:>5} {r.char_count:>7}{flag}")
            totals["old"] += old_s
            totals["new"] += new_s
            totals["pages"] += 1
            totals["old_pages"] += 1 if old_s else 0
            totals["new_pages"] += 1 if new_s else 0
        doc.close()

    elapsed = time.perf_counter() - started
    print("=" * 78)
    print(f"spliced lines   OLD {totals['old']:>4}   ->   NEW {totals['new']:>4}")
    print(f"affected pages  OLD {totals['old_pages']:>4}   ->   NEW {totals['new_pages']:>4}"
          f"   (of {totals['pages']})")
    print(f"{elapsed:.0f}s for {totals['pages']} pages "
          f"= {elapsed / totals['pages']:.1f}s/page single-threaded")


if __name__ == "__main__":
    main()
