"""
Re-OCR every page of every PDF into ocr_text/pages.jsonl.

Runs the pages across a process pool and is resumable: pages already present in
the output file are skipped, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "ocr_text"
OUT_FILE = OUT_DIR / "pages.jsonl"

PDFS = ["chapter 1_merged.pdf", "merged_ocr.pdf"]
WORKERS = 8


def _init_worker() -> None:
    # Tesseract parallelises internally via OpenMP. With several worker processes
    # that oversubscribes the CPU and runs slower than serial, so pin each worker
    # to one thread and let the pool provide the parallelism.
    os.environ["OMP_THREAD_LIMIT"] = "1"


def _ocr_one(job: tuple[str, int]) -> dict | None:
    """Worker: OCR a single page. Imported lazily so each process sets up once."""
    from ocr_pipeline import ocr_page

    path, index = job
    doc = pymupdf.open(ROOT / path)
    try:
        r = ocr_page(doc, index, source=path)
        return {
            "source": r.source,
            "page_no": r.page_no,
            "text": r.text,
            "skew_deg": r.skew_deg,
            "splice_score": r.splice_score,
            "method": r.method,
            "n_blocks": r.n_blocks,
        }
    except Exception as exc:                      # keep going; report at the end
        return {
            "source": path, "page_no": index + 1, "text": "",
            "skew_deg": 0.0, "splice_score": 0, "method": "ERROR",
            "n_blocks": 0, "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        doc.close()


def _already_done() -> set[tuple[str, int]]:
    if not OUT_FILE.exists():
        return set()
    done = set()
    with OUT_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                          # tolerate a torn final line
            if rec.get("method") != "ERROR":
                done.add((rec["source"], rec["page_no"]))
    return done


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    done = _already_done()

    jobs: list[tuple[str, int]] = []
    for path in PDFS:
        doc = pymupdf.open(ROOT / path)
        for i in range(doc.page_count):
            if (path, i + 1) not in done:
                jobs.append((path, i))
        doc.close()

    if not jobs:
        print(f"nothing to do - {len(done)} pages already in {OUT_FILE.name}")
        return

    print(f"{len(done)} pages already done, {len(jobs)} to go, {WORKERS} workers")
    started = time.perf_counter()
    errors, chars = 0, 0

    with OUT_FILE.open("a", encoding="utf-8") as out, \
            Pool(WORKERS, initializer=_init_worker) as pool:
        for n, rec in enumerate(pool.imap_unordered(_ocr_one, jobs, chunksize=4), 1):
            if rec is None:
                continue
            if rec["method"] == "ERROR":
                errors += 1
                print(f"  ERROR {rec['source']} p{rec['page_no']}: {rec.get('error')}")
            chars += len(rec["text"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 25 == 0 or n == len(jobs):
                rate = n / (time.perf_counter() - started)
                eta = (len(jobs) - n) / rate if rate else 0
                out.flush()
                print(f"  {n}/{len(jobs)} pages  {rate:.1f} pages/s  ETA {eta/60:.1f} min",
                      flush=True)

    elapsed = time.perf_counter() - started
    print(f"\ndone: {len(jobs)} pages in {elapsed/60:.1f} min, "
          f"{chars:,} chars, {errors} errors")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
