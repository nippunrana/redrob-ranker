"""rank.py — end-to-end entry point for the Redrob candidate ranker.

Usage:
    python rank.py --candidates <path/to/candidates.jsonl> --out <submission.csv>

Produces a CSV with exactly 100 rows (header + data) passing validate_submission.py.
Runs in ≤5 minutes on a 16 GB CPU-only machine by loading precomputed dense recall
artifacts from artifacts/dense_recall_ids.npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from src.bm25_index import bm25_top_k, rrf_fuse
from src.data import parse_candidate
from src.embeddings import load_dense_recall
from src.honeypot import honeypot_multiplier
from src.reasoning import generate_reasoning
from src.scoring import score_candidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BM25 query: JD key technical vocabulary for lexical recall
_BM25_QUERY: str = (
    "retrieval ranking embeddings vector search BM25 FAISS Qdrant Pinecone "
    "sentence-transformers NDCG MRR production recommendation ML engineer India "
    "hybrid search learning-to-rank evaluation offline online correlation"
)

# Recall + scoring limits
_DENSE_TOP_N: int = 10_000
_BM25_TOP_N: int = 10_000
_RRF_TOP_N: int = 5_000
_FINAL_TOP_N: int = 100


def _build_text(raw: dict) -> str:
    """Build BM25 search text from a raw candidate dict (no full Candidate parse)."""
    profile = raw.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]
    for ch in raw.get("career_history", []):
        parts.append(ch.get("description", ""))
        parts.append(ch.get("title", ""))
    for sk in raw.get("skills", []):
        parts.append(sk.get("name", ""))
    return " ".join(p for p in parts if p)


def _stream_pass(
    candidates_path: Path,
    dense_set: set[str],
) -> tuple[list[str], list[str], dict[str, dict]]:
    """Single streaming pass over candidates.jsonl.

    Returns:
        all_ids:    Ordered list of all candidate IDs (for BM25 corpus).
        all_texts:  Corresponding BM25 texts.
        raw_cache:  Raw dicts for candidates in dense_set (for scoring later).
    """
    all_ids: list[str] = []
    all_texts: list[str] = []
    raw_cache: dict[str, dict] = {}

    with candidates_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cid = raw["candidate_id"]
            text = _build_text(raw)
            all_ids.append(cid)
            all_texts.append(text)
            if cid in dense_set:
                raw_cache[cid] = raw

    return all_ids, all_texts, raw_cache


def _fetch_missing(
    candidates_path: Path,
    needed: set[str],
) -> dict[str, dict]:
    """Second streaming pass — collect raw dicts for IDs not in cache."""
    found: dict[str, dict] = {}
    if not needed:
        return found

    with candidates_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cid = raw["candidate_id"]
            if cid in needed:
                found[cid] = raw
                if len(found) == len(needed):
                    break  # early exit when all found

    return found


def _write_csv(
    results: list[tuple[str, int, float, str]],
    out_path: Path,
) -> None:
    """Write the submission CSV: candidate_id, rank, score, reasoning."""
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, score, reasoning in results:
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank top-100 candidates for Senior AI Engineer @ Redrob AI."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output submission CSV path",
    )
    args = parser.parse_args()

    t0 = time.time()

    candidates_path: Path = args.candidates
    out_path: Path = args.out
    artifacts_dir = Path(__file__).parent / "artifacts"

    if not candidates_path.exists():
        log.error("candidates file not found: %s", candidates_path)
        return 1

    # ------------------------------------------------------------------
    # Step 1: Load dense recall artifact
    # ------------------------------------------------------------------
    log.info("Loading dense recall artifact …")
    dense_ids = load_dense_recall(artifacts_dir, top_n=_DENSE_TOP_N)
    dense_set = set(dense_ids)
    log.info("Dense recall: %d IDs loaded", len(dense_ids))

    # ------------------------------------------------------------------
    # Step 2: Single streaming pass — build BM25 corpus + cache dense hits
    # ------------------------------------------------------------------
    log.info("Streaming candidates for BM25 corpus (pass 1) …")
    all_ids, all_texts, raw_cache = _stream_pass(candidates_path, dense_set)
    log.info(
        "Corpus: %d candidates, %d dense hits cached",
        len(all_ids),
        len(raw_cache),
    )

    # ------------------------------------------------------------------
    # Step 3: BM25 retrieval
    # ------------------------------------------------------------------
    log.info("Building BM25 index + scoring …")
    bm25_ids = bm25_top_k(all_texts, all_ids, _BM25_QUERY, k=_BM25_TOP_N)
    log.info("BM25 top-%d retrieved", len(bm25_ids))

    # ------------------------------------------------------------------
    # Step 4: RRF fusion → recall pool
    # ------------------------------------------------------------------
    recall_pool = rrf_fuse(dense_ids, bm25_ids, k=60, top_n=_RRF_TOP_N)
    log.info("RRF recall pool: %d candidates", len(recall_pool))

    # ------------------------------------------------------------------
    # Step 5: Fetch any BM25-only hits not in dense cache (pass 2)
    # ------------------------------------------------------------------
    missing_ids = {cid for cid in recall_pool if cid not in raw_cache}
    if missing_ids:
        log.info("Fetching %d BM25-only candidates (pass 2) …", len(missing_ids))
        extra = _fetch_missing(candidates_path, missing_ids)
        raw_cache.update(extra)
        log.info("Fetched %d / %d", len(extra), len(missing_ids))

    # ------------------------------------------------------------------
    # Step 6: Score each recall candidate
    # ------------------------------------------------------------------
    log.info("Scoring %d recall candidates …", len(recall_pool))
    scored: list[tuple[str, float]] = []
    skipped = 0
    for cid in recall_pool:
        raw = raw_cache.get(cid)
        if raw is None:
            skipped += 1
            continue
        try:
            c = parse_candidate(raw)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        fit, _ = score_candidate(c)
        hp = honeypot_multiplier(c)
        final_score = fit * hp
        scored.append((cid, final_score))

    if skipped:
        log.warning("Skipped %d candidates (missing raw or parse error)", skipped)
    log.info("Scored %d candidates", len(scored))

    # ------------------------------------------------------------------
    # Step 7: Sort and take top-100
    # ------------------------------------------------------------------
    # Primary: score DESC; tie-break: candidate_id ASC (per validate spec)
    scored.sort(key=lambda x: (-x[1], x[0]))
    top100 = scored[:_FINAL_TOP_N]

    if len(top100) < _FINAL_TOP_N:
        log.error(
            "Only %d candidates scored; need %d. Recall pool may be too small.",
            len(top100),
            _FINAL_TOP_N,
        )
        return 1

    # ------------------------------------------------------------------
    # Step 8: Generate reasoning + build output rows
    # ------------------------------------------------------------------
    log.info("Generating reasoning for top-%d …", _FINAL_TOP_N)
    results: list[tuple[str, int, float, str]] = []
    for rank, (cid, final_score) in enumerate(top100, start=1):
        raw = raw_cache[cid]
        c = parse_candidate(raw)
        _, features = score_candidate(c)
        reasoning = generate_reasoning(c, features, rank=rank)
        results.append((cid, rank, final_score, reasoning))

    # ------------------------------------------------------------------
    # Step 9: Write CSV
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(results, out_path)
    log.info("Submission written → %s", out_path)

    elapsed = time.time() - t0
    log.info("Total time: %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Quick sanity check: non-increasing scores
    # ------------------------------------------------------------------
    scores = [r[2] for r in results]
    violations = sum(1 for i in range(len(scores) - 1) if scores[i] < scores[i + 1])
    if violations:
        log.error("Score ordering violated in %d place(s)", violations)
        return 1

    log.info("Done. Wrote %d rows. Run validate_submission.py to confirm.", 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
