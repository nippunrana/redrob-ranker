"""BM25 lexical retrieval and Reciprocal Rank Fusion.

At runtime, BM25 is built from all 100K candidate texts (takes ~15-25 sec on CPU).
The IDF is computed over the full corpus so term weights are correct.

rrf_fuse() combines dense recall ranks with BM25 recall ranks into a single
fused recall list for structured feature scoring in Phase 4.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import]

log = logging.getLogger(__name__)

# BM25 parameters (Okapi BM25 standard defaults)
BM25_K1: float = 1.5
BM25_B: float = 0.75


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer, lowercased. Matches precompute symmetry."""
    return text.lower().split()


def bm25_top_k(
    all_texts: list[str],
    all_ids: list[str],
    query: str,
    k: int = 10_000,
) -> list[str]:
    """Build a BM25 index over all_texts and return top-k candidate IDs.

    Args:
        all_texts: One text per candidate, same order as all_ids.
        all_ids: Candidate IDs matching all_texts.
        query: BM25 query string (tokenized identically to corpus).
        k: Number of top candidates to return.

    Returns:
        List of candidate_id strings, BM25-best-first, length ≤ k.
    """
    if not all_texts:
        return []
    log.info("Building BM25 index over %d candidates …", len(all_texts))
    tokenized_corpus = [_tokenize(t) for t in all_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_k = min(k, len(all_ids))
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_ids = [all_ids[i] for i in top_indices]
    log.info("BM25 top-5: %s", top_ids[:5])
    return top_ids


def rrf_fuse(
    dense_ids: list[str],
    bm25_ids: list[str],
    k: int = 60,
    top_n: int = 5_000,
) -> list[str]:
    """Reciprocal Rank Fusion of dense and BM25 recall lists.

    Score formula: RRF(doc) = Σ 1 / (k + rank)  where rank is 1-indexed.
    Docs absent from a ranking are skipped (not penalised as rank ∞).

    Args:
        dense_ids: Candidate IDs ordered best-first by dense similarity.
        bm25_ids: Candidate IDs ordered best-first by BM25 score.
        k: Smoothing constant (default 60, Cormack et al. 2009).
        top_n: Return only the top-n fused candidates.

    Returns:
        Fused list of candidate_id strings, highest-RRF-score first.
    """
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] += 1.0 / (k + rank)
    fused = sorted(scores, key=scores.__getitem__, reverse=True)
    return fused[:top_n]
