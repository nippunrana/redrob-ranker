"""Ranking evaluation metrics.

Pure numpy implementation — zero dependencies beyond numpy.

Convention choices (each choice is pinned by a test in test_metrics.py):

NDCG@k
  Positions are 1-indexed; discount = 1 / log2(rank + 1).
  gains="exponential" (default): gain = 2^rel - 1
    Matches sklearn.metrics.ndcg_score.
  gains="linear": gain = rel
    Simpler; produces identical ranking order to exponential for binary 0/1 labels.
  NDCG = DCG / IDCG; returns 0.0 when IDCG = 0 (no relevant docs in corpus).

Average Precision (AP) → MAP
  AP = Σ_k (P@k × rel_k) / R
  where the sum runs over ranks where a relevant doc appears, and R = total relevant docs.
  NOTE: different from sklearn.average_precision_score, which computes PR-curve AP
  using trapezoidal interpolation. Do not substitute sklearn's version for ranking AP.
  See references/evaluation.md for a full explanation of the difference.

P@k
  Standard: fraction of top-k results that are relevant. Denominator = k (not R).
  R-Precision (denominator = R) is NOT this function; apply precision_at_k(rels, k=R).

Recall@k
  relevant-in-top-k / total-relevant. Returns 0.0 if total_relevant = 0.

MRR
  Mean Reciprocal Rank. Reciprocal rank = 1 / rank of first relevant item (1-indexed).
  Returns 0.0 per query if no relevant item is found in the ranking.

Input format for all functions:
  `relevances` (per-query): list or array of ground-truth relevance grades in the order
  the retriever returned the documents. Index 0 = rank 1 (first result returned).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# P@k
# ---------------------------------------------------------------------------


def precision_at_k(relevances: list[int] | np.ndarray, k: int) -> float:
    """P@k: fraction of top-k results that are relevant. Denominator = k."""
    if k == 0:
        return 0.0
    rels = np.asarray(relevances, dtype=float)[:k]
    return float(np.sum(rels > 0) / k)


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def recall_at_k(
    relevances: list[int] | np.ndarray,
    k: int,
    num_relevant: int,
) -> float:
    """Recall@k: relevant-in-top-k / total-relevant. Returns 0.0 if num_relevant = 0."""
    if num_relevant == 0:
        return 0.0
    rels = np.asarray(relevances, dtype=float)[:k]
    return float(np.sum(rels > 0) / num_relevant)


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------


def _dcg_at_k(relevances: np.ndarray, k: int, gains: str) -> float:
    """Raw Discounted Cumulative Gain at k (internal helper)."""
    rels = relevances[:k]
    if len(rels) == 0:
        return 0.0
    positions = np.arange(1, len(rels) + 1, dtype=float)  # 1-indexed
    discounts = np.log2(positions + 1)
    if gains == "exponential":
        gain_values = 2.0**rels - 1.0
    elif gains == "linear":
        gain_values = rels.astype(float)
    else:
        raise ValueError(f"gains must be 'exponential' or 'linear', got {gains!r}")
    return float(np.sum(gain_values / discounts))


def ndcg_at_k(
    relevances: list[int] | np.ndarray,
    k: int,
    gains: str = "exponential",
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        relevances: Ground-truth grades in retrieval order. Index 0 = rank 1.
        k: Cutoff rank.
        gains: "exponential" (default, 2^rel - 1, matches sklearn.metrics.ndcg_score)
               or "linear" (rel; simpler, equivalent for binary 0/1 labels).

    Returns:
        NDCG in [0, 1]. Returns 0.0 if no relevant docs exist.
    """
    rels = np.asarray(relevances, dtype=float)
    actual_k = min(k, len(rels))
    dcg = _dcg_at_k(rels, actual_k, gains)
    ideal_rels = np.sort(rels)[::-1]
    idcg = _dcg_at_k(ideal_rels, actual_k, gains)
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


# ---------------------------------------------------------------------------
# Average Precision / MAP
# ---------------------------------------------------------------------------


def average_precision(relevances: list[int] | np.ndarray) -> float:
    """Average Precision (AP) for a single query.

    Formula: AP = Σ_k (P@k × rel_k) / R
    The sum runs over ranks where a relevant document appears; R = total relevant docs.

    NOTE: differs from sklearn.average_precision_score (PR-curve area). See module
    docstring for details.

    Returns 0.0 if no relevant documents appear in the ranking.
    """
    rels = np.asarray(relevances, dtype=float)
    num_relevant = int(np.sum(rels > 0))
    if num_relevant == 0:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for i, rel in enumerate(rels, start=1):
        if rel > 0:
            hits += 1
            ap_sum += hits / i
    return float(ap_sum / num_relevant)


def mean_average_precision(
    list_of_relevances: list[list[int]] | list[np.ndarray],
) -> float:
    """MAP: mean of AP across multiple queries.

    Returns 0.0 for an empty list.
    """
    if not list_of_relevances:
        return 0.0
    return float(np.mean([average_precision(rels) for rels in list_of_relevances]))


# ---------------------------------------------------------------------------
# Reciprocal Rank / MRR
# ---------------------------------------------------------------------------


def reciprocal_rank(relevances: list[int] | np.ndarray) -> float:
    """Reciprocal Rank for one query: 1 / rank of first relevant item (1-indexed).

    Returns 0.0 if no relevant document appears in the ranking.
    """
    rels = np.asarray(relevances, dtype=float)
    hit_indices = np.where(rels > 0)[0]
    if len(hit_indices) == 0:
        return 0.0
    first_hit_rank = int(hit_indices[0]) + 1  # convert 0-indexed to 1-indexed
    return float(1.0 / first_hit_rank)


def mean_reciprocal_rank(
    list_of_relevances: list[list[int]] | list[np.ndarray],
) -> float:
    """MRR: mean of reciprocal rank across multiple queries.

    Returns 0.0 for an empty list.
    """
    if not list_of_relevances:
        return 0.0
    return float(np.mean([reciprocal_rank(rels) for rels in list_of_relevances]))
