"""Load precomputed dense recall artifact at runtime.

precompute.py (offline) generates:
  artifacts/dense_recall_ids.npy  — candidate IDs ordered best-first by JD cosine
  artifacts/dense_recall_sims.npy — corresponding similarity scores (diagnostics only)

rank.py loads these at startup via load_dense_recall().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dense_recall(
    artifacts_dir: Path,
    top_n: int = 10_000,
) -> list[str]:
    """Return candidate IDs pre-ranked by JD cosine similarity (best-first).

    The returned list is ordered: index 0 = rank 1 (most similar to JD).
    Pass this to rrf_fuse() as the dense ranking.

    Args:
        artifacts_dir: Directory containing dense_recall_ids.npy.
        top_n: Slice to this many candidates (default 10 000).

    Returns:
        List of candidate_id strings, best-first.
    """
    path = artifacts_dir / "dense_recall_ids.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Dense recall artifact not found: {path}\n"
            "Run: python precompute.py --candidates <path>"
        )
    ids: np.ndarray = np.load(path, allow_pickle=True)
    return list(ids[:top_n])
