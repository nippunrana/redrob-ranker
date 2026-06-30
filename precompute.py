"""Offline precomputation: dense embeddings → top-10K recall artifact.

Run once before rank.py. Encodes all 100K candidate texts with all-MiniLM-L6-v2
and saves the top-10K candidate IDs by JD cosine similarity to artifacts/.

This step is allowed to exceed the 5-min ranking budget (it takes ~3 min on CPU).
rank.py loads the artifact at runtime and re-runs BM25 live (fast: ~30 sec).

Usage:
    python precompute.py --candidates problem-statement/data/candidates.jsonl
    python precompute.py --candidates problem-statement/data/candidates.jsonl \\
        --artifacts artifacts/ --top-n 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make src importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent))

from src.data import iter_candidates
from src.jd import JD_TEXT

log = logging.getLogger(__name__)

# Number of top candidates to store in the dense recall artifact.
DENSE_TOP_N: int = 10_000
# Batch size for encoding — larger = faster on CPU (up to memory limit).
ENCODE_BATCH: int = 64


def _load_model(model_name: str = "all-MiniLM-L6-v2"):  # type: ignore[no-untyped-def]
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    log.info("Loading model: %s", model_name)
    return SentenceTransformer(model_name)


def _encode(model, texts: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Encode texts and return float32 normalized embeddings [N, D]."""
    log.info("Encoding %d texts (batch_size=%d) …", len(texts), ENCODE_BATCH)
    embs = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        convert_to_tensor=False,  # stay numpy for easy np.save
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.array(embs, dtype=np.float32)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    candidates_path = Path(args.candidates)
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Stream all candidates, collect IDs and texts ─────────────────────
    log.info("Streaming candidates from %s …", candidates_path)
    all_ids: list[str] = []
    all_texts: list[str] = []
    for c in iter_candidates(candidates_path):
        all_ids.append(c.candidate_id)
        all_texts.append(c.text)
    log.info("Loaded %d candidates.", len(all_ids))

    # ── 2. Load model and encode ─────────────────────────────────────────────
    model = _load_model()

    log.info("Encoding JD query …")
    query_emb = model.encode(
        JD_TEXT,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    query_emb = np.array(query_emb, dtype=np.float32)  # shape [D]

    corpus_embs = _encode(model, all_texts)  # shape [N, D]

    # ── 3. Compute cosine similarity (normalized → dot product) ──────────────
    log.info("Computing similarities …")
    sims = corpus_embs @ query_emb  # shape [N]

    # ── 4. Take top-N by similarity ──────────────────────────────────────────
    top_n = min(args.top_n, len(all_ids))
    top_indices = np.argsort(sims)[::-1][:top_n]
    top_ids = np.array([all_ids[i] for i in top_indices])
    top_sims = sims[top_indices]

    log.info(
        "Top-5 dense recall: %s",
        list(zip(top_ids[:5].tolist(), top_sims[:5].tolist())),  # noqa: B905
    )

    # ── 5. Save artifacts ─────────────────────────────────────────────────────
    dense_path = artifacts_dir / "dense_recall_ids.npy"
    np.save(dense_path, top_ids)
    log.info("Saved %d dense recall IDs → %s", len(top_ids), dense_path)

    # Also save similarity scores for diagnostics.
    sims_path = artifacts_dir / "dense_recall_sims.npy"
    np.save(sims_path, top_sims)
    log.info("Saved dense recall similarities → %s", sims_path)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (or .jsonl.gz)",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory to write precomputed artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DENSE_TOP_N,
        help=f"Number of top-N dense recall IDs to save (default: {DENSE_TOP_N})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
