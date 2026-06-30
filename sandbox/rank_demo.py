"""Sandbox-mode ranker: BM25-only recall (no precomputed dense artifact).

Used by sandbox/app.py to run a live demo on a small sample of candidates
without requiring the artifacts/ directory. Scoring and reasoning are
identical to the full rank.py pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure parent package is importable when run from sandbox/
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.bm25_index import bm25_top_k
from src.data import parse_candidate
from src.honeypot import honeypot_multiplier
from src.reasoning import generate_reasoning
from src.scoring import score_candidate

_BM25_QUERY: str = (
    "retrieval ranking embeddings vector search BM25 FAISS Qdrant Pinecone "
    "sentence-transformers NDCG MRR production recommendation ML engineer India "
    "hybrid search learning-to-rank evaluation offline online"
)


def _build_text(raw: dict) -> str:
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


def rank_candidates(
    candidates_path: Path,
    top_n: int = 20,
) -> list[dict]:
    """Rank candidates from a jsonl file; return top_n as list of dicts."""
    all_ids: list[str] = []
    all_texts: list[str] = []
    raw_by_id: dict[str, dict] = {}

    with candidates_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cid = raw["candidate_id"]
            all_ids.append(cid)
            all_texts.append(_build_text(raw))
            raw_by_id[cid] = raw

    recall_ids = bm25_top_k(all_texts, all_ids, _BM25_QUERY, k=min(500, len(all_ids)))

    scored: list[tuple[str, float]] = []
    for cid in recall_ids:
        raw = raw_by_id[cid]
        try:
            c = parse_candidate(raw)
        except Exception:  # noqa: BLE001
            continue
        fit, _ = score_candidate(c)
        hp = honeypot_multiplier(c)
        scored.append((cid, fit * hp))

    scored.sort(key=lambda x: (-x[1], x[0]))

    results = []
    for rank, (cid, score) in enumerate(scored[:top_n], start=1):
        raw = raw_by_id[cid]
        c = parse_candidate(raw)
        _, features = score_candidate(c)
        reasoning = generate_reasoning(c, features, rank=rank)
        results.append(
            {
                "rank": rank,
                "candidate_id": cid,
                "score": round(score, 4),
                "title": c.profile.current_title,
                "company": c.profile.current_company,
                "yoe": c.profile.years_of_experience,
                "location": c.profile.location,
                "reasoning": reasoning,
            }
        )

    return results
