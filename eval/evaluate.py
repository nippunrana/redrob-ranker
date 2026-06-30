"""Offline evaluation against the hand-labeled gold set.

Compute NDCG@10, NDCG@50, MAP, P@10 for a ranked submission CSV.
Also runs sanity checks: are keyword stuffers absent from top-10?
Is CAND_0000031 (archetype tier-5) in the top-20?

Usage:
    python eval/evaluate.py --submission submission.csv

The gold set treats the submission as a single-query ranking.
Relevance grades: tier 0-5 (5 = ideal).  Only candidates present in the
gold set contribute; unlabeled candidates are ignored.

NOTE: This eval is circular — we wrote both labels and the ranker.
Use it for regression / sanity checks only, not as a final quality claim.
Primary quality check: manual inspection of the top-30 (stuffers out?
tier-5s in? no honeypots?).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from eval.metrics import mean_average_precision, ndcg_at_k, precision_at_k

log = logging.getLogger(__name__)

# Tier 0-1 candidates that must NOT appear in top-10 (trap / disqualifier check)
TRAP_CANDIDATES: frozenset[str] = frozenset(
    {
        "CAND_0000002",  # Operations Manager
        "CAND_0000003",  # Customer Support
        "CAND_0000004",  # Marketing Manager
        "CAND_0000005",  # Accountant
        "CAND_0000006",  # Business Analyst
        "CAND_0000007",  # Civil Engineer
        "CAND_0000021",  # Project Manager (keyword-stuffer)
        "CAND_0000422",  # AI Research Engineer (JD-rejected)
        "CAND_0001056",  # Computer Vision Engineer (JD-rejected)
    }
)

# The archetype tier-5 candidate must appear in top-20.
ARCHETYPE_ID: str = "CAND_0000031"


def load_gold(gold_path: Path) -> dict[str, int]:
    """Return {candidate_id: tier} from gold_set.jsonl."""
    gold: dict[str, int] = {}
    with open(gold_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gold[row["candidate_id"]] = int(row["tier"])
    return gold


def load_submission(csv_path: Path) -> list[tuple[str, int, float]]:
    """Return [(candidate_id, rank, score)] sorted by rank ascending."""
    rows: list[tuple[str, int, float]] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r["candidate_id"].strip(), int(r["rank"]), float(r["score"])))
    rows.sort(key=lambda x: x[1])
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    gold_path = Path(__file__).parent / "gold_set.jsonl"
    gold = load_gold(gold_path)
    rows = load_submission(args.submission)

    # Build relevance list in submission rank order, for gold-labeled candidates.
    # Unlabeled candidates (not in gold) are not scored — we skip them.
    ranked_ids = [cid for cid, _, _ in rows]
    relevances: list[int] = [gold[cid] for cid in ranked_ids if cid in gold]
    labeled_in_submission = [cid for cid in ranked_ids if cid in gold]
    n_labeled = len(labeled_in_submission)

    if n_labeled == 0:
        log.error("No gold-labeled candidates found in submission — check IDs.")
        return 1

    log.info("Labeled candidates in submission: %d / %d", n_labeled, len(rows))

    ndcg10 = ndcg_at_k(relevances, k=10)
    ndcg50 = ndcg_at_k(relevances, k=50)
    map_score = mean_average_precision([[r for r in relevances]])
    p10 = precision_at_k(relevances, k=10)

    # Weighted composite (same as challenge formula)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10

    log.info("--- Offline eval (gold-labeled subset, circular) ---")
    log.info("  NDCG@10  : %.4f", ndcg10)
    log.info("  NDCG@50  : %.4f", ndcg50)
    log.info("  MAP      : %.4f", map_score)
    log.info("  P@10     : %.4f", p10)
    log.info("  Composite: %.4f", composite)

    # Sanity checks
    top10_ids = set(ranked_ids[:10])
    top20_ids = set(ranked_ids[:20])

    trap_in_top10 = top10_ids & TRAP_CANDIDATES
    if trap_in_top10:
        log.warning("TRAP ALERT: disqualifier/stuffer in top-10: %s", trap_in_top10)
    else:
        log.info("TRAP CHECK: no known disqualifiers in top-10 ✓")

    if ARCHETYPE_ID in top20_ids:
        arch_rank = ranked_ids.index(ARCHETYPE_ID) + 1
        log.info("ARCHETYPE CHECK: %s at rank %d ✓", ARCHETYPE_ID, arch_rank)
    else:
        log.warning(
            "ARCHETYPE MISSING: %s not in top-20 (ranked %s)",
            ARCHETYPE_ID,
            ranked_ids.index(ARCHETYPE_ID) + 1 if ARCHETYPE_ID in ranked_ids else "N/A",
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission", type=Path, required=True, help="Path to submission CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
