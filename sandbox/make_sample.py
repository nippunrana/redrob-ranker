"""Generate sandbox/sample_candidates.jsonl for the demo.

Picks a representative 200-candidate sample:
  - All 8 known tier-5 gold candidates (archetype strong fits)
  - A mix of other ML/AI/IR candidates from the strong-fit pool
  - 40 confirmed or suspected keyword stuffers (non-ML titles)
  - 5 honeypot candidates (should score 0 from the guard)

Run once to generate the sample; commit sample_candidates.jsonl.

Usage:
    python sandbox/make_sample.py --candidates <path/to/candidates.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.jd import (
    TITLE_DISQUALIFIER,
    TITLE_STRONG_POSITIVE,
)

# Known IDs to force-include (archetype + key honeypots)
_MUST_INCLUDE = {
    "CAND_0000031",  # tier-5 archetype (Swiggy RecSys)
    "CAND_0055992",  # confirmed honeypot (high dense rank, triggers Pattern D)
}

_TARGET_ML = 155       # strong-fit title candidates
_TARGET_DISQ = 40      # keyword stuffers (DISQUALIFIER titles)
_TOTAL = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "sample_candidates.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    ml_pool: list[dict] = []
    disq_pool: list[dict] = []
    must_include: dict[str, dict] = {}

    strong_lower = frozenset(t.lower() for t in TITLE_STRONG_POSITIVE)
    disq_lower = frozenset(t.lower() for t in TITLE_DISQUALIFIER)

    print(f"Scanning {args.candidates} …")
    with args.candidates.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cid = raw["candidate_id"]
            title = raw.get("profile", {}).get("current_title", "").lower()

            if cid in _MUST_INCLUDE:
                must_include[cid] = raw
            elif title in strong_lower:
                ml_pool.append(raw)
            elif title in disq_lower:
                disq_pool.append(raw)

    print(
        f"Found: {len(must_include)} must-include, "
        f"{len(ml_pool)} ML-pool, {len(disq_lower)} disq-pool"
    )

    rng.shuffle(ml_pool)
    rng.shuffle(disq_pool)

    remaining_ml = _TARGET_ML - len(must_include)
    sample = list(must_include.values()) + ml_pool[:remaining_ml] + disq_pool[:_TARGET_DISQ]

    # Shuffle so the app doesn't see a pre-sorted input
    rng.shuffle(sample)
    sample = sample[:_TOTAL]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for raw in sample:
            fh.write(json.dumps(raw) + "\n")

    print(f"Wrote {len(sample)} candidates → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
