"""Build the hand-labeled gold set for offline evaluation.

Tiers (follow the JD rubric exactly):
  5 — Ideal: ret>=4, prod>=3, skills>=4, india=True, yoe 5-9, active/owf
  4 — Strong: ret>=3 OR major signals, india=True, yoe 4-10
  3 — Relevant: some retrieval career evidence, right title, partial signals
  2 — Weak: ML/AI title, skills listed but no retrieval career evidence
  1 — Wrong-fit: research/CV-primary, adjacent tech, barely relevant
  0 — Disqualified: non-tech role, services-only w/ no product exp

Run once: python eval/build_gold_set.py
Outputs: eval/gold_set.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Candidates manually labeled from the exploratory analysis.
# Format: (candidate_id, tier, note)
# Tier assignment traces directly to the JD rubric and the scoring vectors
# printed in the explore script (ret=retrieval_hits, prod=production_hits,
# skills=must_have_skill_matches, india, active, owf, rrr, yoe).
GOLD_LABELS: list[tuple[str, int, str]] = [
    # --- Tier 5: Ideal fit ---
    (
        "CAND_0000031",
        5,
        "RecSys@Swiggy; ret=5 prod=4 skills=5; india active owf; yoe=6",
    ),
    ("CAND_0011162", 5, "RecSysEng; ret=5 prod=5 skills=9; india active owf; yoe=5.8"),
    (
        "CAND_0006557",
        5,
        "NLP Engineer; ret=8 prod=4 skills=8; india active owf; yoe=7.9",
    ),
    ("CAND_0007412", 5, "Applied ML; ret=8 prod=4 skills=4; india active owf; yoe=7.4"),
    (
        "CAND_0008425",
        5,
        "Senior NLP; ret=11 prod=4 skills=5; india active owf; yoe=7.8",
    ),
    (
        "CAND_0002025",
        5,
        "Senior AI Eng; ret=6 prod=5 skills=6; india active owf; yoe=5.9",
    ),
    ("CAND_0009691", 5, "Applied ML; ret=6 prod=3 skills=4; india active owf; yoe=6.2"),
    (
        "CAND_0044222",
        5,
        "AI Eng (strong); ret=10 prod=3 skills=5; india active owf; yoe=7.7",
    ),
    # --- Tier 4: Strong fit ---
    (
        "CAND_0007009",
        4,
        "RecSysEng; ret=10 prod=0 skills=7; india owf; inactive; yoe=7.9",
    ),
    (
        "CAND_0007411",
        4,
        "Sr MLE; ret=7 prod=4 skills=6; india; inactive not-owf; yoe=8",
    ),
    (
        "CAND_0005260",
        4,
        "Senior NLP; ret=13 prod=3 skills=4; india active; not owf; yoe=5.2",
    ),
    ("CAND_0006418", 4, "MLE; ret=3 prod=4 skills=6; india owf; inactive; yoe=5.7"),
    (
        "CAND_0010685",
        4,
        "NLP Eng; ret=11 prod=5 skills=3; india owf; inactive; yoe=6.7",
    ),
    (
        "CAND_0005538",
        4,
        "Senior AI Eng; ret=6 prod=6 skills=1; india active owf; low skills",
    ),
    (
        "CAND_0006567",
        4,
        "Senior AI Eng; ret=4 prod=4 skills=2; india active owf; yoe=7.9",
    ),
    (
        "CAND_0030031",
        4,
        "AI Eng; ret=7 prod=5 skills=7; india active; not owf; yoe=5.7",
    ),
    ("CAND_0009024", 4, "Search Eng; ret=6 prod=3 skills=3; india active owf; yoe=5.2"),
    ("CAND_0010149", 4, "MLE; ret=2 prod=2 skills=4; india active owf; yoe=6.9"),
    ("CAND_0010603", 4, "MLE; ret=2 prod=2 skills=2; india active owf; yoe=5.3"),
    ("CAND_0050454", 4, "AI Eng; ret=9 prod=1 skills=3; india active owf; yoe=6.8"),
    ("CAND_0062247", 4, "AI Eng; ret=9 prod=1 skills=7; india active owf; yoe=7.3"),
    ("CAND_0058575", 4, "AI Eng; ret=10 prod=1 skills=2; india active owf; yoe=5.8"),
    # --- Tier 3: Relevant but missing key signals ---
    ("CAND_0000200", 3, "MLE; ret=1 prod=1 skills=3; india active owf; yoe=4.3 (low)"),
    ("CAND_0000273", 3, "MLE; ret=2 prod=2 skills=2; india active owf; yoe=5.8 ok"),
    ("CAND_0000981", 3, "MLE; ret=3 prod=2 skills=2; india; inactive not-owf; yoe=6.4"),
    ("CAND_0001610", 3, "MLE; ret=13 prod=4 skills=5; india active owf; yoe=3 too low"),
    ("CAND_0002037", 3, "MLE; ret=3 prod=2 skills=3; india active; not owf; yoe=5.5"),
    ("CAND_0002120", 3, "MLE; ret=3 prod=2 skills=5; non-India owf; yoe=6.5"),
    ("CAND_0003557", 3, "MLE; ret=2 prod=2 skills=4; india owf; inactive; yoe=6.8"),
    ("CAND_0003841", 3, "MLE; ret=2 prod=2 skills=2; india owf; inactive; yoe=5.0"),
    ("CAND_0004972", 3, "MLE; ret=1 prod=1 skills=3; india active; not owf; yoe=6.0"),
    ("CAND_0008049", 3, "MLE; ret=3 prod=2 skills=2; india; inactive not-owf; yoe=6.3"),
    ("CAND_0010655", 3, "MLE; ret=1 prod=1 skills=1; india active owf; yoe=5.8"),
    (
        "CAND_0010770",
        3,
        "RecSys; ret=11 prod=4 skills=3; india active owf; yoe=15 high",
    ),
    (
        "CAND_0003977",
        3,
        "RecSys; ret=7 prod=0 skills=3; india; inactive not-owf; yoe=4.6",
    ),
    ("CAND_0007460", 3, "AI Eng; ret=3 prod=1 skills=4; india active owf; yoe=4.7 low"),
    (
        "CAND_0008239",
        3,
        "AI Eng; ret=3 prod=1 skills=6; india owf; inactive; yoe=4.0 low",
    ),
    (
        "CAND_0015578",
        3,
        "AI Eng; ret=3 prod=0 skills=6; india active owf; no prod evidence",
    ),
    (
        "CAND_0060054",
        3,
        "AI Eng; ret=13 prod=4 skills=3; india; inactive not-owf; yoe=6.4",
    ),
    # --- Tier 2: ML/AI title but weak career evidence ---
    (
        "CAND_0001131",
        2,
        "MLE; ret=0 prod=1 skills=2; india owf; no retrieval career; yoe=5.8",
    ),
    (
        "CAND_0001808",
        2,
        "MLE; ret=0 prod=1 skills=2; india; inactive not-owf; yoe=3.8 low",
    ),
    (
        "CAND_0001819",
        2,
        "MLE; ret=0 prod=1 skills=2; india active owf; no retrieval; yoe=4.4",
    ),
    ("CAND_0003791", 2, "MLE; ret=0 prod=1 skills=3; non-India; inactive owf; yoe=6.6"),
    (
        "CAND_0005477",
        2,
        "MLE; ret=0 prod=1 skills=1; india active owf; low signals; yoe=6.2",
    ),
    (
        "CAND_0005704",
        2,
        "MLE; ret=0 prod=0 skills=4; india owf; no career evidence; yoe=3.3",
    ),
    (
        "CAND_0006870",
        2,
        "MLE; ret=0 prod=1 skills=2; india active owf; no retrieval; yoe=3.9",
    ),
    (
        "CAND_0007052",
        2,
        "MLE; ret=0 prod=1 skills=5; india owf; no retrieval career; yoe=6.5",
    ),
    ("CAND_0007088", 2, "MLE; ret=1 prod=0 skills=5; india; inactive not-owf; yoe=4.4"),
    ("CAND_0011140", 2, "MLE; ret=2 prod=2 skills=4; non-India; inactive owf; yoe=5.1"),
    (
        "CAND_0020350",
        2,
        "AI Eng; ret=0 prod=1 skills=4; india active owf; no retrieval; yoe=5.8",
    ),
    # --- Tier 1: Research/CV-primary or wrong fit ---
    (
        "CAND_0000112",
        1,
        "AI Specialist; ret=1 prod=0 skills=1; india; inactive not-owf",
    ),
    (
        "CAND_0000165",
        1,
        "AI Specialist; non-India active owf; JD no-visa, no reloc note",
    ),
    ("CAND_0000422", 1, "AI Research Eng; india owf; explicitly JD-rejected title"),
    ("CAND_0000705", 1, "AI Research Eng; ret=0; india active; pure research red flag"),
    (
        "CAND_0000969",
        1,
        "AI Specialist; india active owf; specialist=research-flavored",
    ),
    ("CAND_0001056", 1, "CV Engineer; india; JD: re-learning fundamentals penalise"),
    ("CAND_0001218", 1, "AI Specialist; non-India; owf; JD explicitly rejected type"),
    (
        "CAND_0001302",
        1,
        "CV Engineer; non-India; owf; CV primary, JD explicitly rejects",
    ),
    ("CAND_0001600", 1, "AI Specialist; india active; not owf; research-flavored"),
    ("CAND_0001940", 1, "AI Research Eng; non-India active; JD explicitly rejects"),
    (
        "CAND_0002270",
        1,
        "CV Engineer; india active; JD: CV/speech/robotics disqualifier",
    ),
    ("CAND_0002770", 1, "AI Research Eng; india owf; inactive; pure research flag"),
    ("CAND_0002793", 1, "AI Specialist; india; inactive not-owf; consulting flavored"),
    ("CAND_0003100", 1, "AI Specialist; india; inactive not-owf; research risk"),
    ("CAND_0003290", 1, "CV Engineer; india owf; inactive; CV primary"),
    (
        "CAND_0003506",
        1,
        "CV Engineer; india active; JD: CV/speech/robotics disqualifier",
    ),
    ("CAND_0003599", 1, "CV Engineer; india owf; inactive; CV primary"),
    ("CAND_0004402", 1, "AI Research Eng; india active owf; JD explicitly rejects"),
    ("CAND_0004628", 1, "AI Research Eng; india owf; JD explicitly rejects"),
    ("CAND_0005191", 1, "AI Specialist; india owf; inactive; research/consulting risk"),
    (
        "CAND_0000001",
        1,
        "Backend Eng; non-India; adjacent with zero retrieval evidence",
    ),
    (
        "CAND_0000010",
        1,
        "Data Eng; non-India; 1 retrieval hit but missing product signals",
    ),
    ("CAND_0000027", 1, "DevOps; india active owf; 1 retrieval hit; wrong domain"),
    ("CAND_0000032", 1, ".NET Dev; india; not-owf; minimal ML relevance"),
    (
        "CAND_0000015",
        1,
        "SW Eng; india owf; 1 retrieval hit; minimal evidence; yoe=5.4",
    ),
    # --- Tier 0: Disqualified ---
    ("CAND_0000002", 0, "Operations Manager; non-tech; no retrieval; india"),
    ("CAND_0000003", 0, "Customer Support; non-tech; non-India"),
    ("CAND_0000004", 0, "Marketing Manager; non-tech; non-India"),
    ("CAND_0000005", 0, "Accountant; non-tech; india"),
    ("CAND_0000006", 0, "Business Analyst; non-tech; non-India"),
    ("CAND_0000007", 0, "Civil Engineer; non-tech; india"),
    ("CAND_0000008", 0, "Operations Manager; non-tech; india; services company"),
    ("CAND_0000009", 0, "Mechanical Engineer; non-tech; non-India"),
    ("CAND_0000012", 0, "Operations Manager; non-tech; india"),
    ("CAND_0000013", 0, "Civil Engineer; non-tech; non-India"),
    ("CAND_0000016", 0, "Accountant; non-tech; india"),
    ("CAND_0000017", 0, "Accountant; non-tech; india"),
    ("CAND_0000019", 0, "Project Manager; non-tech; india"),
    ("CAND_0000020", 0, "Mechanical Engineer; non-tech; india"),
    ("CAND_0000021", 0, "Project Manager; non-tech; india; keyword-stuffer"),
    ("CAND_0000022", 0, "Mechanical Engineer; non-tech; non-India"),
    ("CAND_0000024", 0, "HR Manager; non-tech; india; services company"),
    ("CAND_0000026", 0, "Graphic Designer; non-tech; india"),
    ("CAND_0000028", 0, "Operations Manager; non-tech; non-India; services"),
    ("CAND_0000029", 0, "Business Analyst; non-tech; india"),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_path = Path(__file__).parent / "gold_set.jsonl"

    lines = []
    for cid, tier, note in GOLD_LABELS:
        lines.append(json.dumps({"candidate_id": cid, "tier": tier, "note": note}))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tier_dist: dict[int, int] = {}
    for _, tier, _ in GOLD_LABELS:
        tier_dist[tier] = tier_dist.get(tier, 0) + 1
    log.info(
        "Wrote %d gold labels to %s | tiers: %s",
        len(GOLD_LABELS),
        out_path,
        dict(sorted(tier_dist.items())),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
