---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: sandbox/app.py
pinned: false
---

# Redrob Candidate Ranker

**Hack2Skill "India Runs" — Track 01: Intelligent Candidate Discovery & Ranking**

Ranks the top 100 best-fit candidates for *Senior AI Engineer, Founding Team @ Redrob AI*
from 100,000 candidates in **~10 seconds on CPU**.

---

## Reproduce

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. (One-time) Precompute dense recall artifact — ~12 min on CPU
#    Skip if artifacts/dense_recall_ids.npy is already committed (it is).
python precompute.py --candidates ./candidates.jsonl

# 3. Rank — produces EgniTech.csv in ~10s
python rank.py --candidates ./candidates.jsonl --out ./EgniTech.csv

# 4. Validate format
python ../validate_submission.py ./EgniTech.csv
```

**No network, no GPU, no model downloads at ranking time.**
`all-MiniLM-L6-v2` is downloaded once during `precompute.py` and not needed at ranking time —
`rank.py` only loads the committed `.npy` artifact.

---

## Architecture

### The core insight

The dataset is adversarial. The provided `sample_submission.csv` deliberately shows the wrong
answer: keyword stuffers (HR Manager, Accountant, Content Writer) ranked #1–#19 because they
paste 8–9 AI skills. Raw embedding or keyword similarity reproduces exactly this failure.

Winning requires:
1. **Title gating** — hard-reject non-tech titles before any embedding math runs.
2. **Career-history reading** — surface engineers who *actually built* retrieval/ranking systems
   (plain-language "tier-5s" hiding under generic titles).
3. **Skill trust** — weight skills by `endorsements × duration`, not mere presence.
4. **Honeypot guard** — the AI-relevant pool contains ~80 candidates with impossible timelines.
   A `honeypot_rate > 10%` in top-100 disqualifies the submission.

### Pipeline

```
candidates.jsonl (100K)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Layer 1 — Dense recall (offline artifact)       │
│  all-MiniLM-L6-v2 cosine vs JD text             │
│  → top-10K IDs  (artifacts/dense_recall_ids.npy) │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Layer 2 — BM25 lexical recall (runtime, ~6s)    │
│  BM25Okapi over headline+summary+career descs    │
│  → top-10K IDs                                  │
└─────────────────────────────────────────────────┘
        │  RRF (k=60)
        ▼
  5 000-candidate recall pool
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Layer 3 — Structured JD-grounding (scoring)     │
│                                                  │
│  fit = title      × 0.20                         │
│      + career_ret × 0.30  (retrieval/ranking     │
│      + career_prod× 0.15   evidence in career)   │
│      + skill_trust× 0.15  (endorsements×dur)     │
│      + exp_fit    × 0.10  (peak: 5–9 yrs)        │
│      + location   × 0.10  (India/relocate boost) │
└─────────────────────────────────────────────────┘
        │  × behavioral modifier [0.5, 1.0]
        │    (recency × OTW × response_rate × ...)
        │  × honeypot_multiplier
        │    (0.0 = both axes fire; 0.5 = one; 1.0 = clean)
        ▼
  sorted top-100
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Feature-grounded reasoning (no LLM)             │
│  Templates cite: role, company, YOE, named       │
│  skills, retrieval_hits, location, OTW status,   │
│  honest concerns (services, under/over-YOE).     │
└─────────────────────────────────────────────────┘
        │
        ▼
  EgniTech.csv  (100 rows, passes validate_submission.py)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Dense recall is precomputed | Encoding 100K on CPU takes ~12 min; ranking must be ≤5 min |
| BM25 rebuilt at runtime | Full-corpus IDF is correct; 6s is acceptable; no stale cache |
| Title gating before embedding | Hard-reject 70K non-tech profiles; saves compute + prevents stuffers |
| `career_retrieval` weight = 0.30 | Largest single feature; only "actually built" evidence counts |
| `DISQUALIFIER` → hard 0 | HR/Accountant/Designer with skills pasted can never rank |
| Honeypot guard two-axis | YOE inflation AND education timeline — both must fire for hard-zero |
| No LLM at ranking | No network; deterministic; all reasoning derivable from candidate object |

### Title classification

Titles are pre-classified against the 47 distinct titles in the dataset:

| Tier | Examples | Score |
|---|---|---|
| `STRONG_POSITIVE` | ML Engineer, NLP Engineer, RecSys Engineer, Search Engineer | 1.0 |
| `ADJACENT` | AI Engineer, Data Scientist, Backend Engineer | 0.5 |
| `UNKNOWN` | Software Engineer, Data Analyst | 0.35 |
| `RESEARCH_OR_CV` | AI Research Engineer, Computer Vision Engineer | 0.2 |
| `DISQUALIFIER` | HR Manager, Accountant, Content Writer, Graphic Designer | 0.0 (hard gate) |

### Honeypot guard

Two independent impossibility axes:

- **Pattern D** (YOE inflation): `claimed_yoe_months > actual_career_span_months + 36`.
  Catches candidates claiming 10+ years when their career history shows ≤4 years.
- **Pattern F** (education timeline): `implied_career_start_year < earliest_graduation_year - 2`.
  Catches candidates implying they started working before finishing their degree by 2+ years.

Both axes fire → `score = 0.0` (hard zero). One axis → `score × 0.50`.
Calibrated against CAND_0000031 (archetype, M.Tech 2002–2006 vs 6 YOE — passes both checks).

---

## Repository structure

```
redrob-ranker/
  rank.py                   # Entry point: candidates.jsonl → EgniTech.csv
  precompute.py             # One-time: build dense recall artifact
  src/
    data.py                 # Streaming JSONL parser, typed Candidate dataclass
    jd.py                   # JD encoding: title sets, career keywords, skill lists
    embeddings.py           # Load dense recall artifact (precomputed)
    bm25_index.py           # BM25 retrieval + RRF fusion
    features.py             # 6-component feature extraction
    behavioral.py           # Behavioral modifier from redrob_signals
    honeypot.py             # Impossibility guard (Patterns D + F)
    scoring.py              # Score combiner
    reasoning.py            # Feature-grounded reasoning string templates
  eval/
    metrics.py              # NDCG@k, MAP, P@k (matches challenge scoring exactly)
    gold_set.jsonl          # 110 hand-labeled candidates, tiers 0–5
    evaluate.py             # Offline NDCG/MAP against gold set
  artifacts/
    dense_recall_ids.npy    # Top-10K candidate IDs by JD cosine (committed)
    dense_recall_sims.npy   # Corresponding similarity scores
  sandbox/
    app.py                  # Streamlit demo app
    rank_demo.py            # BM25-only ranker for the demo (no artifact needed)
    make_sample.py          # Script to generate sample_candidates.jsonl
    sample_candidates.jsonl # 195-candidate demo sample (committed)
  tests/                    # 116 pytest tests (ruff + mypy clean)
  submission_metadata.yaml
  requirements.txt
```

---

## Tests

```bash
pytest tests/ -q
# 116 passed, 1 xfailed
```

---

## Sandbox demo

```bash
pip install -e ".[sandbox]"
streamlit run sandbox/app.py
```

Runs the BM25-only pipeline on `sandbox/sample_candidates.jsonl` (195 candidates including
the archetype CAND_0000031 and confirmed honeypot CAND_0055992) to demonstrate:
- No keyword stuffers in the top results
- Honeypot scored 0.0 and ranked last
- Feature-grounded reasoning for each result
