"""Streamlit sandbox app — Redrob AI Candidate Ranker demo.

Runs the full scoring pipeline on a 200-candidate sample to demonstrate:
  - Keyword stuffers correctly ranked below real ML engineers
  - Honeypot detection (confirmed honeypots scored 0)
  - Feature-grounded reasoning (no LLM, no network)
  - <10s end-to-end on CPU

Deploy: streamlit run sandbox/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sandbox.rank_demo import rank_candidates

_SAMPLE_PATH = Path(__file__).parent / "sample_candidates.jsonl"

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob AI — Senior AI Engineer Candidate Ranker")
st.caption(
    "Hybrid retrieval + grounded rerank on a 200-candidate sample. "
    "No LLM, no network, CPU-only. Runs ≤10 seconds."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Pipeline:**
        1. BM25 lexical recall over sample
        2. 6-component structured feature scoring
           - Title tier (0.20)
           - Career retrieval evidence (0.30)
           - Production deployment (0.15)
           - Skill trust (0.15)
           - Experience fit (0.10)
           - Location boost (0.10)
        3. Behavioral modifier (activity × response rate)
        4. Honeypot guard (hard-zero confirmed fakes)
        5. Feature-grounded reasoning strings

        **Full pipeline:** run `rank.py --candidates candidates.jsonl`
        on 100K candidates in ~10s.
        """
    )
    top_n = st.slider("Show top N results", min_value=5, max_value=50, value=20, step=5)

st.divider()

if not _SAMPLE_PATH.exists():
    st.error(
        f"Sample file not found: `{_SAMPLE_PATH}`\n\n"
        "Run `python sandbox/make_sample.py --candidates <path/to/candidates.jsonl>` "
        "to generate it."
    )
    st.stop()

if st.button("▶ Run Ranker", type="primary"):
    with st.spinner("Ranking candidates…"):
        results = rank_candidates(_SAMPLE_PATH, top_n=top_n)

    st.success(f"Ranked {top_n} candidates in the sample.")

    for row in results:
        hp_flag = " 🚨 HONEYPOT" if row["score"] == 0.0 else ""
        with st.expander(
            f"**#{row['rank']}** {row['title']} @ {row['company']}  "
            f"— score {row['score']:.4f}{hp_flag}"
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Candidate ID", row["candidate_id"])
            col2.metric("YOE", f"{row['yoe']:.1f} yrs")
            col3.metric("Location", row["location"] or "—")
            st.info(row["reasoning"])
