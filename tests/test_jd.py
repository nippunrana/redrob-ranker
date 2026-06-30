"""Tests for src/jd.py — JD requirement definitions."""

from src.jd import (
    TITLE_DISQUALIFIER,
    TITLE_RESEARCH_OR_CV,
    TITLE_STRONG_POSITIVE,
    SERVICES_COMPANIES,
    CAREER_RETRIEVAL_KEYWORDS,
    SKILLS_MUST_HAVE,
)


def test_titles_are_mutually_exclusive() -> None:
    """No title should appear in more than one tier."""
    overlap_strong_dq = TITLE_STRONG_POSITIVE & TITLE_DISQUALIFIER
    overlap_strong_research = TITLE_STRONG_POSITIVE & TITLE_RESEARCH_OR_CV
    overlap_dq_research = TITLE_DISQUALIFIER & TITLE_RESEARCH_OR_CV
    assert not overlap_strong_dq, f"Strong + disqualifier overlap: {overlap_strong_dq}"
    assert not overlap_strong_research, f"Strong + research overlap: {overlap_strong_research}"
    assert not overlap_dq_research, f"DQ + research overlap: {overlap_dq_research}"


def test_sample_submission_titles_are_disqualified() -> None:
    """The titles the sample_submission ranks #1-19 must all be disqualifiers."""
    # These are the titles from the deliberately-bad sample_submission.csv
    trap_titles = {
        "HR Manager",
        "ML Engineer",  # this one is NOT a disqualifier — sample also has real titles
        "Content Writer",
        "Business Analyst",
        "Marketing Manager",
        "Mechanical Engineer",
        "Accountant",
        "Sales Executive",
        "Operations Manager",
        "Graphic Designer",
        "Civil Engineer",
        "Project Manager",
        "Customer Support",
    }
    non_ml_trap_titles = trap_titles - {"ML Engineer"}
    for title in non_ml_trap_titles:
        assert title in TITLE_DISQUALIFIER, f"Expected '{title}' in TITLE_DISQUALIFIER"


def test_jd_explicitly_rejected_titles_are_penalised() -> None:
    """AI Research Engineer and Computer Vision Engineer must be in TITLE_RESEARCH_OR_CV."""
    assert "AI Research Engineer" in TITLE_RESEARCH_OR_CV
    assert "Computer Vision Engineer" in TITLE_RESEARCH_OR_CV


def test_archetype_title_is_strong_positive() -> None:
    """The tier-5 archetype (CAND_0000031) has title Recommendation Systems Engineer."""
    assert "Recommendation Systems Engineer" in TITLE_STRONG_POSITIVE


def test_services_companies_normalised() -> None:
    """Services company names are stored lowercase for case-insensitive matching."""
    for name in SERVICES_COMPANIES:
        assert name == name.lower(), f"Services company not lowercase: {name}"


def test_retrieval_keywords_include_core_must_haves() -> None:
    for kw in ("retrieval", "embedding", "ndcg", "bm25", "faiss", "ranking"):
        assert kw in CAREER_RETRIEVAL_KEYWORDS, f"Missing retrieval keyword: {kw}"


def test_skills_must_have_includes_key_skills() -> None:
    for skill in ("faiss", "pinecone", "python", "embeddings", "bm25"):
        assert skill in SKILLS_MUST_HAVE, f"Missing must-have skill: {skill}"
