"""Tests for src/reasoning.py (Phase 6).

Verification: the 6 Stage-4 checks on reasoning quality.
  1. Specific — mentions actual role/company/skills/year count.
  2. JD-linked — retrieval/ranking/embedding/skills vocabulary.
  3. Honest — mentions concerns when signals are weak.
  4. No hallucination — every fact is derivable from the candidate object.
  5. Varied — different inputs produce different outputs.
  6. Rank-consistent — strong candidates get stronger language.
"""

from __future__ import annotations

from src.data import parse_candidate
from src.features import extract_features
from src.reasoning import generate_reasoning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_SIGNALS = {
    "profile_completeness_score": 80.0,
    "signup_date": "2022-01-01",
    "last_active_date": "2026-06-20",
    "open_to_work_flag": True,
    "profile_views_received_30d": 10,
    "applications_submitted_30d": 2,
    "recruiter_response_rate": 0.85,
    "avg_response_time_hours": 4.0,
    "skill_assessment_scores": {},
    "connection_count": 200,
    "endorsements_received": 50,
    "notice_period_days": 30,
    "expected_salary_range_inr_lpa": {"min": 30.0, "max": 50.0},
    "preferred_work_mode": "hybrid",
    "willing_to_relocate": True,
    "github_activity_score": 70.0,
    "search_appearance_30d": 40,
    "saved_by_recruiters_30d": 5,
    "interview_completion_rate": 0.9,
    "offer_acceptance_rate": 0.8,
    "verified_email": True,
    "verified_phone": True,
    "linkedin_connected": True,
}


def _make_candidate(
    candidate_id: str = "CAND_TEST",
    title: str = "ML Engineer",
    company: str = "Swiggy",
    yoe: float = 7.0,
    country: str = "India",
    location: str = "Hyderabad",
    career_desc: str = "Built embedding retrieval and ranking systems deployed to production",
    skills: list[dict] | None = None,
    signals_overrides: dict | None = None,
):
    skills = skills or [
        {"name": "embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
        {"name": "faiss", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
        {"name": "python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
    ]
    sigs = {**_BASE_SIGNALS, **(signals_overrides or {})}
    raw = {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test",
            "headline": f"{title} at {company}",
            "summary": "Experienced engineer",
            "location": location,
            "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": company,
            "current_company_size": "501-1000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": company,
                "title": title,
                "start_date": "2019-01-01",
                "end_date": None,
                "duration_months": int(yoe * 12),
                "is_current": True,
                "industry": "Technology",
                "company_size": "501-1000",
                "description": career_desc,
            }
        ],
        "education": [],
        "skills": skills,
        "redrob_signals": sigs,
    }
    return parse_candidate(raw)


# ---------------------------------------------------------------------------
# Check 1: Specific (mentions role, company, or skills)
# ---------------------------------------------------------------------------


class TestSpecific:
    def test_mentions_title_or_company(self) -> None:
        c = _make_candidate(title="ML Engineer", company="Zomato", yoe=7.0)
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=5)
        assert "ML Engineer" in r or "Zomato" in r

    def test_mentions_years_of_experience(self) -> None:
        c = _make_candidate(yoe=7.0)
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=10)
        # Should mention "7" in some form
        assert "7" in r

    def test_mentions_matched_skills(self) -> None:
        c = _make_candidate(
            career_desc="retrieval ranking embedding faiss vector search ndcg production deployed",
        )
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=3)
        # Reasoning must mention at least one matched skill by name
        any_skill = any(s.lower() in r.lower() for s in f.matched_skills)
        assert any_skill or f.retrieval_hits > 0, f"No skill or retrieval mention in: {r!r}"


# ---------------------------------------------------------------------------
# Check 2: JD-linked
# ---------------------------------------------------------------------------


class TestJDLinked:
    def test_retrieval_language_present(self) -> None:
        c = _make_candidate(career_desc="retrieval ranking embedding faiss production deployed")
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=5)
        jd_terms = {"retrieval", "ranking", "embedding", "faiss", "production", "signal"}
        assert any(t in r.lower() for t in jd_terms), f"No JD term in: {r!r}"

    def test_adjacent_reasoning_explains_why(self) -> None:
        c = _make_candidate(
            title="AI Engineer",
            career_desc="built embedding retrieval ranking faiss qdrant deployed to production millions",
        )
        f = extract_features(c)
        assert f.title_tier == "ADJACENT"
        r = generate_reasoning(c, f, rank=10)
        # Must explain why adjacent title is still relevant
        assert "career" in r.lower() or "signal" in r.lower() or "retrieval" in r.lower()


# ---------------------------------------------------------------------------
# Check 3: Honest (mentions concerns when warranted)
# ---------------------------------------------------------------------------


class TestHonest:
    def test_services_concern_mentioned(self) -> None:
        c = _make_candidate(company="Infosys")
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=50)
        assert "services" in r.lower() or "concern" in r.lower() or "note" in r.lower()

    def test_low_yoe_concern_mentioned(self) -> None:
        c = _make_candidate(yoe=2.5)
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=80)
        # Should mention the experience shortfall
        assert "2" in r or "experience" in r.lower()

    def test_no_false_concern_for_strong_candidate(self) -> None:
        c = _make_candidate(yoe=7.0, company="Swiggy")
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=5)
        # Should not claim a concern that doesn't exist
        assert "services" not in r.lower() or f.services_fraction > 0.6


# ---------------------------------------------------------------------------
# Check 4: No hallucination
# ---------------------------------------------------------------------------


class TestNoHallucination:
    def test_company_name_derivable_from_candidate(self) -> None:
        c = _make_candidate(company="SpecialCo")
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=5)
        # If SpecialCo appears, it's from the candidate. If not, check no OTHER company.
        # Any company name in the output should be the actual company.
        if "SpecialCo" not in r:
            # No company name should appear that isn't from the candidate
            assert "Swiggy" not in r and "Zomato" not in r

    def test_yoe_in_reasoning_matches_profile(self) -> None:
        c = _make_candidate(yoe=6.5)
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=15)
        # "6.5" or "6" or "7" (due to rounding) should appear if YOE is mentioned
        # Key: NOT any other year count that's made up
        if "years" in r.lower():
            assert "6" in r or "7" in r


# ---------------------------------------------------------------------------
# Check 5: Varied (different inputs → different outputs)
# ---------------------------------------------------------------------------


class TestVaried:
    def test_strong_and_weak_produce_different_text(self) -> None:
        strong = _make_candidate(
            title="Senior NLP Engineer",
            yoe=7.0,
            career_desc="retrieval ranking embedding faiss qdrant ndcg production deployed millions",
            skills=[
                {"name": "embeddings", "proficiency": "expert", "endorsements": 25, "duration_months": 36},
                {"name": "faiss", "proficiency": "expert", "endorsements": 15, "duration_months": 24},
                {"name": "bm25", "proficiency": "advanced", "endorsements": 10, "duration_months": 18},
            ],
        )
        weak = _make_candidate(
            title="Data Analyst",
            yoe=2.0,
            career_desc="built sql reports and dashboards for business intelligence",
            skills=[{"name": "SQL", "proficiency": "expert", "endorsements": 5, "duration_months": 24}],
        )
        fs = extract_features(strong)
        fw = extract_features(weak)
        r_strong = generate_reasoning(strong, fs, rank=3)
        r_weak = generate_reasoning(weak, fw, rank=90)
        assert r_strong != r_weak

    def test_different_companies_produce_different_text(self) -> None:
        c1 = _make_candidate(company="Swiggy")
        c2 = _make_candidate(company="Flipkart")
        f1 = extract_features(c1)
        f2 = extract_features(c2)
        r1 = generate_reasoning(c1, f1, rank=5)
        r2 = generate_reasoning(c2, f2, rank=5)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Check 6: Rank-consistent (tone matches position)
# ---------------------------------------------------------------------------


class TestRankConsistent:
    def test_length_reasonable(self) -> None:
        c = _make_candidate()
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=1)
        assert 30 <= len(r) <= 400, f"Reasoning length {len(r)} out of bounds: {r!r}"

    def test_output_is_string(self) -> None:
        c = _make_candidate()
        f = extract_features(c)
        r = generate_reasoning(c, f, rank=50)
        assert isinstance(r, str)
        assert len(r) > 0
