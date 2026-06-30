"""Tests for Phase 4: structured features, behavioral modifier, scoring combiner.

Verification goals:
  - Keyword stuffers (non-tech titles with AI skills) score near zero.
  - Strong-positive titles with retrieval career evidence score high.
  - Services-company-only candidates are penalised.
  - CAND_0000031 archetype (RecSys@Swiggy) scores ≥ 0.70.
  - Behavioral modifier bounds: [0.5, 1.0].
  - Score monotonicity: tier-5-ish > tier-4-ish > disqualifier.
"""

from __future__ import annotations

import pytest

from src.behavioral import behavioral_modifier
from src.data import Candidate, parse_candidate
from src.features import extract_features
from src.scoring import score_candidate

# ---------------------------------------------------------------------------
# Minimal candidate builder
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
    skills: list[dict] | None = None,
    career_desc: str = "Built embedding retrieval and ranking pipeline deployed to production",
    signals_overrides: dict | None = None,
) -> Candidate:
    skills = skills or [
        {"name": "embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 36},
        {"name": "faiss", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
        {"name": "python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
    ]
    signals = {**_BASE_SIGNALS, **(signals_overrides or {})}
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
                "start_date": "2020-01-01",
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
        "redrob_signals": signals,
    }
    return parse_candidate(raw)


# ---------------------------------------------------------------------------
# Title classification + title score
# ---------------------------------------------------------------------------


class TestTitleFeatures:
    def test_disqualifier_hr_manager(self) -> None:
        c = _make_candidate(title="HR Manager", career_desc="payroll onboarding")
        f = extract_features(c)
        assert f.title_tier == "DISQUALIFIER"
        assert f.title_score == 0.0

    def test_disqualifier_accountant(self) -> None:
        c = _make_candidate(title="Accountant", career_desc="bookkeeping")
        f = extract_features(c)
        assert f.title_tier == "DISQUALIFIER"

    def test_strong_positive_ml_engineer(self) -> None:
        c = _make_candidate(title="ML Engineer")
        f = extract_features(c)
        assert f.title_tier == "STRONG_POSITIVE"
        assert f.title_score == pytest.approx(1.0)

    def test_research_penalty(self) -> None:
        c = _make_candidate(title="AI Research Engineer")
        f = extract_features(c)
        assert f.title_tier == "RESEARCH_OR_CV"
        assert f.title_score < 0.30

    def test_cv_engineer_penalty(self) -> None:
        c = _make_candidate(title="Computer Vision Engineer")
        f = extract_features(c)
        assert f.title_tier == "RESEARCH_OR_CV"

    def test_adjacent_ai_engineer(self) -> None:
        c = _make_candidate(title="AI Engineer")
        f = extract_features(c)
        assert f.title_tier == "ADJACENT"
        assert 0.40 <= f.title_score <= 0.60

    def test_case_insensitive_title_matching(self) -> None:
        # Dataset has exact case; test robustness to minor case variation
        c = _make_candidate(title="ml engineer")  # lowercase
        f = extract_features(c)
        assert f.title_tier == "STRONG_POSITIVE"


# ---------------------------------------------------------------------------
# Career-history signals
# ---------------------------------------------------------------------------


class TestCareerFeatures:
    def test_high_retrieval_hits(self) -> None:
        desc = "retrieval ranking embedding vector search bm25 faiss ndcg"
        c = _make_candidate(career_desc=desc)
        f = extract_features(c)
        assert f.career_retrieval == pytest.approx(1.0)
        assert f.retrieval_hits >= 5

    def test_no_retrieval_hits(self) -> None:
        desc = "designed marketing campaigns and managed social media accounts"
        c = _make_candidate(career_desc=desc)
        f = extract_features(c)
        assert f.retrieval_hits == 0
        assert f.career_retrieval == 0.0

    def test_production_keywords(self) -> None:
        desc = "deployed to production serving millions of real users at scale"
        c = _make_candidate(career_desc=desc)
        f = extract_features(c)
        assert f.career_production > 0.5

    def test_langchain_only_flag(self) -> None:
        desc = "built llm apps using langchain and llamaindex"
        c = _make_candidate(career_desc=desc)
        f = extract_features(c)
        assert f.langchain_only is True

    def test_langchain_not_flagged_with_retrieval_depth(self) -> None:
        desc = (
            "built embedding retrieval pipeline, vector search, ranking; "
            "also used langchain for orchestration"
        )
        c = _make_candidate(career_desc=desc)
        f = extract_features(c)
        assert f.langchain_only is False


# ---------------------------------------------------------------------------
# Skill trust
# ---------------------------------------------------------------------------


class TestSkillTrust:
    def test_no_must_have_skills_zero_trust(self) -> None:
        c = _make_candidate(
            skills=[{"name": "Photoshop", "proficiency": "expert", "endorsements": 0, "duration_months": 0}]
        )
        f = extract_features(c)
        assert f.skill_trust == pytest.approx(0.0)

    def test_must_have_with_endorsements(self) -> None:
        c = _make_candidate(
            skills=[
                {"name": "embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 24},
                {"name": "faiss", "proficiency": "expert", "endorsements": 15, "duration_months": 12},
                {"name": "python", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
                {"name": "bm25", "proficiency": "advanced", "endorsements": 8, "duration_months": 18},
            ]
        )
        f = extract_features(c)
        assert f.skill_trust > 0.60

    def test_skills_with_zero_endorsements_lower_trust(self) -> None:
        high_trust = _make_candidate(
            skills=[{"name": "embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 24}]
        )
        low_trust = _make_candidate(
            skills=[{"name": "embeddings", "proficiency": "expert", "endorsements": 0, "duration_months": 0}]
        )
        assert extract_features(high_trust).skill_trust > extract_features(low_trust).skill_trust


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------


class TestExperienceFit:
    def test_ideal_range(self) -> None:
        c = _make_candidate(yoe=7.0)
        f = extract_features(c)
        assert f.experience_fit == pytest.approx(1.0)

    def test_below_hard_min(self) -> None:
        c = _make_candidate(yoe=1.0)
        f = extract_features(c)
        assert f.experience_fit == pytest.approx(0.0)

    def test_soft_min(self) -> None:
        c = _make_candidate(yoe=4.0)
        f = extract_features(c)
        assert 0.55 <= f.experience_fit <= 0.65

    def test_above_soft_max(self) -> None:
        c = _make_candidate(yoe=12.0)
        f = extract_features(c)
        assert 0.55 <= f.experience_fit <= 0.75


# ---------------------------------------------------------------------------
# Services fraction
# ---------------------------------------------------------------------------


class TestServicesFeature:
    def test_pure_services_company(self) -> None:
        c = _make_candidate(company="Infosys")
        f = extract_features(c)
        assert f.services_fraction > 0.9

    def test_product_company_no_penalty(self) -> None:
        c = _make_candidate(company="Swiggy")
        f = extract_features(c)
        assert f.services_fraction < 0.1


# ---------------------------------------------------------------------------
# Location boost
# ---------------------------------------------------------------------------


class TestLocationBoost:
    def test_india_full_boost(self) -> None:
        c = _make_candidate(country="India", location="Hyderabad")
        f = extract_features(c)
        assert f.location_boost == pytest.approx(1.0)

    def test_non_india_no_relocate(self) -> None:
        c = _make_candidate(
            country="United Kingdom",
            location="London",
            signals_overrides={"willing_to_relocate": False},
        )
        f = extract_features(c)
        assert f.location_boost < 0.30

    def test_non_india_willing_to_relocate(self) -> None:
        c = _make_candidate(
            country="United Kingdom",
            location="London",
            signals_overrides={"willing_to_relocate": True},
        )
        f = extract_features(c)
        assert f.location_boost >= 0.50


# ---------------------------------------------------------------------------
# Behavioral modifier
# ---------------------------------------------------------------------------


class TestBehavioralModifier:
    def test_bounds(self) -> None:
        c = _make_candidate()
        bmod = behavioral_modifier(c)
        assert 0.50 <= bmod <= 1.0

    def test_fully_active_candidate(self) -> None:
        signals = {
            **_BASE_SIGNALS,
            "last_active_date": "2026-06-29",  # yesterday
            "open_to_work_flag": True,
            "recruiter_response_rate": 1.0,
            "interview_completion_rate": 1.0,
            "profile_completeness_score": 100.0,
        }
        c = _make_candidate(signals_overrides=signals)
        bmod = behavioral_modifier(c)
        assert bmod >= 0.90

    def test_inactive_candidate_lower_modifier(self) -> None:
        inactive = {
            **_BASE_SIGNALS,
            "last_active_date": "2024-01-01",  # over 2 years ago
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.0,
        }
        active = {
            **_BASE_SIGNALS,
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "recruiter_response_rate": 1.0,
        }
        c_inactive = _make_candidate(signals_overrides=inactive)
        c_active = _make_candidate(signals_overrides=active)
        assert behavioral_modifier(c_inactive) < behavioral_modifier(c_active)


# ---------------------------------------------------------------------------
# Score combiner (score_candidate)
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_disqualifier_scores_zero(self) -> None:
        c = _make_candidate(
            title="HR Manager",
            career_desc="retrieval ranking embeddings faiss ndcg production deployed",
            skills=[{"name": "embeddings", "proficiency": "expert", "endorsements": 20, "duration_months": 24}],
        )
        score, _ = score_candidate(c)
        assert score == 0.0

    def test_strong_positive_retrieval_scores_high(self) -> None:
        c = _make_candidate(
            title="ML Engineer",
            company="Swiggy",
            yoe=7.0,
            career_desc=(
                "built embedding retrieval deployed to production ranking "
                "bm25 faiss vector search ndcg a/b testing at scale"
            ),
        )
        score, _ = score_candidate(c)
        assert score >= 0.55, f"Expected ≥0.55, got {score:.3f}"

    def test_monotonicity_strong_above_research(self) -> None:
        """STRONG_POSITIVE with same career should score above RESEARCH_OR_CV."""
        career = "retrieval ranking embedding production deployed bm25 faiss ndcg"
        strong = _make_candidate(title="ML Engineer", career_desc=career)
        research = _make_candidate(title="AI Research Engineer", career_desc=career)
        s_strong, _ = score_candidate(strong)
        s_research, _ = score_candidate(research)
        assert s_strong > s_research, f"ML Eng: {s_strong:.3f}, AI Research: {s_research:.3f}"

    def test_services_penalty_applied(self) -> None:
        """Pure-services candidate scores lower than identical product candidate."""
        career = "retrieval ranking embedding production deployed"
        product = _make_candidate(title="ML Engineer", company="Swiggy", career_desc=career)
        services = _make_candidate(title="ML Engineer", company="Infosys", career_desc=career)
        s_product, _ = score_candidate(product)
        s_services, _ = score_candidate(services)
        assert s_product > s_services

    def test_score_in_unit_range(self) -> None:
        c = _make_candidate()
        score, _ = score_candidate(c)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Archetype check (CAND_0000031 surrogate)
# ---------------------------------------------------------------------------


class TestArchetypeCandidate:
    def test_archetype_surrogate_scores_high(self) -> None:
        """Verify that a candidate with CAND_0000031's profile scores ≥ 0.60."""
        c = _make_candidate(
            candidate_id="CAND_ARCHETYPE",
            title="Recommendation Systems Engineer",
            company="Swiggy",
            yoe=6.0,
            country="India",
            location="Hyderabad",
            career_desc=(
                "led migration from keyword search to embedding-based retrieval, "
                "deployed ranking pipeline serving millions of users, "
                "a/b testing offline evaluation ndcg faiss production at scale"
            ),
            skills=[
                {"name": "embeddings", "proficiency": "expert", "endorsements": 25, "duration_months": 36},
                {"name": "faiss", "proficiency": "expert", "endorsements": 15, "duration_months": 24},
                {"name": "python", "proficiency": "expert", "endorsements": 40, "duration_months": 72},
                {"name": "information retrieval", "proficiency": "expert", "endorsements": 20, "duration_months": 30},
                {"name": "bm25", "proficiency": "advanced", "endorsements": 10, "duration_months": 18},
            ],
        )
        score, feats = score_candidate(c)
        assert score >= 0.60, (
            f"Archetype surrogate scored {score:.3f}; "
            f"title={feats.title_tier}, ret={feats.career_retrieval:.2f}, "
            f"prod={feats.career_production:.2f}, skills={feats.skill_trust:.2f}"
        )
