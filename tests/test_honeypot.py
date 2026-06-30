"""Tests for src/honeypot.py (Phase 5 guard).

Verification goals:
  - CAND_0000031 archetype passes (multiplier = 1.0).
  - Clear Pattern D hits (YOE >> actual career) → confirmed honeypot.
  - Clear Pattern F hits (career before graduation) → confirmed honeypot.
  - Double-hit → hard zero. Single-hit → 0.50 penalty. No hit → 1.0.
"""

from __future__ import annotations

from src.data import parse_candidate
from src.honeypot import honeypot_multiplier, is_confirmed_honeypot

# ---------------------------------------------------------------------------
# Shared builder
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


def _make(
    candidate_id: str = "CAND_TEST",
    title: str = "ML Engineer",
    yoe: float = 7.0,
    career: list[dict] | None = None,
    education: list[dict] | None = None,
) -> object:
    if career is None:
        career = [
            {
                "company": "Swiggy",
                "title": title,
                "start_date": "2018-01-01",
                "end_date": None,
                "duration_months": int(yoe * 12),
                "is_current": True,
                "industry": "Technology",
                "company_size": "1001-5000",
                "description": "Built ranking systems",
            }
        ]
    if education is None:
        education = [
            {
                "institution": "IIT Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2010,
                "end_year": 2014,
                "grade": None,
                "tier": "tier_1",
            }
        ]
    raw = {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test",
            "headline": f"{title}",
            "summary": "Engineer",
            "location": "Hyderabad",
            "country": "India",
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": "Swiggy",
            "current_company_size": "1001-5000",
            "current_industry": "Technology",
        },
        "career_history": career,
        "education": education,
        "skills": [],
        "redrob_signals": _BASE_SIGNALS,
    }
    return parse_candidate(raw)


# ---------------------------------------------------------------------------
# Archetype must pass
# ---------------------------------------------------------------------------


class TestArchetypeSafe:
    def test_archetype_surrogate_clean(self) -> None:
        """CAND_0000031 surrogate: yoe=6.0, career from 2020. Must be clean."""
        c = _make(
            candidate_id="CAND_0000031",
            yoe=6.0,
            career=[
                {
                    "company": "Swiggy",
                    "title": "Recommendation Systems Engineer",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "duration_months": 14,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "1001-5000",
                    "description": "led embedding retrieval",
                },
                {
                    "company": "Mad Street Den",
                    "title": "Search Engineer",
                    "start_date": "2018-09-01",
                    "end_date": "2020-01-01",
                    "duration_months": 16,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "11-50",
                    "description": "search ranking",
                },
                {
                    "company": "Uber",
                    "title": "NLP Engineer",
                    "start_date": "2016-06-01",
                    "end_date": "2018-09-01",
                    "duration_months": 27,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "10000+",
                    "description": "nlp models",
                },
                {
                    "company": "Zomato",
                    "title": "Applied ML Engineer",
                    "start_date": "2015-05-01",
                    "end_date": "2016-06-01",
                    "duration_months": 13,
                    "is_current": False,
                    "industry": "Technology",
                    "company_size": "1001-5000",
                    "description": "ml pipeline",
                },
            ],
            education=[
                {
                    "institution": "NIT Trichy",
                    "degree": "M.Tech",
                    "field_of_study": "CS",
                    "start_year": 2002,
                    "end_year": 2006,
                    "grade": None,
                    "tier": "tier_1",
                }
            ],
        )
        mult = honeypot_multiplier(c)
        assert mult == 1.0, f"Archetype should be clean, got multiplier={mult}"
        assert not is_confirmed_honeypot(c)


# ---------------------------------------------------------------------------
# Pattern D: YOE inflation
# ---------------------------------------------------------------------------


class TestPatternD:
    def test_normal_yoe_is_clean(self) -> None:
        # yoe=7, career starts 2019 → actual ≈ 7.5 years → no flag
        c = _make(yoe=7.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2019-01-01",
             "end_date": None, "duration_months": 90, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ])
        assert honeypot_multiplier(c) == 1.0

    def test_slightly_inflated_yoe_gets_penalty(self) -> None:
        # yoe=10 (120m), career starts 2022 → actual ≈ 4.5y (54m) → 120-54=66 > 36
        c = _make(yoe=10.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 54, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ], education=[
            {"institution": "MIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2010, "end_year": 2014, "grade": None, "tier": "tier_1"}
        ])
        # Pattern D fires (66m inflation), Pattern F: impl_start=2016, grad=2014, 2016>2014-2=2012 → OK
        # So only D fires → 0.50 penalty
        mult = honeypot_multiplier(c)
        assert mult == 0.50

    def test_extreme_yoe_inflation_double_fires(self) -> None:
        # yoe=16 (192m), career starts 2023 → actual ≈ 3.5y (42m) → 192-42=150 >> 36
        # edu end=2022 → impl_start=2026-16=2010 < 2022-2=2020 → F also fires
        c = _make(yoe=16.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2023-01-01",
             "end_date": None, "duration_months": 42, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ], education=[
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2018, "end_year": 2022, "grade": None, "tier": "tier_1"}
        ])
        mult = honeypot_multiplier(c)
        assert mult == 0.0, f"Expected hard zero, got {mult}"
        assert is_confirmed_honeypot(c)


# ---------------------------------------------------------------------------
# Pattern F: Education timeline
# ---------------------------------------------------------------------------


class TestPatternF:
    def test_normal_edu_timeline(self) -> None:
        # yoe=6, grad 2018 → impl_start=2020, which is > 2018-2=2016 → OK
        c = _make(yoe=6.0, education=[
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2014, "end_year": 2018, "grade": None, "tier": "tier_1"}
        ])
        assert honeypot_multiplier(c) == 1.0

    def test_borderline_edu_timeline_allowed(self) -> None:
        # yoe=8, grad 2019 → impl_start=2018, which is 2019-1=2018 → just at boundary
        # 2018 < 2019 - 2 = 2017? No! 2018 is NOT < 2017 → clean
        c = _make(yoe=8.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2016-01-01",
             "end_date": None, "duration_months": 126, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ], education=[
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2015, "end_year": 2019, "grade": None, "tier": "tier_1"}
        ])
        # Pattern D: claimed=96m, actual from 2016=126m → no flag
        # Pattern F: impl_start=2026-8=2018, grad=2019, 2018 < 2019-2=2017? No (2018>2017) → clean
        assert honeypot_multiplier(c) == 1.0

    def test_clear_edu_violation_double_fire(self) -> None:
        # yoe=15 (180m), grad 2020 → impl_start=2011, which < 2020-2=2018 → F fires
        # career from 2022 → actual=48m, 180>48+36=84 → D fires
        c = _make(yoe=15.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 48, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ], education=[
            {"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
             "start_year": 2016, "end_year": 2020, "grade": None, "tier": "tier_1"}
        ])
        assert honeypot_multiplier(c) == 0.0
        assert is_confirmed_honeypot(c)


# ---------------------------------------------------------------------------
# Clean baseline
# ---------------------------------------------------------------------------


class TestCleanCandidates:
    def test_no_career_history_is_clean(self) -> None:
        # Cannot trigger Pattern D without career history
        c = _make(yoe=5.0, career=[])
        assert honeypot_multiplier(c) == 1.0

    def test_no_education_is_clean(self) -> None:
        # Cannot trigger Pattern F without education
        c = _make(yoe=15.0, career=[
            {"company": "A", "title": "ML Engineer", "start_date": "2022-01-01",
             "end_date": None, "duration_months": 48, "is_current": True,
             "industry": "Tech", "company_size": "1001-5000", "description": "ML"}
        ], education=[])
        # Only D fires → 0.50 (not hard zero since F needs education)
        assert honeypot_multiplier(c) == 0.50
