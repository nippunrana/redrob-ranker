"""Tests for src/data.py — candidate loading and parsing."""

import json
from pathlib import Path

from src.data import Candidate, load_all, parse_candidate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_RAW: dict = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Test User",
        "headline": "ML Engineer",
        "summary": "I build retrieval systems.",
        "location": "Hyderabad",
        "country": "India",
        "years_of_experience": 6.0,
        "current_title": "ML Engineer",
        "current_company": "Swiggy",
        "current_company_size": "5001-10000",
        "current_industry": "Food Delivery",
    },
    "career_history": [
        {
            "company": "Swiggy",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Food Delivery",
            "company_size": "5001-10000",
            "description": "Built vector search and ranking pipeline.",
        }
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2014,
            "end_year": 2018,
            "grade": "8.5 CGPA",
            "tier": "tier_1",
        }
    ],
    "skills": [
        {
            "name": "Embeddings",
            "proficiency": "expert",
            "endorsements": 40,
            "duration_months": 30,
        }
    ],
    "redrob_signals": {
        "profile_completeness_score": 90.0,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 100,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.85,
        "avg_response_time_hours": 5.0,
        "skill_assessment_scores": {"Embeddings": 88.0},
        "connection_count": 300,
        "endorsements_received": 50,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 30.0, "max": 60.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 70.0,
        "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_candidate_fields() -> None:
    c = parse_candidate(MINIMAL_RAW)
    assert isinstance(c, Candidate)
    assert c.candidate_id == "CAND_0000001"
    assert c.profile.current_title == "ML Engineer"
    assert c.profile.years_of_experience == 6.0
    assert len(c.career_history) == 1
    assert c.career_history[0].company == "Swiggy"
    assert len(c.skills) == 1
    assert c.skills[0].name == "Embeddings"
    assert c.skills[0].endorsements == 40
    assert c.redrob_signals.open_to_work_flag is True
    assert c.redrob_signals.recruiter_response_rate == 0.85


def test_candidate_text_property() -> None:
    c = parse_candidate(MINIMAL_RAW)
    assert "ML Engineer" in c.text
    assert "retrieval" in c.text.lower()
    assert "Embeddings" in c.text


def test_load_all_from_jsonl(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "test.jsonl"
    lines = [
        json.dumps(MINIMAL_RAW),
        json.dumps({**MINIMAL_RAW, "candidate_id": "CAND_0000002"}),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    candidates = load_all(jsonl_file)
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "CAND_0000001"
    assert candidates[1].candidate_id == "CAND_0000002"


def test_missing_optional_fields_dont_crash() -> None:
    """Parsing a candidate with only required fields must not raise."""
    minimal = {
        "candidate_id": "CAND_0099999",
        "profile": {
            "anonymized_name": "X",
            "headline": "",
            "summary": "",
            "location": "",
            "country": "",
            "years_of_experience": 0,
            "current_title": "",
            "current_company": "",
            "current_company_size": "1-10",
            "current_industry": "",
        },
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 0,
            "signup_date": "2024-01-01",
            "last_active_date": "2024-01-01",
            "open_to_work_flag": False,
            "profile_views_received_30d": 0,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0,
            "avg_response_time_hours": 999,
            "skill_assessment_scores": {},
            "connection_count": 0,
            "endorsements_received": 0,
            "notice_period_days": 90,
            "expected_salary_range_inr_lpa": {"min": 0, "max": 0},
            "preferred_work_mode": "flexible",
            "willing_to_relocate": False,
            "github_activity_score": -1,
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0,
            "offer_acceptance_rate": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
        },
    }
    c = parse_candidate(minimal)
    assert c.candidate_id == "CAND_0099999"
    assert c.text == ""  # no content but doesn't crash
