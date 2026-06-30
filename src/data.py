"""Candidate data loading and typed access.

Loads candidates from a JSONL file (or gzipped JSONL) one-by-one to keep
memory usage bounded — the full file is ~465 MB uncompressed.

All field access goes through typed dataclasses so the rest of the pipeline
gets IDE completion and mypy coverage without needing to inspect JSON keys
throughout the codebase.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed sub-structures
# ---------------------------------------------------------------------------


@dataclass
class JobEntry:
    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None
    tier: str  # "tier_1" … "tier_4" or "unknown"


@dataclass
class Skill:
    name: str
    proficiency: str  # "beginner" | "intermediate" | "advanced" | "expert"
    endorsements: int
    duration_months: int


@dataclass
class SalaryRange:
    min: float
    max: float


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str  # "remote"|"hybrid"|"onsite"|"flexible"
    willing_to_relocate: bool
    github_activity_score: float  # -1 if no GitHub
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float  # -1 if no offer history
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass
class Candidate:
    candidate_id: str
    profile: Profile
    career_history: list[JobEntry]
    education: list[Education]
    skills: list[Skill]
    redrob_signals: RedrobSignals

    # Convenience properties
    @property
    def text(self) -> str:
        """Concatenated text used for embedding and BM25 indexing."""
        parts = [
            self.profile.headline,
            self.profile.summary,
        ]
        for job in self.career_history:
            parts.append(f"{job.title} at {job.company}: {job.description}")
        for skill in self.skills:
            parts.append(skill.name)
        return " ".join(parts)

    @property
    def skills_text(self) -> str:
        return " ".join(s.name for s in self.skills)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_job(raw: dict) -> JobEntry:
    return JobEntry(
        company=raw.get("company", ""),
        title=raw.get("title", ""),
        start_date=raw.get("start_date", ""),
        end_date=raw.get("end_date"),
        duration_months=int(raw.get("duration_months", 0)),
        is_current=bool(raw.get("is_current", False)),
        industry=raw.get("industry", ""),
        company_size=raw.get("company_size", ""),
        description=raw.get("description", ""),
    )


def _parse_education(raw: dict) -> Education:
    return Education(
        institution=raw.get("institution", ""),
        degree=raw.get("degree", ""),
        field_of_study=raw.get("field_of_study", ""),
        start_year=int(raw.get("start_year", 0)),
        end_year=int(raw.get("end_year", 0)),
        grade=raw.get("grade"),
        tier=raw.get("tier", "unknown"),
    )


def _parse_skill(raw: dict) -> Skill:
    return Skill(
        name=raw.get("name", ""),
        proficiency=raw.get("proficiency", "beginner"),
        endorsements=int(raw.get("endorsements", 0)),
        duration_months=int(raw.get("duration_months", 0)),
    )


def _parse_signals(raw: dict) -> RedrobSignals:
    salary_raw = raw.get("expected_salary_range_inr_lpa", {})
    return RedrobSignals(
        profile_completeness_score=float(raw.get("profile_completeness_score", 0)),
        signup_date=raw.get("signup_date", ""),
        last_active_date=raw.get("last_active_date", ""),
        open_to_work_flag=bool(raw.get("open_to_work_flag", False)),
        profile_views_received_30d=int(raw.get("profile_views_received_30d", 0)),
        applications_submitted_30d=int(raw.get("applications_submitted_30d", 0)),
        recruiter_response_rate=float(raw.get("recruiter_response_rate", 0)),
        avg_response_time_hours=float(raw.get("avg_response_time_hours", 999)),
        skill_assessment_scores=dict(raw.get("skill_assessment_scores", {})),
        connection_count=int(raw.get("connection_count", 0)),
        endorsements_received=int(raw.get("endorsements_received", 0)),
        notice_period_days=int(raw.get("notice_period_days", 90)),
        expected_salary_range_inr_lpa=SalaryRange(
            min=float(salary_raw.get("min", 0)),
            max=float(salary_raw.get("max", 0)),
        ),
        preferred_work_mode=raw.get("preferred_work_mode", "flexible"),
        willing_to_relocate=bool(raw.get("willing_to_relocate", False)),
        github_activity_score=float(raw.get("github_activity_score", -1)),
        search_appearance_30d=int(raw.get("search_appearance_30d", 0)),
        saved_by_recruiters_30d=int(raw.get("saved_by_recruiters_30d", 0)),
        interview_completion_rate=float(raw.get("interview_completion_rate", 0)),
        offer_acceptance_rate=float(raw.get("offer_acceptance_rate", -1)),
        verified_email=bool(raw.get("verified_email", False)),
        verified_phone=bool(raw.get("verified_phone", False)),
        linkedin_connected=bool(raw.get("linkedin_connected", False)),
    )


def _parse_profile(raw: dict) -> Profile:
    return Profile(
        anonymized_name=raw.get("anonymized_name", ""),
        headline=raw.get("headline", ""),
        summary=raw.get("summary", ""),
        location=raw.get("location", ""),
        country=raw.get("country", ""),
        years_of_experience=float(raw.get("years_of_experience", 0)),
        current_title=raw.get("current_title", ""),
        current_company=raw.get("current_company", ""),
        current_company_size=raw.get("current_company_size", ""),
        current_industry=raw.get("current_industry", ""),
    )


def parse_candidate(raw: dict) -> Candidate:
    """Parse one raw JSON dict into a typed Candidate."""
    return Candidate(
        candidate_id=raw["candidate_id"],
        profile=_parse_profile(raw.get("profile", {})),
        career_history=[_parse_job(j) for j in raw.get("career_history", [])],
        education=[_parse_education(e) for e in raw.get("education", [])],
        skills=[_parse_skill(s) for s in raw.get("skills", [])],
        redrob_signals=_parse_signals(raw.get("redrob_signals", {})),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_candidates(path: Path) -> Iterator["Candidate"]:
    """Yield Candidate objects one-by-one from a JSONL or gzipped JSONL file.

    Keeps memory usage bounded — suitable for 100K × 465 MB files.
    """

    opener = gzip.open if path.suffix == ".gz" else open
    log.info("Streaming candidates from %s", path)
    n = 0
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield parse_candidate(json.loads(line))
            n += 1
    log.info("Loaded %d candidates from %s", n, path)


def load_all(path: Path) -> list[Candidate]:
    """Load all candidates into memory. Use only when the full list is needed."""
    return list(iter_candidates(path))
