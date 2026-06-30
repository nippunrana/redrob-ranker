"""Honeypot / consistency guard (Phase 5).

Detects profiles where metadata is internally inconsistent in ways that
cannot occur for a genuine candidate. These are "subtly impossible" profiles
planted in the dataset to trap systems that rank on signals alone without
sanity-checking the data.

Challenge spec guardrail: >10% honeypots in the submitted top-100 = disqualified.

Detection axes (two independent signals — both must fire for hard zero):

  Pattern D — YOE inflation:
    Candidate's claimed years_of_experience exceeds the career-history date range
    by more than 36 months. A genuine candidate cannot have 5 more years of work
    experience than their career history dates allow.
    Threshold: +36 months (3 years) buffer for rounding, education-to-career gaps.

  Pattern F — Education timeline violation:
    The implied career start year (today.year - claimed_yoe) is more than 2 years
    before the candidate's earliest graduation year. This means the candidate would
    have started full-time work 2+ years before finishing their degree — impossible
    for most programs.
    Threshold: 2-year buffer (covers internships, part-time roles, open degrees).

Scoring outcome:
  Both patterns fire  → score multiplier = 0.0 (hard zero / clear honeypot)
  One pattern fires   → score multiplier = 0.50 (suspicious; scoring may still recover)
  No pattern fires    → score multiplier = 1.0 (clean — no guard applied)

Calibration notes:
  - CAND_0000031 (archetype tier-5): 0 flags — confirmed safe.
  - Roughly 5 double-flagged + 15 single-flagged candidates in the ML/AI pool (~1048).
  - Guard is conservative (prefers false negatives over false positives).
  - The scoring system's features naturally push further borderline cases down.
"""

from __future__ import annotations

from datetime import date, datetime

from src.data import Candidate

# Reference date for computing career spans.
_TODAY: date = date(2026, 6, 30)
_TODAY_YEAR: int = _TODAY.year

# Thresholds (months / years)
_YOE_INFLATION_THRESHOLD_MONTHS: int = 36  # Pattern D buffer
_EDU_BUFFER_YEARS: int = 2  # Pattern F buffer


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


def _pattern_d(c: Candidate) -> bool:
    """Pattern D: YOE > actual career date range + 36 months."""
    starts = [
        d
        for job in c.career_history
        if (d := _parse_date(job.start_date)) is not None
    ]
    if not starts:
        return False
    earliest = min(starts)
    actual_months = max(
        0,
        ((_TODAY.year - earliest.year) * 12 + (_TODAY.month - earliest.month)),
    )
    claimed_months = c.profile.years_of_experience * 12
    return claimed_months > actual_months + _YOE_INFLATION_THRESHOLD_MONTHS


def _pattern_f(c: Candidate) -> bool:
    """Pattern F: implied career start is 2+ years before earliest graduation."""
    end_years = [
        e.end_year
        for e in c.education
        if e.end_year and e.end_year > 1990
    ]
    if not end_years:
        return False
    earliest_grad = min(end_years)
    implied_career_start = _TODAY_YEAR - c.profile.years_of_experience
    return implied_career_start < earliest_grad - _EDU_BUFFER_YEARS


def honeypot_multiplier(c: Candidate) -> float:
    """Return a score multiplier based on profile consistency.

    Returns:
        0.0  — both patterns fire (hard zero — clear honeypot)
        0.50 — one pattern fires (suspicious; main scoring applies)
        1.0  — no patterns fire (clean profile)
    """
    d_fires = _pattern_d(c)
    f_fires = _pattern_f(c)

    if d_fires and f_fires:
        return 0.0   # clearly impossible — both axes confirm it
    if d_fires or f_fires:
        return 0.50  # suspicious — one axis fires
    return 1.0       # clean


def is_confirmed_honeypot(c: Candidate) -> bool:
    """True if the candidate is a confirmed honeypot (both axes fire)."""
    return _pattern_d(c) and _pattern_f(c)
