"""Behavioral modifier from redrob_signals (Phase 4).

Returns a multiplicative factor in [0.5, 1.0] that up/down-weights a
candidate based on their platform signals — recency, responsiveness,
open-to-work status, and profile completeness.

Design intent: behavioral signals never dominate fit. A great fit with a
sluggish profile gets ~0.75× score; a weak fit with perfect signals stays
weak. The floor of 0.5 ensures no candidate is zeroed out purely by signals.
"""

from __future__ import annotations

from datetime import date, datetime

from src.data import Candidate

# Reference date — "today" for computing days_since_active.
_TODAY: date = date(2026, 6, 30)


def _days_since(date_str: str) -> int:
    """Parse ISO date string and return days elapsed since _TODAY."""
    try:
        d = datetime.fromisoformat(date_str).date()
        return (_TODAY - d).days
    except (ValueError, TypeError):
        return 365  # conservative fallback if date is missing or malformed


def _recency_score(last_active_date: str) -> float:
    """Score based on how recently the candidate was active on the platform."""
    days = _days_since(last_active_date)
    if days <= 14:
        return 1.0
    if days <= 30:
        return 0.90
    if days <= 90:
        return 0.75
    if days <= 180:
        return 0.55
    return 0.30


def behavioral_modifier(c: Candidate) -> float:
    """Return a multiplicative modifier [0.5, 1.0] from redrob_signals.

    Weights:
        recency        30%  — is the candidate active on the platform?
        open_to_work   25%  — explicitly looking?
        response_rate  25%  — do they respond to recruiters?
        completeness   10%  — is their profile full?
        interview_rate 10%  — do they complete interviews?
    """
    sigs = c.redrob_signals

    recency = _recency_score(sigs.last_active_date)
    owf = 1.0 if sigs.open_to_work_flag else 0.60
    rrr = float(sigs.recruiter_response_rate)  # already [0, 1]

    # completeness_score may be 0-100 or 0-1; normalise defensively
    completeness_raw = float(sigs.profile_completeness_score)
    # completeness_score may be 0-100 or 0-1; normalise defensively.
    if completeness_raw > 1.0:
        completeness = completeness_raw / 100.0
    else:
        completeness = completeness_raw

    interview = float(sigs.interview_completion_rate)  # already [0, 1]

    raw = (
        recency * 0.30
        + owf * 0.25
        + rrr * 0.25
        + completeness * 0.10
        + interview * 0.10
    )

    # Scale to [0.5, 1.0] so signals modify but never override fit.
    return 0.5 + 0.5 * raw
