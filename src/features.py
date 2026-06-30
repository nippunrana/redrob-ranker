"""Structured JD-grounding features for the precision layer (Phase 4).

Each feature is a [0, 1] float unless noted. Together they encode *what the JD
actually means* rather than keyword similarity.

Key design:
- Title tier determines title_score AND gates disqualifiers at 0 in scoring.py.
- Career keywords capture "actually built" retrieval/ranking — stuffers can't fake this.
- Skill trust is weighted by endorsements + duration, not mere presence.
- Services fraction penalises pure-consulting careers (JD: "not at TCS/Wipro").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data import Candidate
from src.jd import (
    CAREER_PRODUCTION_KEYWORDS,
    CAREER_RETRIEVAL_KEYWORDS,
    EXPERIENCE_HARD_MIN,
    EXPERIENCE_IDEAL_MAX,
    EXPERIENCE_IDEAL_MIN,
    EXPERIENCE_SOFT_MAX,
    EXPERIENCE_SOFT_MIN,
    LANGCHAIN_ONLY_KEYWORDS,
    LOCATIONS_NON_INDIA,
    LOCATIONS_PREFERRED,
    SERVICES_COMPANIES,
    SKILLS_CV_SPEECH,
    SKILLS_MUST_HAVE,
    TITLE_ADJACENT_TECH,
    TITLE_DISQUALIFIER,
    TITLE_RESEARCH_OR_CV,
    TITLE_STRONG_POSITIVE,
)

# Pre-built lowercase versions for case-insensitive matching.
_STRONG_LOWER = frozenset(t.lower() for t in TITLE_STRONG_POSITIVE)
_RESEARCH_LOWER = frozenset(t.lower() for t in TITLE_RESEARCH_OR_CV)
_ADJACENT_LOWER = frozenset(t.lower() for t in TITLE_ADJACENT_TECH)
_DISQ_LOWER = frozenset(t.lower() for t in TITLE_DISQUALIFIER)
_SKILLS_MUST_LOWER = frozenset(s.lower() for s in SKILLS_MUST_HAVE)
_SKILLS_CV_LOWER = frozenset(s.lower() for s in SKILLS_CV_SPEECH)

# Title → numeric score (before gating)
_TITLE_TIER_SCORE: dict[str, float] = {
    "STRONG_POSITIVE": 1.00,
    "ADJACENT": 0.50,
    "UNKNOWN": 0.35,
    "RESEARCH_OR_CV": 0.20,
    "DISQUALIFIER": 0.00,  # gated to 0 in scoring.py
}


@dataclass
class Features:
    """Extracted per-candidate features for scoring and reasoning."""

    # Title tier: STRONG_POSITIVE | ADJACENT | RESEARCH_OR_CV | DISQUALIFIER | UNKNOWN
    title_tier: str
    title_score: float

    # Career evidence [0, 1]
    career_retrieval: float
    career_production: float

    # Skill depth [0, 1]
    skill_trust: float
    cv_speech_fraction: float  # penalise CV/speech-heavy profiles

    # Contextual fit [0, 1]
    experience_fit: float
    location_boost: float

    # Red-flag signals
    services_fraction: float
    langchain_only: bool

    # Raw counts and matches for reasoning
    retrieval_hits: int
    production_hits: int
    matched_skills: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Title classification
# ---------------------------------------------------------------------------


def _classify_title(title: str) -> str:
    t = title.lower().strip()
    # Check in priority order: DISQ first (prevents stuffer titles from leaking)
    if t in _DISQ_LOWER:
        return "DISQUALIFIER"
    if t in _STRONG_LOWER:
        return "STRONG_POSITIVE"
    if t in _RESEARCH_LOWER:
        return "RESEARCH_OR_CV"
    if t in _ADJACENT_LOWER:
        return "ADJACENT"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Career-history signals
# ---------------------------------------------------------------------------


def _build_career_text(c: Candidate) -> str:
    """All career history text lowercased — searched for keyword hits."""
    parts = []
    for job in c.career_history:
        parts.append(f"{job.title} {job.description}")
    return " ".join(parts).lower()


def _retrieval_hits(career_text: str) -> int:
    return sum(1 for kw in CAREER_RETRIEVAL_KEYWORDS if kw in career_text)


def _production_hits(career_text: str) -> int:
    return sum(1 for kw in CAREER_PRODUCTION_KEYWORDS if kw in career_text)


def _services_fraction(c: Candidate) -> float:
    """Fraction of career months spent at services companies."""
    services_months = sum(
        job.duration_months
        for job in c.career_history
        if any(svc in job.company.lower() for svc in SERVICES_COMPANIES)
    )
    total_months = sum(job.duration_months for job in c.career_history)
    if total_months <= 0:
        return 0.0
    return min(1.0, services_months / total_months)


def _langchain_only(c: Candidate, retrieval_hit_count: int) -> bool:
    """True if most recent job is LangChain-wrapper only with no retrieval depth."""
    if retrieval_hit_count >= 3:
        return False  # enough retrieval depth elsewhere
    if not c.career_history:
        return False
    recent_desc = c.career_history[0].description.lower()
    return any(kw in recent_desc for kw in LANGCHAIN_ONLY_KEYWORDS)


# ---------------------------------------------------------------------------
# Skill trust
# ---------------------------------------------------------------------------


def _skill_trust(c: Candidate) -> tuple[float, list[str]]:
    """Return (trust_score [0,1], matched_skill_names)."""
    matched = [s for s in c.skills if s.name.lower() in _SKILLS_MUST_LOWER]
    if not matched:
        return 0.0, []

    trust_per_skill = []
    for s in matched:
        endorsement_factor = min(1.0, s.endorsements / 10.0)
        duration_factor = min(1.0, s.duration_months / 12.0)
        trust_per_skill.append(0.5 * endorsement_factor + 0.5 * duration_factor)

    avg_trust = sum(trust_per_skill) / len(trust_per_skill)
    coverage = min(1.0, len(matched) / 4.0)  # 4 matched skills = full coverage
    trust_score = coverage * (0.4 + 0.6 * avg_trust)

    matched_names = sorted({s.name for s in matched})
    return trust_score, matched_names


def _cv_speech_fraction(c: Candidate) -> float:
    """Fraction of skills that are CV/speech/robotics (penalise CV-heavy profiles)."""
    if not c.skills:
        return 0.0
    cv_count = sum(1 for s in c.skills if s.name.lower() in _SKILLS_CV_LOWER)
    return cv_count / len(c.skills)


# ---------------------------------------------------------------------------
# Experience fit
# ---------------------------------------------------------------------------


def _experience_fit(yoe: float) -> float:
    if yoe < EXPERIENCE_HARD_MIN:
        return 0.0
    if yoe < EXPERIENCE_SOFT_MIN:
        # Ramp 0.2 → 0.6 from hard_min to soft_min
        t = (yoe - EXPERIENCE_HARD_MIN) / (EXPERIENCE_SOFT_MIN - EXPERIENCE_HARD_MIN)
        return 0.2 + 0.4 * t
    if yoe < EXPERIENCE_IDEAL_MIN:
        # Ramp 0.6 → 0.9 from soft_min to ideal_min
        t = (yoe - EXPERIENCE_SOFT_MIN) / (EXPERIENCE_IDEAL_MIN - EXPERIENCE_SOFT_MIN)
        return 0.6 + 0.3 * t
    if yoe <= EXPERIENCE_IDEAL_MAX:
        return 1.0  # ideal zone 6-8 years
    if yoe <= EXPERIENCE_SOFT_MAX:
        return 0.85  # slightly above ideal
    return 0.65  # over 10 years — over-seniorised for founding-team role


# ---------------------------------------------------------------------------
# Location boost
# ---------------------------------------------------------------------------


def _location_boost(c: Candidate) -> float:
    location_text = f"{c.profile.location} {c.profile.country}".lower()
    # Direct India signal
    in_india = "india" in location_text
    in_preferred = any(loc in location_text for loc in LOCATIONS_PREFERRED)
    if in_india or in_preferred:
        return 1.0
    # Known non-India with no relocation intent
    is_non_india = any(loc in location_text for loc in LOCATIONS_NON_INDIA)
    if is_non_india and not c.redrob_signals.willing_to_relocate:
        return 0.15  # JD: no visa sponsorship
    if c.redrob_signals.willing_to_relocate:
        return 0.65  # willing to relocate but currently abroad
    return 0.40  # location ambiguous


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(c: Candidate) -> Features:
    """Extract all JD-grounding features for one candidate."""
    career_text = _build_career_text(c)

    title_tier = _classify_title(c.profile.current_title)
    title_score = _TITLE_TIER_SCORE[title_tier]

    r_hits = _retrieval_hits(career_text)
    p_hits = _production_hits(career_text)
    career_retrieval = min(1.0, r_hits / 5.0)   # saturate at 5 hits
    career_production = min(1.0, p_hits / 3.0)  # saturate at 3 hits

    skill_trust_score, matched = _skill_trust(c)
    cv_frac = _cv_speech_fraction(c)

    yoe = c.profile.years_of_experience
    exp_fit = _experience_fit(yoe)
    loc_boost = _location_boost(c)

    svc_frac = _services_fraction(c)
    lco = _langchain_only(c, r_hits)

    return Features(
        title_tier=title_tier,
        title_score=title_score,
        career_retrieval=career_retrieval,
        career_production=career_production,
        skill_trust=skill_trust_score,
        cv_speech_fraction=cv_frac,
        experience_fit=exp_fit,
        location_boost=loc_boost,
        services_fraction=svc_frac,
        langchain_only=lco,
        retrieval_hits=r_hits,
        production_hits=p_hits,
        matched_skills=matched,
    )
