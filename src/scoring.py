"""Final fit score combiner (Phase 4).

Combines structured JD-grounding features and behavioral modifier into
one final score per candidate in [0, 1].

Score formula:
    fit = title * 0.20
        + career_retrieval * 0.30
        + career_production * 0.15
        + skill_trust * 0.15
        + experience_fit * 0.10
        + location_boost * 0.10

    services_mult = max(0.5, 1.0 - max(0, frac - 0.3) * 0.7)
    bmod = behavioral_modifier(c)           # [0.5, 1.0]

    final = fit * services_mult * bmod

Design rationale:
- Title is a gate for DISQUALIFIER (hard zero) and a weight otherwise.
- career_retrieval is the single largest component (30%) — stuffers cannot
  fake real retrieval/ranking career descriptions.
- services_mult starts penalising at 30% services exposure; pure-services
  careers are capped at 0.5× their fit score.
- behavioral modifier is multiplicative and capped [0.5, 1.0] — signals
  modulate fit but cannot override it.

Adjust _FIT_WEIGHTS if empirical tuning shows a component is under/over-weighted.
"""

from __future__ import annotations

from src.behavioral import behavioral_modifier
from src.data import Candidate
from src.features import Features, extract_features

# Feature weights (must sum to 1.0)
_FIT_WEIGHTS: dict[str, float] = {
    "title": 0.20,
    "career_retrieval": 0.30,
    "career_production": 0.15,
    "skill_trust": 0.15,
    "experience_fit": 0.10,
    "location_boost": 0.10,
}


def _fit_score(f: Features) -> float:
    """Weighted combination of JD-grounding features, [0, 1]."""
    return (
        f.title_score * _FIT_WEIGHTS["title"]
        + f.career_retrieval * _FIT_WEIGHTS["career_retrieval"]
        + f.career_production * _FIT_WEIGHTS["career_production"]
        + f.skill_trust * _FIT_WEIGHTS["skill_trust"]
        + f.experience_fit * _FIT_WEIGHTS["experience_fit"]
        + f.location_boost * _FIT_WEIGHTS["location_boost"]
    )


def _services_multiplier(services_fraction: float) -> float:
    """Penalise careers heavily weighted towards pure-consulting companies.

    No penalty below 30% services exposure.  Reaches 0.5× at 100% services.
    """
    excess = max(0.0, services_fraction - 0.30)
    return max(0.50, 1.0 - excess * 0.70)


def _langchain_penalty(f: Features) -> float:
    """Light penalty for LangChain-wrapper-only candidates (JD disqualifier)."""
    return 0.85 if f.langchain_only else 1.0


def score_candidate(c: Candidate) -> tuple[float, Features]:
    """Compute the final fit score and return (score, features).

    Returns score=0.0 immediately for DISQUALIFIER titles (HR Manager,
    Accountant, …). All other titles are scored via the weighted formula.

    Returns:
        (final_score [0, 1], Features for reasoning/debugging)
    """
    f = extract_features(c)

    if f.title_tier == "DISQUALIFIER":
        return 0.0, f

    fit = _fit_score(f)
    svc_mult = _services_multiplier(f.services_fraction)
    lco_mult = _langchain_penalty(f)
    bmod = behavioral_modifier(c)

    final = fit * svc_mult * lco_mult * bmod
    return final, f
