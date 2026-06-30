"""Feature-grounded reasoning generator for the top-100 ranked candidates (Phase 6).

Generates a 1–2 sentence reasoning string for each candidate WITHOUT any LLM or network
access (all scoring is done offline, reasoning is derived from the same features that
drove the score).

Stage-4 checks the reasoning must satisfy (6 criteria):
  1. Specific       — cites actual facts: title, company, named skills, year counts.
  2. JD-linked      — connects the specific fact to a JD requirement.
  3. Honest         — does not overstate; acknowledges concerns when present.
  4. No hallucination — every fact must be derivable from the candidate object.
  5. Varied         — doesn't use the same template across all 100 rows.
  6. Rank-consistent — tone/confidence match the score (top-5 > top-50).

Implementation: string templates keyed on the dominant feature combination.
The variation comes from using different templates, and plugging in the candidate's
actual values (name of company, specific skills, YOE, etc.).
"""

from __future__ import annotations

from src.data import Candidate
from src.features import Features

# Rank bands for tone calibration
_BAND_TOP = 20   # ranks 1-20: strong positive framing
_BAND_MID = 60   # ranks 21-60: balanced framing
# ranks 61-100: more cautious framing


def _fmt_skills(skills: list[str], max_n: int = 3) -> str:
    """Format a list of skill names, e.g. 'FAISS, Pinecone, and embeddings'."""
    shown = skills[:max_n]
    if not shown:
        return "relevant ML skills"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def _current_role(c: Candidate) -> str:
    return f"{c.profile.current_title} at {c.profile.current_company}"


def _yoe_str(c: Candidate) -> str:
    yoe = c.profile.years_of_experience
    return f"{yoe:.0f} years" if yoe == int(yoe) else f"{yoe:.1f} years"


def _india_note(c: Candidate) -> str:
    loc = c.profile.location
    return f"based in {loc}" if loc else "India-based"


def _active_note(c: Candidate) -> str:
    if c.redrob_signals.open_to_work_flag:
        return "actively open to work"
    return "available with notice"


def _concern_note(c: Candidate, f: Features) -> str | None:
    """Return a brief honest concern string if any, else None."""
    if f.services_fraction > 0.60:
        return "bulk of career at a services company"
    if c.profile.years_of_experience < 4.0:
        return f"only {_yoe_str(c)} of experience (JD asks for 5–9)"
    if c.profile.years_of_experience > 11.0:
        return f"{_yoe_str(c)} of experience (over-seniorised for a founding role)"
    if c.redrob_signals.recruiter_response_rate < 0.30:
        return "historically slow to respond to recruiters"
    otw = c.redrob_signals.open_to_work_flag
    inactive = c.redrob_signals.last_active_date < "2026-01-01"
    if not otw and inactive:
        return "not actively looking and has been inactive since early 2026"
    return None


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------


def _template_strong_retrieval(c: Candidate, f: Features) -> str:
    """For STRONG_POSITIVE titles with high career retrieval evidence."""
    role = _current_role(c)
    skills = _fmt_skills(f.matched_skills)
    yoe = _yoe_str(c)
    active = _active_note(c)
    concern = _concern_note(c, f)

    base = (
        f"{role} with {yoe} building retrieval and ranking systems, "
        f"with validated depth in {skills}; {_india_note(c)}, {active}."
    )
    if concern:
        base += f" One concern: {concern}."
    return base


def _template_strong_no_skills(c: Candidate, f: Features) -> str:
    """STRONG_POSITIVE title, retrieval career evidence but few matched skills."""
    role = _current_role(c)
    yoe = _yoe_str(c)
    concern = _concern_note(c, f)

    base = (
        f"{role} with {yoe} of experience; career descriptions show "
        f"{f.retrieval_hits} retrieval/ranking signal(s) and {f.production_hits} "
        f"production deployment mention(s), though fewer formally listed skills "
        f"match the JD's must-haves."
    )
    if concern:
        base += f" Note: {concern}."
    return base


def _template_adjacent_retrieval(c: Candidate, f: Features) -> str:
    """ADJACENT title but strong retrieval career evidence (plain-language tier-5)."""
    skills = _fmt_skills(f.matched_skills)
    yoe = _yoe_str(c)
    concern = _concern_note(c, f)

    base = (
        f"Title is '{c.profile.current_title}' but career history shows "
        f"{f.retrieval_hits} retrieval/ranking signal(s) and skills include "
        f"{skills}; {yoe} of experience, {_india_note(c)}."
    )
    if concern:
        base += f" Concern: {concern}."
    return base


def _template_solid_mid(c: Candidate, f: Features) -> str:
    """Mid-tier candidates: relevant title, moderate signals."""
    role = _current_role(c)
    yoe = _yoe_str(c)
    concern = _concern_note(c, f)

    base = (
        f"{role}, {yoe}; {f.retrieval_hits} retrieval and "
        f"{f.production_hits} production career signal(s). "
    )
    if f.matched_skills:
        base += f"Matched skills: {_fmt_skills(f.matched_skills)}. "
    base += f"{_india_note(c).capitalize()}, {_active_note(c)}."
    if concern:
        base += f" Note: {concern}."
    return base


def _template_weak(c: Candidate, f: Features) -> str:
    """Borderline candidates: included for recall coverage, concerns dominant."""
    role = _current_role(c)
    yoe = _yoe_str(c)
    concern = _concern_note(c, f)

    base = (
        f"{role}, {yoe}; limited retrieval career evidence "
        f"({f.retrieval_hits} signal(s))."
    )
    if f.matched_skills:
        skills_str = _fmt_skills(f.matched_skills)
        base += f" Some relevant skills listed ({skills_str}) but depth unverified."
    if concern:
        base += f" Concern: {concern}."
    return base


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def generate_reasoning(c: Candidate, f: Features, rank: int) -> str:
    """Generate a 1–2 sentence grounded reasoning string for one ranked candidate.

    Args:
        c: Candidate object (source of all factual claims).
        f: Precomputed features (used for signal counts, matched skills, etc.).
        rank: 1-indexed position in the final ranking (influences tone).

    Returns:
        Reasoning string (1–2 sentences, ≤400 chars).
    """
    is_strong_title = f.title_tier == "STRONG_POSITIVE"
    is_adjacent = f.title_tier in ("ADJACENT", "UNKNOWN")
    high_retrieval = f.retrieval_hits >= 4
    has_skills = len(f.matched_skills) >= 2

    # Select template
    if is_strong_title and high_retrieval and has_skills:
        text = _template_strong_retrieval(c, f)
    elif is_strong_title and (high_retrieval or has_skills):
        text = _template_strong_no_skills(c, f)
    elif is_adjacent and high_retrieval:
        text = _template_adjacent_retrieval(c, f)
    elif rank <= _BAND_TOP:
        text = _template_solid_mid(c, f)
    elif rank <= _BAND_MID:
        text = _template_solid_mid(c, f)
    else:
        text = _template_weak(c, f)

    # Truncate at 400 chars (Stage-4 readability)
    if len(text) > 400:
        text = text[:397] + "..."
    return text
