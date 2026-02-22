"""
risk_profile.py
---------------
Risk profiling — 3 questions filter recommendations
based on user's investment horizon, risk tolerance and capital.
"""

import os
import json
import logging

log = logging.getLogger(__name__)

_BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK_PROFILE_FILE = os.path.join(_BASE_DIR, "data_storage", "risk_profile.json")


# Risk profile definitions
PROFILES = {
    "conservative": {
        "label":       "Conservative",
        "description": "Capital preservation. Only STRONG BUY signals.",
        "allowed_verdicts": ["STRONG BUY"],
        "max_single_stock_pct": 10,   # max 10% in one stock
        "preferred_sectors":    ["Banking", "FMCG", "Pharma", "IT"],
    },
    "moderate": {
        "label":       "Moderate",
        "description": "Balanced growth and safety. BUY and STRONG BUY signals.",
        "allowed_verdicts": ["STRONG BUY", "BUY"],
        "max_single_stock_pct": 20,
        "preferred_sectors":    ["Banking", "IT", "FMCG", "Auto", "Pharma", "Finance"],
    },
    "aggressive": {
        "label":       "Aggressive",
        "description": "Maximum growth. All signals including HOLD.",
        "allowed_verdicts": ["STRONG BUY", "BUY", "HOLD"],
        "max_single_stock_pct": 40,
        "preferred_sectors":    None,  # all sectors
    },
}


def _score_answers(horizon: str, tolerance: str, capital: float) -> str:
    """
    Map 3 answers to a risk profile.

    horizon   : "short" (< 1yr) | "medium" (1-3yr) | "long" (3yr+)
    tolerance : "low" | "medium" | "high"
    capital   : amount in INR
    """
    score = 0

    # Horizon scoring
    if horizon == "long":
        score += 2
    elif horizon == "medium":
        score += 1

    # Tolerance scoring
    if tolerance == "high":
        score += 2
    elif tolerance == "medium":
        score += 1

    # Capital scoring (larger capital = can take more risk)
    if capital >= 500_000:
        score += 1

    if score >= 4:
        return "aggressive"
    elif score >= 2:
        return "moderate"
    return "conservative"


def set_risk_profile(horizon: str, tolerance: str, capital: float) -> dict:
    """
    Save risk profile based on 3 answers.
    Returns the computed profile.
    """
    horizon   = horizon.lower().strip()
    tolerance = tolerance.lower().strip()

    if horizon not in ("short", "medium", "long"):
        return {"error": "horizon must be 'short', 'medium', or 'long'"}
    if tolerance not in ("low", "medium", "high"):
        return {"error": "tolerance must be 'low', 'medium', or 'high'"}
    if capital <= 0:
        return {"error": "capital must be a positive number"}

    profile_key = _score_answers(horizon, tolerance, capital)
    profile     = PROFILES[profile_key]

    data = {
        "profile_key":  profile_key,
        "horizon":      horizon,
        "tolerance":    tolerance,
        "capital":      capital,
        "label":        profile["label"],
        "description":  profile["description"],
        "allowed_verdicts":      profile["allowed_verdicts"],
        "max_single_stock_pct":  profile["max_single_stock_pct"],
        "preferred_sectors":     profile["preferred_sectors"],
    }

    os.makedirs(os.path.dirname(RISK_PROFILE_FILE), exist_ok=True)
    with open(RISK_PROFILE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return data


def get_risk_profile() -> dict:
    """Return saved risk profile, or None if not set."""
    if not os.path.exists(RISK_PROFILE_FILE):
        return {"error": "No risk profile set. Use POST /risk-profile first."}
    with open(RISK_PROFILE_FILE, "r") as f:
        return json.load(f)


def filter_by_risk(recommendations: list) -> list:
    """
    Filter a list of stock recommendations by the saved risk profile.
    Each item must have a 'recommendation' or 'verdict' key.
    Returns only stocks that match the profile.
    """
    profile = get_risk_profile()
    if "error" in profile:
        return recommendations  # no profile set, return all

    allowed = profile["allowed_verdicts"]
    sectors = profile["preferred_sectors"]  # None means all

    filtered = []
    for r in recommendations:
        verdict = r.get("recommendation") or r.get("verdict", "")
        sector  = r.get("sector", "")

        if verdict not in allowed:
            continue
        if sectors and sector not in sectors:
            continue

        filtered.append(r)

    return filtered