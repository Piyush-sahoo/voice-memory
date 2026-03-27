"""
Company policy rules for the Voice Agent Prompt Optimizer.

These rules define constraints that the optimized prompt must respect.
The grader checks for policy violations in submitted prompts.
"""

from typing import Dict, List


# ─────────────────────────────────────────────────────────────
# Policies: keywords/phrases that indicate a violation
# ─────────────────────────────────────────────────────────────

POLICY_VIOLATIONS: List[Dict] = [
    {
        "id": "promise_discount_over_limit",
        "description": "Promising discount above 20% without supervisor approval",
        "violation_keywords": ["50% off", "50% discount", "half price", "75% off", "free forever"],
        "severity": 0.15,
    },
    {
        "id": "guarantee_outcome",
        "description": "Guaranteeing outcomes the company cannot ensure",
        "violation_keywords": ["guarantee", "100% uptime", "never fail", "promise you will never"],
        "severity": 0.10,
    },
    {
        "id": "share_internal_info",
        "description": "Instructing the agent to share internal processes or competitor intel",
        "violation_keywords": ["our internal", "confidential", "secret pricing", "off the record"],
        "severity": 0.20,
    },
    {
        "id": "ignore_dnc",
        "description": "Ignoring do-not-call or opt-out requests",
        "violation_keywords": [
            "ignore their request to stop",
            "call anyway",
            "disregard opt-out",
            "keep calling regardless",
        ],
        "severity": 0.25,
    },
    {
        "id": "badmouth_competitor",
        "description": "Directly disparaging competitors",
        "violation_keywords": [
            "terrible company",
            "awful service",
            "scam",
            "they are liars",
            "their product is garbage",
        ],
        "severity": 0.15,
    },
    {
        "id": "pressure_tactics",
        "description": "Using high-pressure or manipulative sales tactics",
        "violation_keywords": [
            "you must decide now",
            "this is your last chance",
            "you'll regret",
            "everyone else is buying",
            "only an idiot would",
        ],
        "severity": 0.20,
    },
    {
        "id": "skip_identity_verification",
        "description": "Skipping identity verification for account operations",
        "violation_keywords": [
            "skip verification",
            "no need to verify",
            "don't bother checking",
            "assume it's them",
        ],
        "severity": 0.15,
    },
]


def check_policy_violations(prompt: str) -> List[Dict]:
    """
    Check an optimized prompt for policy violations.

    Args:
        prompt: The proposed system prompt text.

    Returns:
        List of violation dicts with id, description, severity.
    """
    prompt_lower = prompt.lower()
    violations = []
    for policy in POLICY_VIOLATIONS:
        for keyword in policy["violation_keywords"]:
            if keyword.lower() in prompt_lower:
                violations.append(
                    {
                        "id": policy["id"],
                        "description": policy["description"],
                        "severity": policy["severity"],
                    }
                )
                break  # one match per policy is enough
    return violations


def total_violation_penalty(prompt: str) -> float:
    """
    Calculate total penalty score for policy violations.

    Returns:
        Float between 0.0 and 1.0 representing the total penalty.
    """
    violations = check_policy_violations(prompt)
    penalty = sum(v["severity"] for v in violations)
    return min(penalty, 1.0)  # cap at 1.0
