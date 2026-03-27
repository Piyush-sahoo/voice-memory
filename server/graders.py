"""
Deterministic graders for the Voice Agent Prompt Optimizer.

Each grader scores an optimized prompt against the scenario's ground truth
required_improvements and ideal_prompt_elements. Scores are 0.0 to 1.0.

Grading logic:
  1. Improvement coverage: fraction of required_improvements addressed
  2. Prompt quality: structural checks (length, specificity, clarity)
  3. Policy compliance: penalty for violations
  4. Final score = weighted combination, clamped to [0.0, 1.0]
"""

from typing import Dict, List, Tuple

from .policy_rules import check_policy_violations, total_violation_penalty


def _check_improvement_addressed(prompt_lower: str, improvement_id: str, ideal_elements: List[str]) -> bool:
    """
    Check if a required improvement is addressed in the prompt.

    Uses a heuristic: if any ideal_prompt_element related to this improvement
    appears in the prompt, the improvement is considered addressed.
    We map improvement IDs to relevant ideal elements via substring matching.
    """
    # For each improvement, check if at least one related ideal element is present
    # The ideal_elements list is ordered to roughly correspond to improvements
    # We use a fuzzy approach: any ideal element match counts
    for element in ideal_elements:
        if element.lower() in prompt_lower:
            return True
    return False


def _compute_improvement_score(
    prompt: str,
    required_improvements: List[str],
    ideal_elements: List[str],
) -> Tuple[float, Dict[str, bool]]:
    """
    Compute what fraction of required improvements are addressed.

    Returns:
        (score, breakdown) where score is 0.0-1.0 and breakdown maps
        each improvement to True/False.
    """
    prompt_lower = prompt.lower()
    breakdown: Dict[str, bool] = {}

    # Map each improvement to a subset of ideal elements using keyword heuristics
    # We split ideal elements roughly evenly across improvements
    elements_per_improvement = max(1, len(ideal_elements) // max(1, len(required_improvements)))

    for i, improvement in enumerate(required_improvements):
        start = i * elements_per_improvement
        end = start + elements_per_improvement
        # Also check the improvement ID keywords themselves
        relevant_elements = ideal_elements[start:end]

        # Check improvement ID keywords (e.g., "include_business_hours" → "business hours")
        improvement_keywords = improvement.replace("_", " ").lower().split()

        addressed = False
        # Check ideal elements
        for elem in relevant_elements:
            if elem.lower() in prompt_lower:
                addressed = True
                break

        # Also check improvement keywords directly
        if not addressed:
            keyword_matches = sum(1 for kw in improvement_keywords if kw in prompt_lower)
            if keyword_matches >= max(1, len(improvement_keywords) // 2):
                addressed = True

        breakdown[improvement] = addressed

    addressed_count = sum(1 for v in breakdown.values() if v)
    score = addressed_count / max(1, len(required_improvements))
    return score, breakdown


def _compute_quality_score(prompt: str) -> Tuple[float, Dict[str, float]]:
    """
    Structural quality checks on the prompt.

    Returns:
        (score, breakdown) where score is 0.0-1.0.
    """
    checks: Dict[str, float] = {}

    # Length: too short = bad, too long = slightly bad, 200-1500 chars = good
    length = len(prompt)
    if length < 50:
        checks["length"] = 0.1
    elif length < 200:
        checks["length"] = 0.5
    elif length <= 1500:
        checks["length"] = 1.0
    elif length <= 3000:
        checks["length"] = 0.7
    else:
        checks["length"] = 0.4

    # Structure: has numbered steps or bullet points
    has_structure = any(
        marker in prompt
        for marker in ["1.", "1)", "- ", "* ", "Step 1", "Rule 1", "First,"]
    )
    checks["structure"] = 1.0 if has_structure else 0.3

    # Specificity: contains numbers, percentages, or concrete details
    import re

    has_numbers = bool(re.search(r"\d+", prompt))
    has_percentage = bool(re.search(r"\d+%", prompt))
    specificity = 0.2
    if has_numbers:
        specificity += 0.4
    if has_percentage:
        specificity += 0.4
    checks["specificity"] = min(specificity, 1.0)

    # Role clarity: defines who the agent is
    role_keywords = ["you are", "your role", "as a", "voice assistant", "agent"]
    has_role = any(kw in prompt.lower() for kw in role_keywords)
    checks["role_clarity"] = 1.0 if has_role else 0.3

    score = sum(checks.values()) / len(checks)
    return score, checks


def grade_prompt(
    prompt: str,
    scenario: Dict,
) -> Dict:
    """
    Grade an optimized prompt against a scenario.

    Args:
        prompt: The optimized system prompt submitted by the RL agent.
        scenario: The scenario dict with required_improvements, ideal_prompt_elements, etc.

    Returns:
        Dict with:
          - score: float 0.0-1.0 (final composite score)
          - improvement_score: float (fraction of improvements addressed)
          - quality_score: float (structural quality)
          - policy_penalty: float (penalty for violations)
          - improvement_breakdown: dict mapping each improvement to bool
          - quality_breakdown: dict of quality checks
          - policy_violations: list of violation dicts
    """
    required = scenario.get("required_improvements", [])
    ideal = scenario.get("ideal_prompt_elements", [])

    # 1. Improvement coverage (60% weight)
    improvement_score, improvement_breakdown = _compute_improvement_score(
        prompt, required, ideal
    )

    # 2. Quality (20% weight)
    quality_score, quality_breakdown = _compute_quality_score(prompt)

    # 3. Policy compliance (20% weight — as penalty)
    violations = check_policy_violations(prompt)
    policy_penalty = total_violation_penalty(prompt)

    # Composite score
    raw_score = (0.60 * improvement_score) + (0.20 * quality_score) + (0.20 * (1.0 - policy_penalty))
    final_score = max(0.0, min(1.0, raw_score))

    return {
        "score": round(final_score, 4),
        "improvement_score": round(improvement_score, 4),
        "quality_score": round(quality_score, 4),
        "policy_penalty": round(policy_penalty, 4),
        "improvement_breakdown": improvement_breakdown,
        "quality_breakdown": quality_breakdown,
        "policy_violations": violations,
    }


def generate_feedback(grade_result: Dict, scenario: Dict) -> str:
    """
    Generate human-readable feedback from grading results.
    """
    lines = []
    score = grade_result["score"]

    if score >= 0.8:
        lines.append("Excellent optimization! The prompt addresses most key issues.")
    elif score >= 0.6:
        lines.append("Good progress. Several improvements addressed but some gaps remain.")
    elif score >= 0.4:
        lines.append("Partial improvement. Significant issues still unaddressed.")
    else:
        lines.append("The optimized prompt needs substantial work.")

    # Unaddressed improvements
    unaddressed = [
        k for k, v in grade_result["improvement_breakdown"].items() if not v
    ]
    if unaddressed:
        lines.append(f"Missing improvements: {', '.join(unaddressed)}")

    # Quality issues
    for check, val in grade_result["quality_breakdown"].items():
        if val < 0.5:
            lines.append(f"Quality issue: {check} is weak.")

    # Policy violations
    if grade_result["policy_violations"]:
        viol_names = [v["description"] for v in grade_result["policy_violations"]]
        lines.append(f"Policy violations: {'; '.join(viol_names)}")

    return " | ".join(lines)
