"""
Reward function for the Voice Agent Prompt Optimizer.

Provides per-turn partial rewards so the RL agent gets signal at every
optimization turn, not just a sparse terminal reward.

Reward = improvement_delta + quality_bonus - violation_penalty + terminal_bonus
"""

from typing import Dict, Optional


def compute_reward(
    current_grade: Dict,
    previous_grade: Optional[Dict],
    is_terminal: bool,
) -> float:
    """
    Compute the reward for a single optimization step.

    Args:
        current_grade: Grade result from graders.grade_prompt for this turn.
        previous_grade: Grade result from the previous turn (None if first turn).
        is_terminal: Whether this is the last turn.

    Returns:
        Float reward value (can be negative for regressions).
    """
    current_score = current_grade["score"]

    # --- 1. Improvement delta (reward progress, penalize regression) ---
    if previous_grade is not None:
        previous_score = previous_grade["score"]
        delta = current_score - previous_score
        # Scale delta: +0.1 improvement in score → +0.3 reward
        improvement_reward = delta * 3.0
    else:
        # First turn: reward based on absolute score
        # A score of 0.5 on first try → 0.25 reward
        improvement_reward = current_score * 0.5

    # --- 2. New improvements addressed bonus ---
    if previous_grade is not None:
        prev_addressed = set(
            k for k, v in previous_grade.get("improvement_breakdown", {}).items() if v
        )
        curr_addressed = set(
            k for k, v in current_grade.get("improvement_breakdown", {}).items() if v
        )
        new_improvements = curr_addressed - prev_addressed
        new_improvement_bonus = len(new_improvements) * 0.1
    else:
        addressed = sum(
            1 for v in current_grade.get("improvement_breakdown", {}).values() if v
        )
        new_improvement_bonus = addressed * 0.05

    # --- 3. Policy violation penalty ---
    violation_penalty = current_grade.get("policy_penalty", 0.0) * -0.5

    # --- 4. Terminal bonus ---
    terminal_bonus = 0.0
    if is_terminal:
        # Bonus proportional to final score
        terminal_bonus = current_score * 0.5

    # --- Combine ---
    reward = improvement_reward + new_improvement_bonus + violation_penalty + terminal_bonus

    # Clamp to reasonable range
    return round(max(-1.0, min(2.0, reward)), 4)
