# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Voice Agent Prompt Optimizer — Environment Implementation.

An RL environment for post-call analysis of AI voice agent conversations.
The agent receives call transcripts and must iteratively optimize the voice
agent's system prompt to improve future call handling.

Pipeline modeled: LiveKit + Deepgram STT (Nova-3) + GPT-4o-mini + OpenAI TTS.

Episode flow:
  1. reset(task_id, scenario_id) → load scenario, return transcript + current prompt
  2. step(optimized_prompt) → grade prompt, return feedback, up to 3 turns
  3. done=True when max_turns reached or agent signals final answer
"""

import random
import time
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import VoiceAgentAction, VoiceAgentObservation
    from .graders import generate_feedback, grade_prompt
    from .rewards import compute_reward
    from .scenarios import ALL_SCENARIOS, TASK_DEFINITIONS
except (ImportError, ModuleNotFoundError):
    from models import VoiceAgentAction, VoiceAgentObservation
    from server.graders import generate_feedback, grade_prompt
    from server.rewards import compute_reward
    from server.scenarios import ALL_SCENARIOS, TASK_DEFINITIONS


class VoiceAgentEnvironment(Environment):
    """
    Voice Agent Prompt Optimizer Environment.

    The RL agent receives call transcripts from a voice AI system and must
    optimize the system prompt to improve future call handling. Each episode
    presents a scenario with failure points and required improvements.
    The agent can iterate on the prompt for up to 3 turns.

    Tasks:
      - faq_resolution (easy): fix simple knowledge gaps
      - complaint_handling (medium): add empathy + de-escalation
      - multi_session_sales (hard): multi-session context + objection handling

    Example:
        >>> env = VoiceAgentEnvironment()
        >>> obs = env.reset()  # random scenario
        >>> obs = env.step(VoiceAgentAction(optimized_prompt="You are...", reasoning="Added..."))
        >>> print(obs.score_breakdown)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_TURNS: int = 3

    def __init__(self):
        """Initialize the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Dict] = None
        self._previous_grade: Optional[Dict] = None
        self._current_grade: Optional[Dict] = None
        self._turn: int = 0
        self._done: bool = False
        self._episode_scores: list = []

    def reset(self, task_id: str = None, scenario_id: str = None, **kwargs) -> VoiceAgentObservation:
        """
        Reset the environment and load a scenario.

        Args:
            task_id: One of 'faq_resolution', 'complaint_handling', 'multi_session_sales'.
                     If None, picks a random task.
            scenario_id: Specific scenario ID. If None, picks random within task.

        Returns:
            VoiceAgentObservation with call transcript, current prompt, and failure analysis.
        """
        # Pick task
        if task_id is None or task_id not in ALL_SCENARIOS:
            task_id = random.choice(list(ALL_SCENARIOS.keys()))

        scenarios = ALL_SCENARIOS[task_id]

        # Pick scenario
        if scenario_id:
            matching = [s for s in scenarios if s["scenario_id"] == scenario_id]
            scenario = matching[0] if matching else random.choice(scenarios)
        else:
            scenario = random.choice(scenarios)

        # Reset state
        self._scenario = scenario
        self._previous_grade = None
        self._current_grade = None
        self._turn = 0
        self._done = False
        self._episode_scores = []
        self._state = State(
            episode_id=kwargs.get("episode_id") or str(uuid4()),
            step_count=0,
        )

        return VoiceAgentObservation(
            done=False,
            reward=None,
            # Task context
            task_id=scenario["task_id"],
            task_difficulty=scenario["task_difficulty"],
            scenario_id=scenario["scenario_id"],
            # Call context
            current_prompt=scenario["current_prompt"],
            call_transcripts=scenario["call_transcripts"],
            call_metadata=scenario["call_metadata"],
            # Failure analysis
            failure_points=scenario["failure_points"],
            required_improvements=scenario["required_improvements"],
            policy_context=scenario["policy_context"],
            # Feedback (empty on reset)
            score_breakdown={},
            feedback_message=(
                f"Analyze the call transcript(s) and optimize the system prompt. "
                f"Task: {scenario['task_id']} ({scenario['task_difficulty']}). "
                f"You have {self.MAX_TURNS} turns to iterate."
            ),
            turn_number=0,
            max_turns=self.MAX_TURNS,
        )

    def step(self, action: VoiceAgentAction) -> VoiceAgentObservation:  # type: ignore[override]
        """
        Process an optimization attempt.

        Args:
            action: VoiceAgentAction with optimized_prompt and reasoning.

        Returns:
            VoiceAgentObservation with score breakdown, feedback, and done status.
        """
        if self._scenario is None:
            return VoiceAgentObservation(
                done=True,
                reward=0.0,
                feedback_message="Error: call reset() before step().",
            )

        if self._done:
            return VoiceAgentObservation(
                done=True,
                reward=0.0,
                feedback_message="Episode already finished. Call reset().",
                score_breakdown=self._current_grade or {},
            )

        self._turn += 1
        self._state.step_count = self._turn

        # Grade the submitted prompt
        self._previous_grade = self._current_grade
        self._current_grade = grade_prompt(action.optimized_prompt, self._scenario)

        # Check if done
        is_last_turn = self._turn >= self.MAX_TURNS
        self._done = is_last_turn

        # Compute reward
        reward = compute_reward(
            self._current_grade, self._previous_grade, is_terminal=is_last_turn
        )

        self._episode_scores.append(self._current_grade["score"])

        # Generate feedback
        feedback = generate_feedback(self._current_grade, self._scenario)
        if not is_last_turn:
            feedback += f" | You have {self.MAX_TURNS - self._turn} turn(s) remaining to improve."

        return VoiceAgentObservation(
            done=self._done,
            reward=reward,
            # Task context (repeated for convenience)
            task_id=self._scenario["task_id"],
            task_difficulty=self._scenario["task_difficulty"],
            scenario_id=self._scenario["scenario_id"],
            # Call context
            current_prompt=self._scenario["current_prompt"],
            call_transcripts=self._scenario["call_transcripts"],
            call_metadata=self._scenario["call_metadata"],
            # Failure analysis
            failure_points=self._scenario["failure_points"],
            required_improvements=self._scenario["required_improvements"],
            policy_context=self._scenario["policy_context"],
            # Feedback
            score_breakdown=self._current_grade,
            feedback_message=feedback,
            turn_number=self._turn,
            max_turns=self.MAX_TURNS,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ─── Helper methods for custom endpoints ───

    def get_last_grade(self) -> Optional[Dict]:
        """Return the last grading result (used by /grader endpoint)."""
        return self._current_grade

    def get_episode_scores(self) -> list:
        """Return all scores from the current episode."""
        return list(self._episode_scores)

    def live_reset(self, live_data: Dict) -> VoiceAgentObservation:
        """
        Reset the environment with a live call transcript.

        Used by the /live-reset endpoint to inject real transcripts
        from LiveKit calls rather than synthetic scenarios.

        Args:
            live_data: Dict with keys:
                - task_id (str)
                - task_difficulty (str)
                - current_prompt (str)
                - call_transcripts (list)
                - call_metadata (dict)
                - failure_points (list, optional)
                - policy_context (str, optional)

        Returns:
            VoiceAgentObservation with the live transcript loaded.
        """
        # Auto-detect failure points if not provided
        failure_points = live_data.get("failure_points", [])
        if not failure_points:
            failure_points = self._auto_detect_failures(live_data)

        # Build a live scenario
        self._scenario = {
            "scenario_id": f"live_{int(time.time())}",
            "task_id": live_data.get("task_id", "live_call"),
            "task_difficulty": live_data.get("task_difficulty", "live"),
            "current_prompt": live_data.get("current_prompt", ""),
            "call_transcripts": live_data.get("call_transcripts", []),
            "call_metadata": live_data.get("call_metadata", {}),
            "failure_points": failure_points,
            "required_improvements": [
                fp.lower().replace(" ", "_")[:40] for fp in failure_points
            ],
            "ideal_prompt_elements": [],  # live calls use LLM-based grading
            "policy_context": live_data.get("policy_context", ""),
        }

        # Reset state
        self._previous_grade = None
        self._current_grade = None
        self._turn = 0
        self._done = False
        self._episode_scores = []
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
        )

        return VoiceAgentObservation(
            done=False,
            reward=None,
            task_id=self._scenario["task_id"],
            task_difficulty=self._scenario["task_difficulty"],
            scenario_id=self._scenario["scenario_id"],
            current_prompt=self._scenario["current_prompt"],
            call_transcripts=self._scenario["call_transcripts"],
            call_metadata=self._scenario["call_metadata"],
            failure_points=self._scenario["failure_points"],
            required_improvements=self._scenario["required_improvements"],
            policy_context=self._scenario["policy_context"],
            score_breakdown={},
            feedback_message=(
                f"Live transcript loaded ({len(self._scenario['call_transcripts'])} session(s)). "
                f"Detected {len(failure_points)} failure point(s). "
                f"You have {self.MAX_TURNS} turns to optimize the prompt."
            ),
            turn_number=0,
            max_turns=self.MAX_TURNS,
        )

    @staticmethod
    def _auto_detect_failures(live_data: Dict) -> List[str]:
        """
        Auto-detect common failure points from a live transcript.

        Looks for patterns that indicate the agent performed poorly:
        - Deflection to website
        - Vague/non-specific answers
        - Customer frustration indicators
        - Unanswered direct questions
        - Unnecessary transfers
        """
        failures = []
        transcripts = live_data.get("call_transcripts", [])

        for t_data in transcripts:
            turns = t_data.get("turns", [])
            for i, turn in enumerate(turns):
                text_lower = turn.get("text", "").lower()
                role = turn.get("role", "")

                if role == "agent":
                    # Deflection to website
                    if any(w in text_lower for w in ["visit our website", "check our website", "go to our site"]):
                        failures.append("Agent deflected to website instead of answering directly")

                    # Vague answers
                    if any(w in text_lower for w in ["i'm not sure", "i think", "i believe", "let me check"]):
                        failures.append("Agent gave vague/uncertain answer")

                    # Hedging with superlatives
                    vague_count = sum(1 for w in ["great", "amazing", "excellent", "competitive", "various"]
                                      if w in text_lower)
                    if vague_count >= 2:
                        failures.append("Agent used vague superlatives without specific details")

                    # Unnecessary transfer
                    if "transfer" in text_lower and i < len(turns) - 1:
                        failures.append("Agent attempted transfer without trying to resolve first")

                elif role == "customer":
                    # Customer frustration
                    if any(w in text_lower for w in ["useless", "terrible", "worst", "never mind", "forget it",
                                                      "that's not helpful", "bye"]):
                        failures.append("Customer expressed frustration or abandoned call")

                    # Customer repeating themselves
                    if any(w in text_lower for w in ["i just said", "i already", "i told you", "as i said"]):
                        failures.append("Customer had to repeat themselves (agent didn't listen)")

        # Check metadata for unresolved calls
        metadata = live_data.get("call_metadata", {})
        if metadata.get("resolution_status") in ["unresolved", "escalated_failed", "lost"]:
            failures.append("Call ended without resolution")

        # Deduplicate
        return list(dict.fromkeys(failures))
