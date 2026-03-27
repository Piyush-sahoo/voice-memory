# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Voice Agent Prompt Optimizer Environment.

An RL environment for post-call analysis of AI voice agent conversations.
The agent receives call transcripts from a voice AI system (LiveKit + Deepgram STT
+ GPT-4o-mini + TTS pipeline) and must optimize the voice agent's system prompt
to improve future call handling.

Real-world domain: Every voice AI company manually tunes prompts after reviewing
call transcripts. This environment automates that feedback loop via RL.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VoiceAgentAction(Action):
    """
    Action: the RL agent submits an optimized system prompt.

    The agent analyzes call transcript(s) and proposes an improved system prompt
    for the voice agent, along with reasoning about what was changed.
    """

    optimized_prompt: str = Field(
        ...,
        description="The improved system prompt for the voice agent",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of what was changed and why",
    )


class VoiceAgentObservation(Observation):
    """
    Observation: call transcript(s), current prompt, failure analysis, and feedback.

    On reset: populated with call context (transcript, current prompt, failures).
    On step: additionally populated with score breakdown and feedback.
    """

    # --- Task context ---
    task_id: str = Field(default="", description="Active task identifier")
    task_difficulty: str = Field(
        default="easy", description="Task difficulty: easy, medium, hard"
    )
    scenario_id: str = Field(default="", description="Specific scenario within the task")

    # --- Call context (provided on reset) ---
    current_prompt: str = Field(
        default="",
        description="The system prompt that was used during the call(s)",
    )
    call_transcripts: List[Dict] = Field(
        default_factory=list,
        description=(
            "Call transcript(s). Each entry: "
            "{'session_id': str, 'turns': [{'role': 'agent'|'customer', "
            "'text': str, 'timestamp': str}], 'metadata': dict}"
        ),
    )
    call_metadata: Dict = Field(
        default_factory=dict,
        description=(
            "Aggregated call metadata: customer_name, intent, sentiment_start, "
            "sentiment_end, resolution_status, call_duration_s, stt_provider, "
            "llm_model, tts_provider"
        ),
    )

    # --- Failure analysis (provided on reset) ---
    failure_points: List[str] = Field(
        default_factory=list,
        description="Identified failure points in the call(s)",
    )
    required_improvements: List[str] = Field(
        default_factory=list,
        description="Specific improvements the optimized prompt must address",
    )
    policy_context: str = Field(
        default="",
        description="Relevant company policy the voice agent must follow",
    )

    # --- Feedback (populated after step) ---
    score_breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed score breakdown after optimization attempt",
    )
    feedback_message: str = Field(
        default="",
        description="Human-readable feedback on the prompt optimization",
    )

    # --- Turn tracking ---
    turn_number: int = Field(default=0, description="Current optimization turn")
    max_turns: int = Field(
        default=3, description="Max turns allowed for iterative optimization"
    )
