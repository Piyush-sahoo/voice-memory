# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Voice Agent Prompt Optimizer — Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import VoiceAgentAction, VoiceAgentObservation


class VoiceAgentEnv(
    EnvClient[VoiceAgentAction, VoiceAgentObservation, State]
):
    """
    Client for the Voice Agent Prompt Optimizer Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with VoiceAgentEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.task_id)
        ...     print(result.observation.call_transcripts)
        ...
        ...     result = env.step(VoiceAgentAction(
        ...         optimized_prompt="You are a professional voice agent...",
        ...         reasoning="Added specific business hours and direct answer policy",
        ...     ))
        ...     print(result.observation.score_breakdown)
        ...     print(result.observation.feedback_message)
    """

    def _step_payload(self, action: VoiceAgentAction) -> Dict:
        """
        Convert VoiceAgentAction to JSON payload for wire format.

        Args:
            action: VoiceAgentAction with optimized_prompt and reasoning.

        Returns:
            Dictionary for JSON encoding.
        """
        return {
            "optimized_prompt": action.optimized_prompt,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[VoiceAgentObservation]:
        """
        Parse server response into StepResult[VoiceAgentObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with VoiceAgentObservation.
        """
        obs_data = payload.get("observation", {})
        observation = VoiceAgentObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            # Task context
            task_id=obs_data.get("task_id", ""),
            task_difficulty=obs_data.get("task_difficulty", "easy"),
            scenario_id=obs_data.get("scenario_id", ""),
            # Call context
            current_prompt=obs_data.get("current_prompt", ""),
            call_transcripts=obs_data.get("call_transcripts", []),
            call_metadata=obs_data.get("call_metadata", {}),
            # Failure analysis
            failure_points=obs_data.get("failure_points", []),
            required_improvements=obs_data.get("required_improvements", []),
            policy_context=obs_data.get("policy_context", ""),
            # Feedback
            score_breakdown=obs_data.get("score_breakdown", {}),
            feedback_message=obs_data.get("feedback_message", ""),
            turn_number=obs_data.get("turn_number", 0),
            max_turns=obs_data.get("max_turns", 3),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
