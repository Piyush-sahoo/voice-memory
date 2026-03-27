"""
LiveKit Bridge — Captures call transcripts and feeds them to the OpenEnv environment.

This module provides:
  1. TranscriptCapture: hooks into a LiveKit AgentSession to record every
     STT utterance and LLM response during a call.
  2. format_for_openenv(): converts captured transcript into the OpenEnv
     scenario format for live prompt optimization.
  3. post_to_openenv(): sends a live transcript to the environment's
     /live-reset endpoint for RL-based prompt optimization.

Usage with your existing LiveKit agent:
    from livekit_bridge import TranscriptCapture, post_to_openenv

    capture = TranscriptCapture(phone_number="+919988776655")
    session = AgentSession(stt=..., llm=..., tts=..., tools=...)

    # Hook into session events
    capture.attach(session)

    # ... call happens ...

    # After call ends
    transcript = capture.get_transcript()
    result = await post_to_openenv(transcript, openenv_url="http://localhost:8000")
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("livekit-bridge")


@dataclass
class TranscriptTurn:
    """A single turn in the conversation."""
    role: str  # "agent" or "customer"
    text: str
    timestamp: str
    raw_event: Optional[str] = None


@dataclass
class TranscriptCapture:
    """
    Captures the full transcript from a LiveKit AgentSession.

    Hooks into session events to record every utterance (STT) from the customer
    and every response (LLM output) from the agent.
    """
    phone_number: str = ""
    customer_name: str = "Unknown"
    session_id: str = ""
    turns: List[TranscriptTurn] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    system_prompt: str = ""
    _attached: bool = False

    def attach(self, session: Any) -> None:
        """
        Attach to a LiveKit AgentSession to capture transcript events.

        The AgentSession emits events we can listen to:
        - agent_speech_committed: when the agent finishes speaking (TTS output)
        - user_speech_committed: when the user finishes speaking (STT output)
        """
        self.start_time = time.time()
        self._attached = True

        @session.on("user_speech_committed")
        def on_user_speech(msg):
            text = msg.content if hasattr(msg, "content") else str(msg)
            if text.strip():
                elapsed = time.time() - self.start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                self.turns.append(TranscriptTurn(
                    role="customer",
                    text=text.strip(),
                    timestamp=f"{mins:02d}:{secs:02d}",
                    raw_event="user_speech_committed",
                ))
                logger.info(f"[CUSTOMER] {text.strip()}")

        @session.on("agent_speech_committed")
        def on_agent_speech(msg):
            text = msg.content if hasattr(msg, "content") else str(msg)
            if text.strip():
                elapsed = time.time() - self.start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                self.turns.append(TranscriptTurn(
                    role="agent",
                    text=text.strip(),
                    timestamp=f"{mins:02d}:{secs:02d}",
                    raw_event="agent_speech_committed",
                ))
                logger.info(f"[AGENT] {text.strip()}")

    def finalize(self) -> None:
        """Mark the call as finished."""
        self.end_time = time.time()

    @property
    def duration_s(self) -> int:
        if self.start_time and self.end_time:
            return int(self.end_time - self.start_time)
        return 0

    def get_transcript(self) -> Dict:
        """
        Return the transcript in OpenEnv scenario format.
        """
        return {
            "session_id": self.session_id or f"live-{self.phone_number}-{int(time.time())}",
            "turns": [
                {
                    "role": t.role,
                    "text": t.text,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
            "metadata": {
                "customer_name": self.customer_name,
                "phone_number": self.phone_number,
                "call_duration_s": self.duration_s,
                "resolution_status": "completed",
                "stt_provider": "deepgram_nova3",
                "llm_model": "gpt-4o-mini",
                "tts_provider": os.getenv("TTS_PROVIDER", "openai"),
                "platform": "livekit",
                "sip_provider": "vobiz",
                "captured_at": datetime.utcnow().isoformat(),
            },
        }


def format_for_openenv(
    transcript: Dict,
    system_prompt: str = "",
    failure_points: Optional[List[str]] = None,
    task_id: str = "live_call",
) -> Dict:
    """
    Format a captured transcript into a full OpenEnv live-reset payload.

    Args:
        transcript: Output from TranscriptCapture.get_transcript()
        system_prompt: The system prompt that was used during the call
        failure_points: Optional manually identified failure points
        task_id: Task identifier for the live scenario

    Returns:
        Dict ready to POST to /live-reset
    """
    return {
        "task_id": task_id,
        "task_difficulty": "live",
        "current_prompt": system_prompt,
        "call_transcripts": [transcript],
        "call_metadata": transcript.get("metadata", {}),
        "failure_points": failure_points or [],
        "policy_context": "",
    }


async def post_to_openenv(
    transcript: Dict,
    system_prompt: str = "",
    failure_points: Optional[List[str]] = None,
    openenv_url: str = "http://localhost:8000",
) -> Dict:
    """
    Post a live transcript to the OpenEnv environment for prompt optimization.

    Args:
        transcript: Output from TranscriptCapture.get_transcript()
        system_prompt: The system prompt used during the call
        failure_points: Optional failure point annotations
        openenv_url: Base URL of the OpenEnv server

    Returns:
        The observation from the environment's /live-reset endpoint
    """
    payload = format_for_openenv(
        transcript=transcript,
        system_prompt=system_prompt,
        failure_points=failure_points,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{openenv_url}/live-reset",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def optimize_prompt_with_openai(
    transcript: Dict,
    current_prompt: str,
    openenv_url: str = "http://localhost:8000",
    openai_api_key: str = "",
) -> Dict:
    """
    Full pipeline: take a live transcript, analyze it with OpenAI,
    submit the optimized prompt to OpenEnv, and return the graded result.

    Args:
        transcript: Output from TranscriptCapture.get_transcript()
        current_prompt: The system prompt used during the call
        openenv_url: Base URL of the OpenEnv server
        openai_api_key: OpenAI API key for prompt optimization

    Returns:
        Dict with optimized_prompt, score, feedback
    """
    from openai import AsyncOpenAI

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for live optimization")

    client = AsyncOpenAI(api_key=api_key)

    # Step 1: Reset the environment with the live transcript
    reset_result = await post_to_openenv(
        transcript=transcript,
        system_prompt=current_prompt,
        openenv_url=openenv_url,
    )

    obs = reset_result.get("observation", reset_result)

    # Step 2: Use OpenAI to generate an optimized prompt
    transcript_text = ""
    for t_data in transcript.get("turns", []):
        transcript_text += f"[{t_data['role']}]: {t_data['text']}\n"

    system_msg = (
        "You are an expert at optimizing system prompts for AI voice agents. "
        "Analyze this real call transcript and the current system prompt. "
        "Identify what went wrong and write an improved prompt."
    )
    user_msg = (
        f"CURRENT SYSTEM PROMPT:\n{current_prompt}\n\n"
        f"CALL TRANSCRIPT:\n{transcript_text}\n\n"
        f"FAILURE POINTS: {obs.get('failure_points', [])}\n\n"
        "Write an improved system prompt that fixes the issues. "
        "Be specific with rules, data, and procedures. Use numbered steps."
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1200,
        temperature=0.3,
    )

    optimized_prompt = response.choices[0].message.content or ""
    reasoning = f"AI-optimized based on live call transcript ({len(transcript.get('turns', []))} turns)"

    # Step 3: Submit the optimized prompt to OpenEnv for grading (use /live-step for stateful session)
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        step_response = await http_client.post(
            f"{openenv_url}/live-step",
            json={
                "optimized_prompt": optimized_prompt,
                "reasoning": reasoning,
            },
        )
        step_response.raise_for_status()
        step_result = step_response.json()

    return {
        "optimized_prompt": optimized_prompt,
        "reasoning": reasoning,
        "score": step_result.get("observation", {}).get("score_breakdown", {}).get("score"),
        "feedback": step_result.get("observation", {}).get("feedback_message", ""),
        "full_result": step_result,
    }
