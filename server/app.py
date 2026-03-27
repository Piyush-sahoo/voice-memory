# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Voice Agent Prompt Optimizer Environment.

Standard OpenEnv endpoints (provided by create_app):
    - POST /reset, POST /step, GET /state, GET /schema, WS /ws
    - GET /health, GET /web, GET /docs

Custom hackathon-required endpoints:
    - GET  /tasks    — list of tasks + action schema
    - POST /grader   — grader score after episode completion
    - POST /baseline — trigger baseline inference and return scores
"""

import os
import random

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import VoiceAgentAction, VoiceAgentObservation
    from .voice_agent_env_environment import VoiceAgentEnvironment
    from .graders import generate_feedback, grade_prompt
    from .scenarios import ALL_SCENARIOS, TASK_DEFINITIONS
except (ImportError, ModuleNotFoundError):
    from models import VoiceAgentAction, VoiceAgentObservation
    from server.voice_agent_env_environment import VoiceAgentEnvironment
    from server.graders import generate_feedback, grade_prompt
    from server.scenarios import ALL_SCENARIOS, TASK_DEFINITIONS

# ─── Create the OpenEnv app ───
app = create_app(
    VoiceAgentEnvironment,
    VoiceAgentAction,
    VoiceAgentObservation,
    env_name="voice_agent_env",
    max_concurrent_envs=10,
)


# ─── Custom endpoints required by hackathon ───

# Shared environment instance for live-reset (stateful endpoint)
_live_env = VoiceAgentEnvironment()


@app.post("/live-reset")
async def live_reset(payload: dict):
    """
    Reset the environment with a live call transcript from LiveKit.

    This endpoint accepts real transcripts captured during live calls
    and loads them into the environment for prompt optimization.

    Payload:
        {
            "task_id": "live_call",
            "task_difficulty": "live",
            "current_prompt": "You are a helpful...",
            "call_transcripts": [{"session_id": "...", "turns": [...], "metadata": {...}}],
            "call_metadata": {...},
            "failure_points": [...],  // optional, auto-detected if empty
            "policy_context": ""
        }
    """
    obs = _live_env.live_reset(payload)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/live-step")
async def live_step(payload: dict):
    """
    Submit an optimized prompt to the live environment session.

    Uses the same _live_env instance as /live-reset so state is preserved.

    Payload:
        {
            "optimized_prompt": "You are a professional...",
            "reasoning": "Added specific pricing..."
        }
    """
    action = VoiceAgentAction(
        optimized_prompt=payload.get("optimized_prompt", ""),
        reasoning=payload.get("reasoning", ""),
    )
    obs = _live_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/tasks")
async def get_tasks():
    """
    Return list of tasks and the action schema.

    Required endpoint: /tasks
    Returns task definitions and the fields required for an action in a step.
    """
    action_schema = VoiceAgentAction.model_json_schema()
    return {
        "tasks": list(TASK_DEFINITIONS.values()),
        "action_schema": action_schema,
    }


@app.post("/grader")
async def run_grader(optimized_prompt: str = "", task_id: str = "faq_resolution", scenario_id: str = ""):
    """
    Return grader score for a submitted prompt against a scenario.

    Required endpoint: /grader
    Can be called independently to evaluate any prompt.
    """
    if task_id not in ALL_SCENARIOS:
        return {"error": f"Unknown task_id: {task_id}. Available: {list(ALL_SCENARIOS.keys())}"}

    scenarios = ALL_SCENARIOS[task_id]
    if scenario_id:
        matching = [s for s in scenarios if s["scenario_id"] == scenario_id]
        scenario = matching[0] if matching else scenarios[0]
    else:
        scenario = scenarios[0]

    if not optimized_prompt:
        return {"error": "optimized_prompt is required"}

    result = grade_prompt(optimized_prompt, scenario)
    feedback = generate_feedback(result, scenario)

    return {
        "task_id": task_id,
        "scenario_id": scenario["scenario_id"],
        "score": result["score"],
        "breakdown": result,
        "feedback": feedback,
    }


@app.post("/baseline")
async def run_baseline():
    """
    Trigger baseline inference and return scores for all 3 tasks.

    Required endpoint: /baseline
    Uses OpenAI API to generate optimized prompts for each task,
    then grades them.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key:
        # Run actual inference with OpenAI
        try:
            return await _run_openai_baseline(api_key)
        except Exception as e:
            # Fall back to deterministic baseline
            return _run_deterministic_baseline(note=f"OpenAI failed ({e}), using deterministic baseline")
    else:
        return _run_deterministic_baseline(note="No OPENAI_API_KEY, using deterministic baseline")


def _run_deterministic_baseline(note: str = "") -> dict:
    """
    Run a deterministic baseline without any LLM — just a hand-crafted prompt.
    Useful when no API key is available (e.g., during automated evaluation).
    """
    results = {}

    for task_id, scenarios in ALL_SCENARIOS.items():
        task_scores = []
        for scenario in scenarios:
            # Build a basic improved prompt from the policy context and failure points
            baseline_prompt = _build_baseline_prompt(scenario)
            grade = grade_prompt(baseline_prompt, scenario)
            task_scores.append(grade["score"])

        avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
        results[task_id] = {
            "average_score": round(avg_score, 4),
            "per_scenario_scores": [round(s, 4) for s in task_scores],
            "num_scenarios": len(task_scores),
        }

    return {
        "baseline_type": "deterministic",
        "note": note,
        "scores": results,
    }


async def _run_openai_baseline(api_key: str) -> dict:
    """
    Run baseline using OpenAI to generate optimized prompts.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    results = {}

    for task_id, scenarios in ALL_SCENARIOS.items():
        task_scores = []
        # Run on first 3 scenarios per task for speed
        for scenario in scenarios[:3]:
            system_msg = (
                "You are an expert at optimizing system prompts for AI voice agents. "
                "Given a call transcript, the current prompt, and failure analysis, "
                "write an improved system prompt that addresses all identified issues."
            )
            user_msg = (
                f"Current prompt: {scenario['current_prompt']}\n\n"
                f"Failure points: {scenario['failure_points']}\n\n"
                f"Required improvements: {scenario['required_improvements']}\n\n"
                f"Policy context: {scenario['policy_context']}\n\n"
                "Write an improved system prompt that fixes all the issues. "
                "Be specific, include actual data from the policy, and structure clearly."
            )

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            optimized = response.choices[0].message.content or ""
            grade = grade_prompt(optimized, scenario)
            task_scores.append(grade["score"])

        avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
        results[task_id] = {
            "average_score": round(avg_score, 4),
            "per_scenario_scores": [round(s, 4) for s in task_scores],
            "num_scenarios": len(task_scores),
        }

    return {
        "baseline_type": "openai_gpt4o_mini",
        "scores": results,
    }


def _build_baseline_prompt(scenario: dict) -> str:
    """
    Build a basic improved prompt from scenario data.
    Used as deterministic baseline.
    """
    parts = [
        "You are a professional and empathetic voice assistant from Vobiz.",
        "",
        "Key rules:",
    ]

    # Add policy context as rules
    policy = scenario.get("policy_context", "")
    if policy:
        for i, sentence in enumerate(policy.split(". "), 1):
            sentence = sentence.strip()
            if sentence:
                parts.append(f"{i}. {sentence}.")

    parts.append("")
    parts.append("When handling calls:")

    # Add improvements as instructions
    for improvement in scenario.get("required_improvements", []):
        instruction = improvement.replace("_", " ").capitalize()
        parts.append(f"- {instruction}")

    return "\n".join(parts)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
