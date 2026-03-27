#!/usr/bin/env python3
"""
Baseline inference script for the Voice Agent Prompt Optimizer.

Uses the OpenAI API to run a model against all 3 tasks and produce
reproducible baseline scores.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py
    OPENAI_API_KEY=sk-... python baseline.py --task faq_resolution
    python baseline.py --deterministic  # no API key needed
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure server modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.graders import generate_feedback, grade_prompt
from server.scenarios import ALL_SCENARIOS, TASK_DEFINITIONS


def run_deterministic_baseline(task_filter: str = None) -> dict:
    """Run deterministic baseline with hand-crafted prompts."""
    results = {}

    for task_id, scenarios in ALL_SCENARIOS.items():
        if task_filter and task_id != task_filter:
            continue

        task_scores = []
        for scenario in scenarios:
            prompt = _build_baseline_prompt(scenario)
            grade = grade_prompt(prompt, scenario)
            task_scores.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "score": grade["score"],
                    "improvement_score": grade["improvement_score"],
                    "quality_score": grade["quality_score"],
                    "policy_penalty": grade["policy_penalty"],
                }
            )

        avg = sum(s["score"] for s in task_scores) / len(task_scores) if task_scores else 0
        results[task_id] = {
            "average_score": round(avg, 4),
            "difficulty": TASK_DEFINITIONS[task_id]["difficulty"],
            "scenarios": task_scores,
        }

    return results


async def run_openai_baseline(api_key: str, task_filter: str = None) -> dict:
    """Run baseline using OpenAI GPT-4o-mini."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    results = {}

    for task_id, scenarios in ALL_SCENARIOS.items():
        if task_filter and task_id != task_filter:
            continue

        task_scores = []
        for scenario in scenarios:
            system_msg = (
                "You are an expert at optimizing system prompts for AI voice agents. "
                "Given a call transcript, the current prompt, and failure analysis, "
                "write an improved system prompt that addresses all identified issues. "
                "Be specific: include actual data, numbers, and structured rules."
            )

            transcript_summary = ""
            for t in scenario["call_transcripts"]:
                transcript_summary += f"\n--- Session: {t['session_id']} ---\n"
                for turn in t["turns"]:
                    transcript_summary += f"[{turn['role']}]: {turn['text']}\n"

            user_msg = (
                f"CURRENT SYSTEM PROMPT:\n{scenario['current_prompt']}\n\n"
                f"CALL TRANSCRIPT(S):{transcript_summary}\n\n"
                f"FAILURE POINTS:\n" + "\n".join(f"- {fp}" for fp in scenario["failure_points"]) + "\n\n"
                f"REQUIRED IMPROVEMENTS:\n" + "\n".join(f"- {ri}" for ri in scenario["required_improvements"]) + "\n\n"
                f"COMPANY POLICY:\n{scenario['policy_context']}\n\n"
                "Write an improved system prompt. Include specific data from the policy. "
                "Use numbered rules. Be concise but thorough."
            )

            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=1200,
                    temperature=0.2,
                )
                optimized = response.choices[0].message.content or ""
            except Exception as e:
                print(f"  [ERROR] {scenario['scenario_id']}: {e}")
                optimized = _build_baseline_prompt(scenario)

            grade = grade_prompt(optimized, scenario)
            task_scores.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "score": grade["score"],
                    "improvement_score": grade["improvement_score"],
                    "quality_score": grade["quality_score"],
                    "policy_penalty": grade["policy_penalty"],
                }
            )
            print(f"  {scenario['scenario_id']}: {grade['score']:.4f}")

        avg = sum(s["score"] for s in task_scores) / len(task_scores) if task_scores else 0
        results[task_id] = {
            "average_score": round(avg, 4),
            "difficulty": TASK_DEFINITIONS[task_id]["difficulty"],
            "scenarios": task_scores,
        }

    return results


def _build_baseline_prompt(scenario: dict) -> str:
    """Build a hand-crafted improved prompt from scenario data."""
    parts = [
        "You are a professional and empathetic voice assistant from Vobiz.",
        "",
        "Key rules:",
    ]
    policy = scenario.get("policy_context", "")
    if policy:
        for i, sentence in enumerate(policy.split(". "), 1):
            sentence = sentence.strip()
            if sentence:
                parts.append(f"{i}. {sentence}.")
    parts.append("")
    parts.append("When handling calls:")
    for improvement in scenario.get("required_improvements", []):
        instruction = improvement.replace("_", " ").capitalize()
        parts.append(f"- {instruction}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Voice Agent Prompt Optimizer - Baseline")
    parser.add_argument("--task", type=str, default=None, help="Filter to specific task")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic baseline (no API)")
    args = parser.parse_args()

    print("=" * 60)
    print("Voice Agent Prompt Optimizer — Baseline Inference")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if args.deterministic or not api_key:
        if not args.deterministic:
            print("[INFO] No OPENAI_API_KEY found. Running deterministic baseline.")
        print("[MODE] Deterministic baseline (hand-crafted prompts)")
        print("-" * 60)
        results = run_deterministic_baseline(args.task)
    else:
        print(f"[MODE] OpenAI GPT-4o-mini baseline")
        print("-" * 60)
        results = asyncio.run(run_openai_baseline(api_key, args.task))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for task_id, data in results.items():
        diff = data["difficulty"]
        avg = data["average_score"]
        print(f"\n  {task_id} ({diff}): {avg:.4f}")
        for s in data["scenarios"]:
            print(f"    {s['scenario_id']}: {s['score']:.4f}")

    print("\n" + "-" * 60)
    overall_scores = [d["average_score"] for d in results.values()]
    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"  OVERALL AVERAGE: {overall:.4f}")
    print("=" * 60)

    # Also output as JSON for machine reading
    output = {
        "baseline_type": "deterministic" if (args.deterministic or not api_key) else "openai_gpt4o_mini",
        "overall_average": round(overall, 4),
        "scores": results,
    }
    print(f"\n[JSON]\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
