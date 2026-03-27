#!/usr/bin/env python3
"""
=============================================================================
  VOICE AGENT PROMPT OPTIMIZER — Full Test Suite & Submission Validator
=============================================================================

  Runs every test, validates every endpoint, checks every file,
  and produces a final submission readiness report.

  Usage:
      python test_all.py                  # Full test (local server)
      python test_all.py --live           # Also test deployed HF Space
      python test_all.py --with-openai    # Also test OpenAI baseline
      python test_all.py --full           # Everything (local + HF + OpenAI)

=============================================================================
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─── Config ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
LOCAL_URL = "http://localhost:8000"
HF_URL = "https://piyushs-voice-agent-env.hf.space"
REQUIRED_FILES = [
    "models.py",
    "client.py",
    "__init__.py",
    "baseline.py",
    "openenv.yaml",
    "pyproject.toml",
    "README.md",
    "livekit_bridge.py",
    "server/app.py",
    "server/voice_agent_env_environment.py",
    "server/scenarios.py",
    "server/graders.py",
    "server/rewards.py",
    "server/policy_rules.py",
    "server/Dockerfile",
    "server/requirements.txt",
    "server/__init__.py",
    "demo/run_live_call.py",
]

# ─── Styling ────────────────────────────────────────────────
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    CHECK = "\033[92m[PASS]\033[0m"
    CROSS = "\033[91m[FAIL]\033[0m"
    WARN = "\033[93m[WARN]\033[0m"
    SKIP = "\033[2m[SKIP]\033[0m"
    RUNNING = "\033[96m[....]"


def banner(text: str, char: str = "=", width: int = 70):
    print(f"\n{C.CYAN}{C.BOLD}{char * width}")
    print(f"  {text}")
    print(f"{char * width}{C.RESET}")


def section(text: str):
    print(f"\n{C.MAGENTA}{C.BOLD}--- {text} ---{C.RESET}")


def result(name: str, passed: bool, detail: str = ""):
    icon = C.CHECK if passed else C.CROSS
    det = f"  {C.DIM}{detail}{C.RESET}" if detail else ""
    print(f"  {icon} {name}{det}")
    return passed


def warn(name: str, detail: str = ""):
    det = f"  {C.DIM}{detail}{C.RESET}" if detail else ""
    print(f"  {C.WARN} {name}{det}")


def skip(name: str, detail: str = ""):
    det = f"  {C.DIM}{detail}{C.RESET}" if detail else ""
    print(f"  {C.SKIP} {name}{det}")


# ─── Test Helpers ───────────────────────────────────────────
import urllib.request
import urllib.error


def http_get(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, data: dict = None, timeout: int = 20) -> dict:
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ─── Test Groups ────────────────────────────────────────────

def test_file_structure() -> list:
    """Check all required files exist."""
    section("FILE STRUCTURE")
    results = []
    for f in REQUIRED_FILES:
        path = PROJECT_ROOT / f
        ok = path.exists()
        results.append(result(f, ok, str(path) if not ok else ""))
    # Check .gitignore protects .env
    gi = PROJECT_ROOT / ".gitignore"
    if gi.exists():
        content = gi.read_text()
        has_env = ".env" in content
        results.append(result(".gitignore protects .env", has_env))
    else:
        results.append(result(".gitignore exists", False, "MISSING"))
    return results


def test_imports() -> list:
    """Verify all modules import correctly."""
    section("IMPORTS")
    results = []

    tests = [
        ("models", "from models import VoiceAgentAction, VoiceAgentObservation"),
        ("client (package import)", "import importlib; importlib.import_module('client')"),
        ("environment", "from server.voice_agent_env_environment import VoiceAgentEnvironment"),
        ("scenarios", "from server.scenarios import ALL_SCENARIOS, TASK_DEFINITIONS"),
        ("graders", "from server.graders import grade_prompt, generate_feedback"),
        ("rewards", "from server.rewards import compute_reward"),
        ("policy_rules", "from server.policy_rules import check_policy_violations"),
        ("app", "from server.app import app"),
        ("livekit_bridge", "from livekit_bridge import TranscriptCapture, format_for_openenv"),
    ]

    for name, import_str in tests:
        try:
            exec(import_str)
            results.append(result(f"import {name}", True))
        except Exception as e:
            results.append(result(f"import {name}", False, str(e)[:80]))

    return results


def test_scenarios() -> list:
    """Validate scenario data integrity."""
    section("SCENARIO DATA")
    results = []

    from server.scenarios import ALL_SCENARIOS, TASK_DEFINITIONS

    results.append(result("3 task types defined", len(ALL_SCENARIOS) == 3, f"got {len(ALL_SCENARIOS)}"))

    total = 0
    for task_id, scenarios in ALL_SCENARIOS.items():
        count = len(scenarios)
        total += count
        results.append(result(f"{task_id}: {count} scenarios", count >= 3))

        for s in scenarios:
            required_keys = ["scenario_id", "task_id", "current_prompt", "call_transcripts",
                             "failure_points", "required_improvements", "policy_context"]
            missing = [k for k in required_keys if k not in s]
            if missing:
                results.append(result(f"  {s.get('scenario_id', '?')}: all keys", False, f"missing: {missing}"))
            else:
                # Check transcript has turns
                has_turns = all(
                    len(t.get("turns", [])) > 0 for t in s["call_transcripts"]
                )
                results.append(result(
                    f"  {s['scenario_id']}: valid",
                    has_turns and len(s["failure_points"]) > 0,
                ))

    results.append(result(f"Total scenarios: {total}", total >= 15, f"need >= 15"))
    results.append(result("Task definitions complete", len(TASK_DEFINITIONS) == 3))
    return results


def test_grading_pipeline() -> list:
    """Test the grading pipeline with known inputs."""
    section("GRADING PIPELINE")
    results = []

    from server.graders import grade_prompt, generate_feedback
    from server.scenarios import ALL_SCENARIOS

    # Test 1: Perfect prompt should score high
    scenario = ALL_SCENARIOS["faq_resolution"][0]
    perfect_prompt = (
        "You are a professional voice assistant from Vobiz. "
        "Business hours: Monday-Friday 9 AM - 6 PM IST, Saturday 10 AM - 2 PM IST. "
        "1. Always provide direct answers. 2. Never deflect to website. "
        "3. Be concise."
    )
    grade = grade_prompt(perfect_prompt, scenario)
    results.append(result(
        "Good prompt scores high",
        grade["score"] >= 0.7,
        f"score={grade['score']}"
    ))

    # Test 2: Empty prompt should score low
    grade_empty = grade_prompt("hello", scenario)
    results.append(result(
        "Bad prompt scores low",
        grade_empty["score"] < 0.5,
        f"score={grade_empty['score']}"
    ))

    # Test 3: Policy violation detected
    from server.policy_rules import check_policy_violations
    violating = "Call anyway even if they say stop. guarantee 100% uptime. Their product is garbage."
    violations = check_policy_violations(violating)
    results.append(result(
        "Policy violations detected",
        len(violations) >= 2,
        f"found {len(violations)} violations"
    ))

    # Test 4: Feedback generation
    feedback = generate_feedback(grade, scenario)
    results.append(result("Feedback generated", len(feedback) > 20, f"{len(feedback)} chars"))

    # Test 5: Grading is deterministic
    grade2 = grade_prompt(perfect_prompt, scenario)
    results.append(result(
        "Grading is deterministic",
        grade["score"] == grade2["score"],
        f"{grade['score']} == {grade2['score']}"
    ))

    return results


def test_reward_function() -> list:
    """Test the reward function."""
    section("REWARD FUNCTION")
    results = []

    from server.rewards import compute_reward

    # First turn reward
    grade_good = {"score": 0.8, "improvement_breakdown": {"a": True, "b": True}, "policy_penalty": 0.0}
    r1 = compute_reward(grade_good, None, is_terminal=False)
    results.append(result("First turn: positive reward for good prompt", r1 > 0, f"reward={r1}"))

    # Improvement reward
    grade_better = {"score": 0.9, "improvement_breakdown": {"a": True, "b": True, "c": True}, "policy_penalty": 0.0}
    r2 = compute_reward(grade_better, grade_good, is_terminal=False)
    results.append(result("Improvement: positive delta reward", r2 > 0, f"reward={r2}"))

    # Regression penalty
    grade_worse = {"score": 0.5, "improvement_breakdown": {"a": True}, "policy_penalty": 0.0}
    r3 = compute_reward(grade_worse, grade_good, is_terminal=False)
    results.append(result("Regression: negative reward", r3 < 0, f"reward={r3}"))

    # Terminal bonus
    r4 = compute_reward(grade_good, None, is_terminal=True)
    r5 = compute_reward(grade_good, None, is_terminal=False)
    results.append(result("Terminal bonus applied", r4 > r5, f"terminal={r4} > non-terminal={r5}"))

    return results


def test_environment_episode() -> list:
    """Test a full 3-turn episode."""
    section("FULL EPISODE (3 TURNS)")
    results = []

    from server.voice_agent_env_environment import VoiceAgentEnvironment
    from models import VoiceAgentAction

    env = VoiceAgentEnvironment()

    # Reset
    obs = env.reset(task_id="faq_resolution", scenario_id="faq_01")
    results.append(result("reset() returns observation", obs.task_id == "faq_resolution"))
    results.append(result("done=False on reset", obs.done is False))
    results.append(result("Has transcript", len(obs.call_transcripts) > 0))
    results.append(result("Has failure points", len(obs.failure_points) > 0))
    results.append(result("turn_number=0", obs.turn_number == 0))

    # Step 1
    action = VoiceAgentAction(
        optimized_prompt="You are a voice assistant. Business hours: Mon-Fri 9-6 IST. 1. Direct answers. 2. No website deflection.",
        reasoning="Added hours",
    )
    obs = env.step(action)
    results.append(result("step() returns score", obs.score_breakdown.get("score", 0) > 0))
    results.append(result("step() returns reward", obs.reward is not None))
    results.append(result("turn_number=1", obs.turn_number == 1))
    results.append(result("done=False after turn 1", obs.done is False))

    # Step 2
    obs = env.step(action)
    results.append(result("turn_number=2", obs.turn_number == 2))
    results.append(result("done=False after turn 2", obs.done is False))

    # Step 3 (final)
    obs = env.step(action)
    results.append(result("turn_number=3", obs.turn_number == 3))
    results.append(result("done=True after turn 3", obs.done is True))

    # Step 4 (should be rejected)
    obs = env.step(action)
    results.append(result("Rejects step after done", "already finished" in obs.feedback_message.lower()))

    # State
    state = env.state
    results.append(result("State tracks steps", state.step_count == 3))
    results.append(result("Episode scores recorded", len(env.get_episode_scores()) == 3))

    return results


def test_live_reset() -> list:
    """Test the live transcript injection."""
    section("LIVE RESET (Auto-Failure Detection)")
    results = []

    from server.voice_agent_env_environment import VoiceAgentEnvironment

    env = VoiceAgentEnvironment()
    obs = env.live_reset({
        "task_id": "live_call",
        "task_difficulty": "live",
        "current_prompt": "You are a helpful voice assistant.",
        "call_transcripts": [{
            "session_id": "test-live-001",
            "turns": [
                {"role": "customer", "text": "What are your prices?", "timestamp": "00:00"},
                {"role": "agent", "text": "I think we have great plans! Check our website.", "timestamp": "00:03"},
                {"role": "customer", "text": "Useless. Bye.", "timestamp": "00:07"},
            ],
            "metadata": {"resolution_status": "unresolved"},
        }],
        "call_metadata": {"resolution_status": "unresolved"},
        "failure_points": [],
        "policy_context": "",
    })

    results.append(result("live_reset() works", obs.task_id == "live_call"))
    results.append(result("Auto-detected failures", len(obs.failure_points) >= 2, f"found {len(obs.failure_points)}"))

    expected_detections = ["website", "vague", "frustrat", "unresolved"]
    found = [e for e in expected_detections if any(e in fp.lower() for fp in obs.failure_points)]
    results.append(result(
        f"Detected patterns: {len(found)}/{len(expected_detections)}",
        len(found) >= 2,
        ", ".join(found),
    ))

    return results


def test_all_tasks_all_scenarios() -> list:
    """Run grading on ALL scenarios to ensure none crash."""
    section("ALL SCENARIOS GRADING (crash test)")
    results = []

    from server.graders import grade_prompt
    from server.scenarios import ALL_SCENARIOS

    test_prompt = (
        "You are a professional and empathetic voice assistant from Vobiz. "
        "1. Always provide direct answers with specific data. "
        "2. Acknowledge customer frustration before responding. "
        "3. Never deflect to website. 4. Follow company policy strictly."
    )

    all_pass = True
    scores = {}
    for task_id, scenarios in ALL_SCENARIOS.items():
        task_scores = []
        for s in scenarios:
            try:
                grade = grade_prompt(test_prompt, s)
                task_scores.append(grade["score"])
            except Exception as e:
                results.append(result(f"  {s['scenario_id']}", False, str(e)[:60]))
                all_pass = False

        avg = sum(task_scores) / len(task_scores) if task_scores else 0
        scores[task_id] = avg
        results.append(result(f"{task_id}: avg={avg:.3f}", len(task_scores) == len(scenarios)))

    results.append(result("All 15 scenarios grade without error", all_pass))
    return results


def test_server_endpoints(base_url: str, label: str) -> list:
    """Test all HTTP endpoints on a running server."""
    section(f"HTTP ENDPOINTS ({label})")
    results = []

    # Health
    try:
        data = http_get(f"{base_url}/health")
        results.append(result("/health", data.get("status") == "healthy"))
    except Exception as e:
        results.append(result("/health", False, str(e)[:60]))
        return results  # server down, skip rest

    # Tasks
    try:
        data = http_get(f"{base_url}/tasks")
        tasks = data.get("tasks", [])
        results.append(result("/tasks", len(tasks) == 3, f"{len(tasks)} tasks"))
        results.append(result("/tasks has action_schema", "action_schema" in data))
    except Exception as e:
        results.append(result("/tasks", False, str(e)[:60]))

    # Schema
    try:
        data = http_get(f"{base_url}/schema")
        results.append(result("/schema", "action" in data and "observation" in data))
    except Exception as e:
        results.append(result("/schema", False, str(e)[:60]))

    # Metadata
    try:
        data = http_get(f"{base_url}/metadata")
        results.append(result("/metadata", "name" in data))
    except Exception as e:
        results.append(result("/metadata", False, str(e)[:60]))

    # Reset
    try:
        data = http_post(f"{base_url}/reset")
        obs = data.get("observation", {})
        results.append(result("/reset", obs.get("task_id", "") != ""))
    except Exception as e:
        results.append(result("/reset", False, str(e)[:60]))

    # Grader
    try:
        data = http_post(f"{base_url}/grader?task_id=faq_resolution&scenario_id=faq_01&optimized_prompt=Business+hours+Monday+9+AM")
        results.append(result("/grader", "score" in data, f"score={data.get('score')}"))
    except Exception as e:
        results.append(result("/grader", False, str(e)[:60]))

    # Baseline
    try:
        data = http_post(f"{base_url}/baseline", timeout=60)
        results.append(result("/baseline", "scores" in data))
    except Exception as e:
        results.append(result("/baseline", False, str(e)[:60]))

    # Live-reset
    try:
        data = http_post(f"{base_url}/live-reset", {
            "task_id": "live_call", "task_difficulty": "live",
            "current_prompt": "Test prompt",
            "call_transcripts": [{"session_id": "t", "turns": [{"role": "agent", "text": "Visit our website.", "timestamp": "00:00"}], "metadata": {"resolution_status": "unresolved"}}],
            "call_metadata": {"resolution_status": "unresolved"},
            "failure_points": [], "policy_context": "",
        })
        obs = data.get("observation", {})
        results.append(result("/live-reset", obs.get("task_id") == "live_call"))
    except Exception as e:
        results.append(result("/live-reset", False, str(e)[:60]))

    # Live-step
    try:
        data = http_post(f"{base_url}/live-step", {
            "optimized_prompt": "You are a professional voice assistant. 1. Answer directly. 2. Never deflect.",
            "reasoning": "test",
        })
        results.append(result("/live-step", data.get("reward") is not None))
    except Exception as e:
        results.append(result("/live-step", False, str(e)[:60]))

    # OpenAPI
    try:
        data = http_get(f"{base_url}/openapi.json")
        results.append(result("/openapi.json", "info" in data))
    except Exception as e:
        results.append(result("/openapi.json", False, str(e)[:60]))

    return results


def test_openai_baseline() -> list:
    """Run actual OpenAI baseline (costs API credits)."""
    section("OPENAI BASELINE (live API)")
    results = []

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        skip("OpenAI baseline", "OPENAI_API_KEY not set")
        return results

    try:
        from baseline import run_openai_baseline
        scores = asyncio.run(run_openai_baseline(api_key, task_filter="faq_resolution"))
        faq = scores.get("faq_resolution", {})
        avg = faq.get("average_score", 0)
        results.append(result(f"OpenAI baseline faq_resolution", avg > 0.5, f"avg={avg:.4f}"))
    except Exception as e:
        results.append(result("OpenAI baseline", False, str(e)[:80]))

    return results


def test_submission_checklist() -> list:
    """Final submission checklist per hackathon rules."""
    section("SUBMISSION CHECKLIST")
    results = []

    # 1. openenv.yaml
    yaml_path = PROJECT_ROOT / "openenv.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text()
        results.append(result("openenv.yaml: spec_version", "spec_version" in content))
        results.append(result("openenv.yaml: name", "name:" in content))
        results.append(result("openenv.yaml: runtime: fastapi", "fastapi" in content))
    else:
        results.append(result("openenv.yaml exists", False))

    # 2. pyproject.toml
    toml_path = PROJECT_ROOT / "pyproject.toml"
    if toml_path.exists():
        content = toml_path.read_text()
        results.append(result("pyproject.toml: openenv-core dep", "openenv-core" in content))
        results.append(result("pyproject.toml: python >= 3.10", "3.10" in content))
    else:
        results.append(result("pyproject.toml exists", False))

    # 3. Dockerfile
    df = PROJECT_ROOT / "server" / "Dockerfile"
    results.append(result("Dockerfile exists", df.exists()))

    # 4. README
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        content = readme.read_text()
        checks = [
            ("README: has title", "Voice Agent" in content),
            ("README: has tasks table", "faq_resolution" in content),
            ("README: has action space", "optimized_prompt" in content),
            ("README: has setup instructions", "uvicorn" in content or "pip install" in content),
            ("README: has HF Spaces tag", "openenv" in content),
            ("README: has baseline scores", "0.98" in content),
        ]
        for name, ok in checks:
            results.append(result(name, ok))
    else:
        results.append(result("README.md exists", False))

    # 5. .env NOT in tracked files (security)
    env_path = PROJECT_ROOT / ".env"
    gi_path = PROJECT_ROOT / ".gitignore"
    if env_path.exists() and gi_path.exists():
        results.append(result(".env protected by .gitignore", ".env" in gi_path.read_text()))

    # 6. uv.lock
    results.append(result("uv.lock generated", (PROJECT_ROOT / "uv.lock").exists()))

    # 7. 3+ tasks
    from server.scenarios import ALL_SCENARIOS
    results.append(result("3+ tasks defined", len(ALL_SCENARIOS) >= 3))

    # 8. 15+ scenarios
    total = sum(len(v) for v in ALL_SCENARIOS.values())
    results.append(result(f"15+ scenarios ({total})", total >= 15))

    return results


# ─── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voice Agent Prompt Optimizer — Full Test Suite")
    parser.add_argument("--live", action="store_true", help="Also test deployed HF Space")
    parser.add_argument("--with-openai", action="store_true", help="Also run OpenAI baseline")
    parser.add_argument("--full", action="store_true", help="Run everything")
    args = parser.parse_args()

    if args.full:
        args.live = True
        args.with_openai = True

    os.chdir(PROJECT_ROOT)

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        pass

    start_time = time.time()
    all_results = []

    banner("VOICE AGENT PROMPT OPTIMIZER  --  TEST SUITE")
    print(f"  {C.DIM}Project: {PROJECT_ROOT}")
    print(f"  Local:  {LOCAL_URL}")
    print(f"  HF:     {HF_URL}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")

    # ── Unit Tests (no server needed) ──
    banner("UNIT TESTS", char="-")

    all_results.extend(test_file_structure())
    all_results.extend(test_imports())
    all_results.extend(test_scenarios())
    all_results.extend(test_grading_pipeline())
    all_results.extend(test_reward_function())
    all_results.extend(test_environment_episode())
    all_results.extend(test_live_reset())
    all_results.extend(test_all_tasks_all_scenarios())

    # ── Server Tests (start local server) ──
    banner("SERVER TESTS (local)", char="-")

    server_proc = None
    try:
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
        )
        time.sleep(4)
        all_results.extend(test_server_endpoints(LOCAL_URL, "localhost:8000"))
    except Exception as e:
        print(f"  {C.CROSS} Could not start local server: {e}")
    finally:
        if server_proc:
            server_proc.kill()
            try:
                server_proc.wait(timeout=3)
            except Exception:
                pass

    # ── HF Space Tests ──
    if args.live:
        banner("HF SPACE TESTS (deployed)", char="-")
        all_results.extend(test_server_endpoints(HF_URL, "HF Space"))
    else:
        banner("HF SPACE TESTS", char="-")
        skip("HF Space tests", "use --live to enable")

    # ── OpenAI Tests ──
    if args.with_openai:
        banner("OPENAI BASELINE", char="-")
        all_results.extend(test_openai_baseline())
    else:
        banner("OPENAI BASELINE", char="-")
        skip("OpenAI baseline", "use --with-openai to enable")

    # ── Submission Checklist ──
    banner("SUBMISSION CHECKLIST", char="-")
    all_results.extend(test_submission_checklist())

    # ── Final Report ──
    elapsed = time.time() - start_time
    passed = sum(1 for r in all_results if r)
    failed = sum(1 for r in all_results if not r)
    total = len(all_results)

    banner("FINAL REPORT")

    if failed == 0:
        print(f"""
  {C.GREEN}{C.BOLD}
    ___   __    __       ____   ___   ____ ____  _____ ___  
   / _ | / /   / /      / __ \\ / _ | / __// __/ / __/ / _ \\ 
  / __ |/ /__ / /__    / /_/ // __ | _\\ \\ _\\ \\ / _/  / // / 
 /_/ |_/____//____/   / .___//_/ |_|/___//___//___/ /____/  
                     /_/                                     
  {C.RESET}""")
    else:
        print(f"\n  {C.RED}{C.BOLD}SOME TESTS FAILED{C.RESET}\n")

    print(f"  {C.GREEN}Passed: {passed}{C.RESET}")
    print(f"  {C.RED}Failed: {failed}{C.RESET}")
    print(f"  Total:  {total}")
    print(f"  Time:   {elapsed:.1f}s")
    print()
    print(f"  {C.BOLD}Submission URL:{C.RESET}")
    print(f"  {C.CYAN}{HF_URL}{C.RESET}")
    print()

    if failed == 0:
        print(f"  {C.GREEN}{C.BOLD}READY TO SUBMIT{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}FIX {failed} FAILING TEST(S) BEFORE SUBMISSION{C.RESET}")

    print(f"\n{C.CYAN}{'=' * 70}{C.RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
