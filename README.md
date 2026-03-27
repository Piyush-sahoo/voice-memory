---
title: Voice Agent Prompt Optimizer
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Voice Agent Prompt Optimizer

An RL environment for **post-call analysis and system prompt optimization** of AI voice agents. Built on the OpenEnv framework.

## Motivation

Every voice AI company (using LiveKit, Twilio, Vonage, etc.) manually reviews call transcripts to tune their agent prompts. This is slow, subjective, and doesn't scale. This environment automates that feedback loop: an RL agent analyzes call transcripts, identifies failure points, and optimizes the voice agent's system prompt to improve future call handling.

**Pipeline modeled:** LiveKit + Deepgram STT (Nova-3) + GPT-4o-mini + OpenAI TTS — a real production voice AI stack.

## How It Works

```
Episode Flow:

1. reset(task_id) --> Agent receives:
   - Call transcript(s) from a voice AI system
   - The current (flawed) system prompt
   - Failure analysis + required improvements
   - Company policy constraints

2. step(optimized_prompt) --> Agent submits improved prompt
   - Environment grades it against ground-truth improvements
   - Returns score breakdown + feedback
   - Agent can iterate (up to 3 turns per episode)

3. done=True after max turns
```

## Tasks (Easy -> Hard)

| Task | Difficulty | Description | Scenarios |
|------|-----------|-------------|-----------|
| `faq_resolution` | Easy | Single FAQ call — agent couldn't answer a direct question (hours, pricing, etc.). Fix the knowledge gap in the prompt. | 5 |
| `complaint_handling` | Medium | Customer complaint call — agent failed to empathize, de-escalate, or resolve. Add proper complaint handling procedures. | 5 |
| `multi_session_sales` | Hard | Multi-session outbound sales sequence (e.g., credit card sales over 3 calls). Agent lost the deal due to broken promises, no product knowledge, or failure to maintain context across sessions. | 5 |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `optimized_prompt` | `str` | The improved system prompt for the voice agent |
| `reasoning` | `str` | Explanation of what was changed and why |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task: `faq_resolution`, `complaint_handling`, `multi_session_sales` |
| `task_difficulty` | `str` | `easy`, `medium`, `hard` |
| `current_prompt` | `str` | The flawed system prompt used during the call |
| `call_transcripts` | `List[Dict]` | Turn-by-turn transcript(s) with role, text, timestamp |
| `call_metadata` | `Dict` | Customer name, intent, sentiment, resolution status, STT/LLM/TTS info |
| `failure_points` | `List[str]` | What went wrong in the call |
| `required_improvements` | `List[str]` | Specific improvements the optimized prompt must address |
| `policy_context` | `str` | Company policy rules the agent must follow |
| `score_breakdown` | `Dict` | Detailed scoring after each step |
| `feedback_message` | `str` | Human-readable feedback |
| `turn_number` / `max_turns` | `int` | Current turn and maximum turns (3) |

## Reward Function

Per-turn partial rewards (not sparse):

- **Improvement delta**: reward/penalize based on score change from previous turn
- **New improvements bonus**: +0.1 per newly addressed requirement
- **Policy violation penalty**: -0.5 * violation severity
- **Terminal bonus**: final score * 0.5 on last turn

## Grading Logic

Deterministic grading (0.0 - 1.0):

| Component | Weight | Description |
|-----------|--------|-------------|
| Improvement coverage | 60% | Fraction of required_improvements addressed in the prompt |
| Prompt quality | 20% | Structural checks: length, numbered rules, specificity, role clarity |
| Policy compliance | 20% | Penalty for policy violations (e.g., badmouthing competitors, pressure tactics) |

## Setup & Usage

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd voice-agent-openenv
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or with reload for development
uvicorn server.app:app --reload
```

### Using the Client

```python
from voice_agent_env import VoiceAgentEnv, VoiceAgentAction

with VoiceAgentEnv(base_url="http://localhost:8000").sync() as env:
    # Reset with specific task
    result = env.reset()  # random task, or pass task_id="faq_resolution"

    # View the call transcript and failure points
    obs = result.observation
    print(f"Task: {obs.task_id} ({obs.task_difficulty})")
    print(f"Failures: {obs.failure_points}")

    # Submit an optimized prompt
    result = env.step(VoiceAgentAction(
        optimized_prompt="You are a professional voice assistant from Vobiz...",
        reasoning="Added business hours and direct answer policy",
    ))
    print(f"Score: {result.observation.score_breakdown['score']}")
    print(f"Feedback: {result.observation.feedback_message}")
```

### Docker

```bash
# Build
openenv build . -t voice-agent-env

# Run
docker run -d -p 8000:8000 voice-agent-env

# Test
curl http://localhost:8000/health
```

### Deploy to HF Spaces

```bash
openenv push --repo-id your-username/voice-agent-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit optimized prompt |
| `/state` | GET | Episode metadata |
| `/tasks` | GET | List tasks + action schema |
| `/grader` | POST | Grade a prompt against a scenario |
| `/baseline` | POST | Run baseline inference on all tasks |
| `/ws` | WebSocket | Persistent session |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI documentation |

## Baseline Scores

### Deterministic Baseline (hand-crafted prompts from ground truth)

| Task | Difficulty | Average Score |
|------|-----------|--------------|
| `faq_resolution` | Easy | 0.98 |
| `complaint_handling` | Medium | 0.98 |
| `multi_session_sales` | Hard | 0.99 |

### Running Baseline

```bash
# Deterministic (no API key needed)
python baseline.py --deterministic

# With OpenAI
OPENAI_API_KEY=sk-... python baseline.py
```

## Project Structure

```
voice-agent-openenv/
├── __init__.py              # Package exports
├── models.py                # VoiceAgentAction, VoiceAgentObservation (Pydantic)
├── client.py                # VoiceAgentEnv (EnvClient subclass)
├── baseline.py              # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package metadata
├── uv.lock                  # Locked dependencies
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI + /tasks, /grader, /baseline endpoints
    ├── voice_agent_env_environment.py  # Core: reset(), step(), state
    ├── scenarios.py          # 15 synthetic call transcripts (5 per task)
    ├── graders.py            # Deterministic scoring (0.0-1.0)
    ├── rewards.py            # Per-turn partial reward function
    ├── policy_rules.py       # Company policy constraints
    ├── Dockerfile            # Container for HF Spaces
    └── requirements.txt      # Server dependencies
```

## Validation

```bash
# Validate structure
openenv validate .

# Validate running server
uvicorn server.app:app --port 8000 &
openenv validate --url http://localhost:8000
# Result: 6/6 criteria passed
```
