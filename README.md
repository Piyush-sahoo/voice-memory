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

**An OpenEnv RL environment that teaches AI to write better prompts for voice agents — by learning from failed phone calls.**

> Built for the Meta x Scaler OpenEnv Hackathon | Team Zukii

## The Problem

We build AI voice agents that make real outbound phone calls using **LiveKit + Deepgram + GPT-4o-mini**. The #1 pain point? The system prompt. When a voice agent doesn't know the answer to "What are your prices?" or fumbles an angry customer, the only fix is a human manually rewriting the prompt after listening to call recordings. This doesn't scale.

## What We Built

An RL environment where an AI agent **learns to optimize voice agent prompts by studying failed call transcripts**. It receives a bad call, figures out what went wrong, and rewrites the prompt to prevent it from happening again.

```
         FAILED CALL                    RL AGENT                    BETTER PROMPT
  ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
  │ Customer: "What are │      │ Analyzes transcript  │      │ "You are a voice    │
  │ your prices?"       │ ───> │ Detects: no product  │ ───> │ agent from Vobiz.   │
  │ Agent: "Check our   │      │ knowledge, deflected │      │ Plans: Starter $29, │
  │ website."           │      │ to website           │      │ Pro $79, Ent $199.  │
  │ Customer: "Bye."    │      │ Rewrites prompt      │      │ Always answer with  │
  └─────────────────────┘      └─────────────────────┘      │ specific numbers."  │
                                                             └─────────────────────┘
```

## How It Works

1. **Reset** — Environment loads a call transcript where the voice agent failed, along with the prompt that was used
2. **Step** — RL agent submits an improved prompt. Environment scores it: Did it fix the knowledge gap? Is it structured? Does it violate any policies?
3. **Iterate** — Agent gets feedback and can refine the prompt for up to 3 turns
4. **Score** — Final grade from 0.0 to 1.0 based on how many issues were actually fixed

## Three Tasks, Increasing Difficulty

| Task | What Happened | What the Agent Must Fix |
|------|--------------|------------------------|
| **FAQ Resolution** (Easy) | Customer asked a simple question (hours, pricing, password reset). Agent didn't know the answer. | Add the missing knowledge to the prompt |
| **Complaint Handling** (Medium) | Angry customer called about billing error / outage. Agent was cold, offered no resolution, tried to transfer immediately. | Add empathy, de-escalation steps, refund authority to the prompt |
| **Multi-Session Sales** (Hard) | 3-call outbound sales sequence. Agent lost the deal — couldn't quote prices, forgot previous conversations, had no counter for competitors. | Add product knowledge, session memory instructions, objection handling |

Each task has **5 realistic scenarios** with real dialogue modeled on our production LiveKit voice agent pipeline. 15 total scenarios.

## LiveKit Integration — Real Calls

This isn't just synthetic data. We built a **LiveKit bridge** that captures transcripts from real phone calls and feeds them directly into the environment:

```
  Real Phone Call (LiveKit + Vobiz SIP)
           │
           ▼
  Deepgram STT ──> GPT-4o-mini ──> OpenAI TTS
           │
           ▼
  TranscriptCapture (livekit_bridge.py)
           │
           ▼
  POST /live-reset ──> Environment auto-detects failures
           │
           ▼
  RL Agent optimizes prompt ──> Graded ──> Updated prompt goes back to agent
```

The auto-detection catches: website deflection, vague superlatives without data, customer frustration, unresolved calls, unnecessary transfers.

## Quick Start

```bash
git clone https://github.com/Piyush-sahoo/voice-memory.git
cd voice-memory
pip install -e .

# Start the server
uvicorn server.app:app --port 8000

# Run the test suite
python test_all.py

# Run baseline (no API key needed)
python baseline.py --deterministic

# Run baseline with OpenAI
OPENAI_API_KEY=sk-... python baseline.py
```

## Live Demo

```bash
# Test mode — uses synthetic transcript, no real call needed
python demo/run_live_call.py --test --optimize

# Real call via LiveKit (needs .env with credentials)
python demo/run_live_call.py --to +919988776655 --optimize
```

## Deployed

**HF Spaces:** [https://huggingface.co/spaces/PiyushS/voice-agent-env](https://huggingface.co/spaces/PiyushS/voice-agent-env)

```bash
# Validate it yourself
pip install openenv-core
openenv validate --url https://piyushs-voice-agent-env.hf.space
# Result: 6/6 criteria passed
```

## Project Structure

```
├── models.py                 # Action (optimized_prompt) + Observation (transcript, scores)
├── client.py                 # WebSocket client for the environment
├── baseline.py               # Baseline inference (deterministic + OpenAI GPT-4o-mini)
├── livekit_bridge.py         # Real call transcript capture from LiveKit
├── test_all.py               # Full test suite (120 tests)
├── demo/
│   └── run_live_call.py      # End-to-end: make call → capture → optimize → grade
└── server/
    ├── environment.py        # Core RL logic: reset() / step() / state
    ├── scenarios.py          # 15 call transcripts across 3 difficulty levels
    ├── graders.py            # Deterministic scoring (0.0–1.0)
    ├── rewards.py            # Per-turn partial rewards
    ├── policy_rules.py       # Policy violation detection
    ├── app.py                # FastAPI server with all endpoints
    └── Dockerfile            # Containerized for HF Spaces
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Voice Calls | LiveKit + Vobiz SIP Trunking |
| Speech-to-Text | Deepgram Nova-3 |
| LLM | OpenAI GPT-4o-mini |
| Text-to-Speech | OpenAI TTS / Cartesia Sonic-2 |
| RL Framework | OpenEnv by Meta |
| Server | FastAPI + Uvicorn |
| Deployment | Docker on Hugging Face Spaces |

## Team

**Team Zukii**
- Piyush Sahoo — [GitHub](https://github.com/Piyush-sahoo)
- Kush Anchalia
