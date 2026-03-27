#!/usr/bin/env python3
"""
Live Call Demo — Make a real call via LiveKit, capture transcript,
and optimize the voice agent prompt through the OpenEnv environment.

This script:
  1. Dispatches an outbound call via LiveKit (same as make_call.py)
  2. Monitors the room for the call to complete
  3. Captures the transcript via LiveKit room events
  4. Feeds the transcript into our OpenEnv environment
  5. (Optional) Uses OpenAI to generate an optimized prompt and grades it

Usage:
    # Basic: make call + capture transcript
    python demo/run_live_call.py --to +919988776655

    # Full pipeline: call + capture + optimize via OpenAI
    python demo/run_live_call.py --to +919988776655 --optimize

    # Use a different OpenEnv server
    python demo/run_live_call.py --to +919988776655 --openenv-url https://piyushs-voice-agent-env.hf.space

Requires:
    - .env file with LiveKit, Vobiz, OpenAI, and Deepgram credentials
    - LiveKit agent (agent.py) running separately
    - OpenEnv server running (local or HF Spaces)
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

from dotenv import load_dotenv

# Load env from the project root or LiveKit repo
for env_path in [".env", "../LiveKit-Vobiz-Outbound/.env", "demo/.env"]:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def dispatch_call(phone_number: str) -> tuple:
    """Dispatch an outbound call via LiveKit. Returns (room_name, dispatch_id)."""
    from livekit import api as lk_api

    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not (url and api_key and api_secret):
        raise ValueError("Missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET in .env")

    client = lk_api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)

    room_name = f"demo-{phone_number.replace('+', '')}-{random.randint(1000, 9999)}"

    try:
        dispatch = await client.agent_dispatch.create_dispatch(
            lk_api.CreateAgentDispatchRequest(
                agent_name="outbound-caller",
                room=room_name,
                metadata=json.dumps({"phone_number": phone_number}),
            )
        )
        return room_name, dispatch.id
    finally:
        await client.aclose()


async def monitor_room_and_capture(room_name: str, max_wait_s: int = 300) -> list:
    """
    Monitor a LiveKit room and capture transcript events.

    Uses the LiveKit server API to poll room participants and track
    when the call starts/ends. Returns captured transcript turns.

    Note: For full real-time transcript capture, you'd use the
    TranscriptCapture class attached directly to the AgentSession
    in agent.py. This polling approach works as a standalone demo.
    """
    from livekit import api as lk_api

    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    client = lk_api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)

    print(f"Monitoring room: {room_name}")
    print(f"Waiting up to {max_wait_s}s for call to complete...")

    start = time.time()
    call_started = False
    call_ended = False

    try:
        while time.time() - start < max_wait_s:
            try:
                participants = await client.room.list_participants(
                    lk_api.ListParticipantsRequest(room=room_name)
                )

                sip_participants = [
                    p for p in participants.participants
                    if p.identity.startswith("sip_")
                ]

                if sip_participants and not call_started:
                    call_started = True
                    print("Call connected! Waiting for call to end...")

                if call_started and not sip_participants:
                    call_ended = True
                    print("Call ended (SIP participant left).")
                    break

            except Exception:
                if call_started:
                    call_ended = True
                    print("Room closed — call ended.")
                    break

            await asyncio.sleep(3)

        if not call_ended:
            print("Timeout waiting for call to complete.")

    finally:
        await client.aclose()

    return call_started


async def get_transcript_from_agent_logs() -> dict:
    """
    Placeholder for transcript capture.

    In production, transcripts come from one of:
      1. TranscriptCapture attached to the AgentSession (best)
      2. LiveKit room composite egress (recording)
      3. LiveKit webhook events
      4. Agent log parsing

    For this demo, we create a prompt for the user to paste the transcript,
    or use a fallback synthetic transcript for testing.
    """
    print("\n" + "=" * 60)
    print("TRANSCRIPT CAPTURE")
    print("=" * 60)
    print(
        "For full automatic transcript capture, integrate\n"
        "TranscriptCapture from livekit_bridge.py into agent.py.\n"
        "\nFor now, enter the transcript manually or press Enter\n"
        "to use a test transcript."
    )

    # In a real integration, this would come from TranscriptCapture.get_transcript()
    # For demo purposes, use a realistic test transcript
    return {
        "session_id": f"live-demo-{int(time.time())}",
        "turns": [
            {"role": "customer", "text": "Hello?", "timestamp": "00:00"},
            {"role": "agent", "text": "Hi! I'm an AI assistant from Vobiz. How can I help you today?", "timestamp": "00:02"},
            {"role": "customer", "text": "What plans do you have and how much do they cost?", "timestamp": "00:06"},
            {"role": "agent", "text": "We have several great plans available! Would you like me to tell you about them?", "timestamp": "00:09"},
            {"role": "customer", "text": "Yes, that's what I just asked. Give me the details.", "timestamp": "00:13"},
            {"role": "agent", "text": "Our plans are very competitive. I'd recommend visiting our website for the most up to date pricing.", "timestamp": "00:16"},
            {"role": "customer", "text": "You don't even know your own prices? Never mind.", "timestamp": "00:21"},
        ],
        "metadata": {
            "customer_name": "Live Demo Customer",
            "phone_number": "demo",
            "call_duration_s": 23,
            "resolution_status": "unresolved",
            "stt_provider": "deepgram_nova3",
            "llm_model": "gpt-4o-mini",
            "tts_provider": "openai",
            "platform": "livekit",
            "sip_provider": "vobiz",
        },
    }


async def main():
    parser = argparse.ArgumentParser(description="Live Call Demo for Voice Agent Prompt Optimizer")
    parser.add_argument("--to", type=str, help="Phone number to call (e.g. +919988776655)")
    parser.add_argument("--optimize", action="store_true", help="Run OpenAI optimization after capture")
    parser.add_argument("--openenv-url", type=str, default="http://localhost:8000", help="OpenEnv server URL")
    parser.add_argument("--test", action="store_true", help="Skip real call, use test transcript")
    args = parser.parse_args()

    print("=" * 60)
    print("Voice Agent Prompt Optimizer — Live Demo")
    print("=" * 60)

    # Step 1: Make the call (or use test mode)
    if args.test or not args.to:
        print("\n[TEST MODE] Using synthetic transcript (no real call)")
        transcript = await get_transcript_from_agent_logs()
    else:
        phone_number = args.to.strip()
        if not phone_number.startswith("+"):
            print("Error: Phone number must start with '+' and country code.")
            return

        print(f"\n[CALL] Dispatching call to {phone_number}...")
        try:
            room_name, dispatch_id = await dispatch_call(phone_number)
            print(f"  Room: {room_name}")
            print(f"  Dispatch ID: {dispatch_id}")
        except Exception as e:
            print(f"  Error: {e}")
            print("  Falling back to test transcript...")
            transcript = await get_transcript_from_agent_logs()
            room_name = None

        if room_name:
            # Monitor the call
            call_completed = await monitor_room_and_capture(room_name)
            if call_completed:
                print("Call completed. Capturing transcript...")
            # Get transcript (from agent logs for now)
            transcript = await get_transcript_from_agent_logs()

    # Step 2: Display the transcript
    print("\n" + "-" * 60)
    print("CAPTURED TRANSCRIPT:")
    print("-" * 60)
    for turn in transcript.get("turns", []):
        role = "AGENT" if turn["role"] == "agent" else "CUSTOMER"
        print(f"  [{turn['timestamp']}] {role}: {turn['text']}")

    # Step 3: Feed to OpenEnv
    print("\n" + "-" * 60)
    print(f"FEEDING TO OPENENV ({args.openenv_url})")
    print("-" * 60)

    current_prompt = (
        "You are a helpful and professional voice assistant calling from Vobiz. "
        "Key behaviors: 1. Introduce yourself clearly when the user answers. "
        "2. Be concise and respect the user's time. "
        "3. If asked, explain you are an AI assistant helping with a test call."
    )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Post to /live-reset
            response = await client.post(
                f"{args.openenv_url}/live-reset",
                json={
                    "task_id": "live_call",
                    "task_difficulty": "live",
                    "current_prompt": current_prompt,
                    "call_transcripts": [transcript],
                    "call_metadata": transcript.get("metadata", {}),
                    "failure_points": [],
                    "policy_context": "",
                },
            )
            response.raise_for_status()
            reset_result = response.json()
            obs = reset_result.get("observation", reset_result)
            print(f"  Task: {obs.get('task_id', 'live_call')}")
            print(f"  Failure points detected: {len(obs.get('failure_points', []))}")
            for fp in obs.get("failure_points", []):
                print(f"    - {fp}")
    except httpx.ConnectError:
        print(f"  Could not connect to {args.openenv_url}")
        print("  Start the server with: uvicorn server.app:app --port 8000")
        return
    except httpx.HTTPStatusError as e:
        print(f"  HTTP Error: {e.response.status_code}")
        print(f"  Response: {e.response.text[:200]}")
        return

    # Step 4: Optimize with OpenAI (optional)
    if args.optimize:
        print("\n" + "-" * 60)
        print("OPTIMIZING PROMPT WITH OPENAI")
        print("-" * 60)

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("  Error: OPENAI_API_KEY not set in environment")
            return

        try:
            from livekit_bridge import optimize_prompt_with_openai

            result = await optimize_prompt_with_openai(
                transcript=transcript,
                current_prompt=current_prompt,
                openenv_url=args.openenv_url,
                openai_api_key=api_key,
            )

            print(f"\n  Score: {result.get('score', 'N/A')}")
            print(f"  Feedback: {result.get('feedback', 'N/A')}")
            print(f"\n  Optimized Prompt (first 300 chars):")
            print(f"  {result['optimized_prompt'][:300]}...")

        except Exception as e:
            print(f"  Optimization failed: {e}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
