"""
Custom Gradio web UI for the Voice Agent Prompt Optimizer.

Provides a rich interactive interface showing:
- Call transcripts with colour-coded turns
- Failure points highlighted
- Score breakdown with visual bars
- Iterative prompt optimization workflow
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr


def _transcript_html(transcripts: List[Dict]) -> str:
    """Render call transcripts as readable HTML."""
    if not transcripts:
        return "<p style='color:#888'>No transcript loaded.</p>"

    html = ""
    for t in transcripts:
        sid = t.get("session_id", "")
        html += f"<p style='color:#aaa;font-size:12px;margin:8px 0 4px'>Session: {sid}</p>"
        for turn in t.get("turns", []):
            role = turn.get("role", "")
            text = turn.get("text", "")
            ts = turn.get("timestamp", "")
            if role == "customer":
                bg = "#1a3a5c"
                label = "CUSTOMER"
                align = "left"
            else:
                bg = "#1a3a2a"
                label = "AGENT"
                align = "right"
            html += f"""
<div style='text-align:{align};margin:4px 0'>
  <span style='background:{bg};color:#eee;padding:6px 10px;border-radius:8px;
               display:inline-block;max-width:80%;font-size:13px'>
    <span style='color:#888;font-size:11px'>[{ts}] {label}</span><br>
    {text}
  </span>
</div>"""
    return html


def _score_bar(score: float, label: str) -> str:
    """Render a score as an HTML progress bar."""
    pct = int(score * 100)
    if pct >= 80:
        colour = "#22c55e"
    elif pct >= 50:
        colour = "#f59e0b"
    else:
        colour = "#ef4444"
    return f"""
<div style='margin:4px 0'>
  <span style='font-size:12px;color:#aaa'>{label}</span>
  <div style='background:#333;border-radius:4px;height:16px;margin-top:2px'>
    <div style='background:{colour};width:{pct}%;height:16px;border-radius:4px;
                display:flex;align-items:center;padding-left:6px'>
      <span style='font-size:11px;color:#fff;font-weight:bold'>{pct}%</span>
    </div>
  </div>
</div>"""


def _format_reset_output(data: Dict) -> tuple:
    """Parse reset response into UI components."""
    obs = data.get("observation", {})
    if not obs:
        return "No data", "", "", "", ""

    task_id = obs.get("task_id", "")
    difficulty = obs.get("task_difficulty", "")
    scenario_id = obs.get("scenario_id", "")
    current_prompt = obs.get("current_prompt", "")
    failure_points = obs.get("failure_points", [])
    required = obs.get("required_improvements", [])
    policy = obs.get("policy_context", "")
    feedback = obs.get("feedback_message", "")

    # Task badge
    diff_colour = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}.get(
        difficulty, "#888"
    )
    task_html = f"""
<div style='padding:10px;background:#1e1e2e;border-radius:8px;margin-bottom:8px'>
  <span style='font-size:16px;font-weight:bold;color:#fff'>{task_id}</span>
  <span style='background:{diff_colour};color:#000;padding:2px 8px;border-radius:12px;
               font-size:12px;margin-left:8px'>{difficulty.upper()}</span>
  <span style='color:#888;font-size:12px;margin-left:8px'>#{scenario_id}</span>
</div>
<div style='background:#2a2a3e;padding:8px 12px;border-radius:6px;margin-bottom:8px'>
  <p style='color:#888;font-size:11px;margin:0 0 4px'>CURRENT (BAD) PROMPT</p>
  <p style='color:#fbbf24;font-size:13px;margin:0'>{current_prompt}</p>
</div>"""

    # Failure points
    fp_html = "<div style='padding:8px'>"
    for fp in failure_points:
        fp_html += f"<div style='color:#ef4444;font-size:13px;margin:3px 0'>✗ {fp}</div>"
    if required:
        fp_html += "<p style='color:#888;font-size:11px;margin:8px 0 4px'>REQUIRED IMPROVEMENTS</p>"
        for r in required:
            fp_html += f"<div style='color:#f59e0b;font-size:12px;margin:2px 0'>→ {r.replace('_', ' ')}</div>"
    fp_html += "</div>"

    transcript_html = _transcript_html(obs.get("call_transcripts", []))

    return task_html, fp_html, transcript_html, current_prompt, feedback


def _format_step_output(data: Dict) -> tuple:
    """Parse step response into UI components."""
    obs = data.get("observation", {})
    reward = data.get("reward")
    done = data.get("done", False)
    sb = obs.get("score_breakdown", {})
    feedback = obs.get("feedback_message", "")
    turn = obs.get("turn_number", 0)
    max_turns = obs.get("max_turns", 3)

    if not sb:
        return "", "", feedback or "Submit a prompt first."

    score = sb.get("score", 0)
    imp = sb.get("improvement_score", 0)
    qual = sb.get("quality_score", 0)
    penalty = sb.get("policy_penalty", 0)

    # Score summary
    score_colour = "#22c55e" if score >= 0.8 else "#f59e0b" if score >= 0.5 else "#ef4444"
    summary_html = f"""
<div style='text-align:center;padding:16px;background:#1e1e2e;border-radius:8px;margin-bottom:8px'>
  <div style='font-size:48px;font-weight:bold;color:{score_colour}'>{score:.2f}</div>
  <div style='color:#888;font-size:14px'>Final Score</div>
  <div style='color:#aaa;font-size:13px;margin-top:4px'>
    Reward: <b style='color:#60a5fa'>{reward:.3f}</b> &nbsp;|&nbsp;
    Turn {turn}/{max_turns} &nbsp;|&nbsp;
    {'<span style="color:#22c55e">DONE</span>' if done else '<span style="color:#f59e0b">IN PROGRESS</span>'}
  </div>
</div>"""

    # Breakdown bars
    bars_html = (
        _score_bar(imp, f"Improvement Coverage  ({imp:.0%})")
        + _score_bar(qual, f"Prompt Quality  ({qual:.0%})")
        + _score_bar(max(0, 1 - penalty), f"Policy Compliance  ({(1-penalty):.0%})")
    )

    # Improvement breakdown
    breakdown_html = "<div style='margin-top:8px'>"
    for k, v in sb.get("improvement_breakdown", {}).items():
        icon = "✓" if v else "✗"
        colour = "#22c55e" if v else "#ef4444"
        breakdown_html += f"<div style='color:{colour};font-size:12px;margin:2px 0'>{icon} {k.replace('_', ' ')}</div>"

    violations = sb.get("policy_violations", [])
    if violations:
        breakdown_html += "<p style='color:#ef4444;font-size:12px;margin:8px 0 2px'>Policy Violations:</p>"
        for v in violations:
            breakdown_html += f"<div style='color:#ef4444;font-size:11px;margin:2px 0'>⚠ {v['description']}</div>"
    breakdown_html += "</div>"

    return summary_html, bars_html + breakdown_html, feedback


def build_voice_agent_gradio(
    web_manager: Any,
    action_fields: List[Dict],
    metadata: Any,
    is_chat_env: bool,
    title: str = "Voice Agent Prompt Optimizer",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Custom Gradio UI for the Voice Agent Prompt Optimizer environment."""

    css = """
    .gradio-container { background: #0f0f1a !important; }
    .task-header { font-size: 20px; font-weight: bold; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Voice Agent Prompt Optimizer",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=css,
    ) as demo:

        gr.Markdown(
            """
# Voice Agent Prompt Optimizer
### RL Environment for post-call voice agent prompt optimization
Train an AI to fix bad voice agent prompts by studying failed phone calls.
---
"""
        )

        with gr.Row():
            # ── Left column: call context ──
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load a Failed Call")

                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=["faq_resolution", "complaint_handling", "multi_session_sales"],
                        value="faq_resolution",
                        label="Task",
                    )
                    reset_btn = gr.Button("Reset / Load Call", variant="primary")

                task_info = gr.HTML(label="Task Info", value="<p style='color:#888'>Click Reset to load a call.</p>")
                failure_info = gr.HTML(label="What Went Wrong")

                gr.Markdown("#### Call Transcript")
                transcript_display = gr.HTML(value="<p style='color:#888'>No transcript loaded.</p>")

            # ── Right column: optimization ──
            with gr.Column(scale=1):
                gr.Markdown("### 2. Optimize the Prompt")

                optimized_prompt = gr.Textbox(
                    label="Your Optimized System Prompt",
                    placeholder="Write an improved prompt here that fixes the failure points...\n\nExample:\nYou are a professional voice assistant from Vobiz.\n\n1. Business hours: Mon-Fri 9AM-6PM IST\n2. Always answer directly with specific data\n3. Never say 'check our website'",
                    lines=12,
                    max_lines=20,
                )

                reasoning_input = gr.Textbox(
                    label="Reasoning (optional)",
                    placeholder="Explain what you changed and why...",
                    lines=2,
                )

                step_btn = gr.Button("Submit Prompt & Grade", variant="primary")

                gr.Markdown("#### Score")
                score_display = gr.HTML(value="<p style='color:#888'>Submit a prompt to see score.</p>")
                breakdown_display = gr.HTML()

                feedback_display = gr.Textbox(
                    label="Feedback",
                    interactive=False,
                    lines=3,
                )

        # ── State ──
        state_task = gr.State("faq_resolution")
        prefill_prompt = gr.State("")

        # ── Handlers ──
        async def do_reset(task_id: str):
            try:
                # web_manager.reset_environment() calls env.reset() with no args
                # For task selection, we use the /reset HTTP endpoint directly
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.post(
                            "http://localhost:8000/reset",
                            json={"task_id": task_id},
                        )
                        data = resp.json()
                except Exception:
                    # Fallback to web_manager if HTTP call fails
                    data = await web_manager.reset_environment()
                task_html, fp_html, t_html, cur_prompt, fb = _format_reset_output(data)
                return task_html, fp_html, t_html, cur_prompt, fb, task_id
            except Exception as e:
                err = f"<p style='color:red'>Error: {e}</p>"
                return err, "", "", "", str(e), task_id

        async def do_step(prompt_text: str, reasoning_text: str):
            if not prompt_text.strip():
                return (
                    "<p style='color:#f59e0b'>Enter a prompt first.</p>",
                    "",
                    "Please enter an optimized prompt above.",
                )
            try:
                action = {"optimized_prompt": prompt_text, "reasoning": reasoning_text or ""}
                data = await web_manager.step_environment(action)
                summary_html, bars_html, fb = _format_step_output(data)
                return summary_html, bars_html, fb
            except Exception as e:
                # Fallback to HTTP endpoint
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=15) as client:
                        resp = await client.post(
                            "http://localhost:8000/step",
                            json={"action": {"optimized_prompt": prompt_text, "reasoning": reasoning_text or ""}},
                        )
                        data = resp.json()
                    summary_html, bars_html, fb = _format_step_output(data)
                    return summary_html, bars_html, fb
                except Exception as e2:
                    return f"<p style='color:red'>Error: {e2}</p>", "", str(e2)

        reset_btn.click(
            fn=do_reset,
            inputs=[task_dropdown],
            outputs=[task_info, failure_info, transcript_display, optimized_prompt, feedback_display, state_task],
        )

        step_btn.click(
            fn=do_step,
            inputs=[optimized_prompt, reasoning_input],
            outputs=[score_display, breakdown_display, feedback_display],
        )

        # ── Quick Start ──
        with gr.Accordion("Quick Start & API", open=False):
            gr.Markdown("""
```python
from voice_agent_env import VoiceAgentEnv, VoiceAgentAction

with VoiceAgentEnv(base_url="http://localhost:8000").sync() as env:
    # Load a failed call transcript
    result = env.reset()
    print(result.observation.failure_points)

    # Submit your optimized prompt
    result = env.step(VoiceAgentAction(
        optimized_prompt="You are a professional voice agent from Vobiz. Business hours: Mon-Fri 9AM-6PM. Plans: Starter $29, Pro $79, Enterprise $199. Always answer directly.",
        reasoning="Added product knowledge and direct answer policy"
    ))
    print(f"Score: {result.observation.score_breakdown['score']}")
```
**API Docs:** [/docs](/docs) &nbsp;|&nbsp; **Health:** [/health](/health) &nbsp;|&nbsp; **Tasks:** [/tasks](/tasks)
""")

        with gr.Accordion("README", open=False):
            readme = metadata.readme_content if metadata and metadata.readme_content else "*No README.*"
            gr.Markdown(readme)

    return demo
