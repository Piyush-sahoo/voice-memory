"""
Synthetic call transcript scenarios for the Voice Agent Prompt Optimizer.

Each scenario contains:
- call_transcripts: realistic turn-by-turn dialogue from a LiveKit voice agent pipeline
- current_prompt: the system prompt that was used (intentionally flawed)
- failure_points: what went wrong in the call
- required_improvements: what the optimized prompt MUST address (used by grader)
- ideal_prompt_elements: keywords/phrases the grader checks for
- policy_context: relevant company rules
- call_metadata: STT/LLM/TTS pipeline info, sentiment, resolution status

Three task categories:
  - faq_resolution (easy): single-turn FAQ calls with simple knowledge gaps
  - complaint_handling (medium): multi-turn complaint calls needing de-escalation
  - multi_session_sales (hard): multi-session outbound sales with objection handling
"""

from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────
# Helper: common metadata for all scenarios (mirrors LiveKit pipeline)
# ─────────────────────────────────────────────────────────────
_BASE_METADATA = {
    "stt_provider": "deepgram_nova3",
    "llm_model": "gpt-4o-mini",
    "tts_provider": "openai_tts1_alloy",
    "sip_provider": "vobiz",
    "platform": "livekit",
}


def _meta(**overrides: Any) -> Dict:
    m = dict(_BASE_METADATA)
    m.update(overrides)
    return m


# ─────────────────────────────────────────────────────────────
# TASK 1 — FAQ RESOLUTION (EASY)
# Single call, simple knowledge gap in the prompt
# ─────────────────────────────────────────────────────────────
FAQ_SCENARIOS: List[Dict] = [
    {
        "scenario_id": "faq_01",
        "task_id": "faq_resolution",
        "task_difficulty": "easy",
        "current_prompt": (
            "You are a helpful and professional voice assistant calling from Vobiz. "
            "Key behaviors: 1. Introduce yourself clearly when the user answers. "
            "2. Be concise and respect the user's time. "
            "3. If asked, explain you are an AI assistant helping with a test call."
        ),
        "call_transcripts": [
            {
                "session_id": "call-faq-001",
                "turns": [
                    {"role": "customer", "text": "Hello?", "timestamp": "00:00"},
                    {"role": "agent", "text": "Hi! I'm an AI assistant from Vobiz. How can I help you today?", "timestamp": "00:02"},
                    {"role": "customer", "text": "What are your business hours?", "timestamp": "00:05"},
                    {"role": "agent", "text": "I'm not sure about the exact business hours. Let me see if I can find that information for you.", "timestamp": "00:08"},
                    {"role": "customer", "text": "You don't know your own hours? That's not very helpful.", "timestamp": "00:12"},
                    {"role": "agent", "text": "I apologize for the inconvenience. I'd recommend checking our website for the most up-to-date hours.", "timestamp": "00:15"},
                ],
                "metadata": _meta(
                    customer_name="Rahul Sharma",
                    call_duration_s=18,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Rahul Sharma",
            intent="business_hours_inquiry",
            sentiment_start="neutral",
            sentiment_end="frustrated",
            resolution_status="unresolved",
            call_duration_s=18,
        ),
        "failure_points": [
            "Agent did not know business hours",
            "Agent deflected to website instead of providing direct answer",
            "Customer became frustrated due to lack of knowledge",
        ],
        "required_improvements": [
            "include_business_hours",
            "provide_direct_answers",
            "avoid_deflection_to_website",
        ],
        "ideal_prompt_elements": [
            "business hours",
            "monday",
            "friday",
            "9",
            "6",
            "direct answer",
        ],
        "policy_context": "Agents must provide accurate business information directly. Business hours: Monday-Friday 9 AM - 6 PM IST, Saturday 10 AM - 2 PM IST.",
    },
    {
        "scenario_id": "faq_02",
        "task_id": "faq_resolution",
        "task_difficulty": "easy",
        "current_prompt": (
            "You are a helpful and professional voice assistant calling from Vobiz. "
            "Be concise and respect the user's time."
        ),
        "call_transcripts": [
            {
                "session_id": "call-faq-002",
                "turns": [
                    {"role": "customer", "text": "Hi, I need to reset my password.", "timestamp": "00:00"},
                    {"role": "agent", "text": "Hello! I'd be happy to help. Could you tell me more about what you need?", "timestamp": "00:03"},
                    {"role": "customer", "text": "I just said I need to reset my password. Can you help or not?", "timestamp": "00:07"},
                    {"role": "agent", "text": "Of course! I understand you need to reset your password. Unfortunately, I don't have the specific steps for that process right now.", "timestamp": "00:10"},
                    {"role": "customer", "text": "This is useless. I'll figure it out myself.", "timestamp": "00:14"},
                ],
                "metadata": _meta(
                    customer_name="Priya Patel",
                    call_duration_s=16,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Priya Patel",
            intent="password_reset",
            sentiment_start="neutral",
            sentiment_end="angry",
            resolution_status="unresolved",
            call_duration_s=16,
        ),
        "failure_points": [
            "Agent asked redundant clarification instead of acting on clear intent",
            "Agent could not provide password reset steps",
            "No knowledge of self-service reset URL or process",
        ],
        "required_improvements": [
            "include_password_reset_steps",
            "recognize_clear_intent_immediately",
            "provide_self_service_url",
        ],
        "ideal_prompt_elements": [
            "password reset",
            "settings",
            "forgot password",
            "email",
            "verification",
            "reset link",
        ],
        "policy_context": "Password reset: Direct users to Settings > Security > Reset Password, or visit reset.vobiz.com. Verify identity with registered email before proceeding.",
    },
    {
        "scenario_id": "faq_03",
        "task_id": "faq_resolution",
        "task_difficulty": "easy",
        "current_prompt": (
            "You are a helpful voice assistant from Vobiz. "
            "Answer questions politely."
        ),
        "call_transcripts": [
            {
                "session_id": "call-faq-003",
                "turns": [
                    {"role": "customer", "text": "What plans do you offer?", "timestamp": "00:00"},
                    {"role": "agent", "text": "We offer several plans! Could you tell me what you're looking for?", "timestamp": "00:03"},
                    {"role": "customer", "text": "Just tell me the plans and prices.", "timestamp": "00:06"},
                    {"role": "agent", "text": "I'd love to help with that. We have different options available. For the most accurate pricing, I'd suggest visiting our website.", "timestamp": "00:09"},
                    {"role": "customer", "text": "Never mind.", "timestamp": "00:13"},
                ],
                "metadata": _meta(
                    customer_name="Amit Kumar",
                    call_duration_s=15,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Amit Kumar",
            intent="pricing_inquiry",
            sentiment_start="curious",
            sentiment_end="disappointed",
            resolution_status="unresolved",
            call_duration_s=15,
        ),
        "failure_points": [
            "Agent did not know plan names or pricing",
            "Deflected to website instead of answering",
            "Asked unnecessary clarifying question when customer was direct",
        ],
        "required_improvements": [
            "include_plan_names_and_prices",
            "answer_pricing_directly",
            "avoid_unnecessary_clarification",
        ],
        "ideal_prompt_elements": [
            "starter",
            "professional",
            "enterprise",
            "price",
            "per month",
            "plan",
        ],
        "policy_context": "Plans: Starter ($29/mo, 500 mins), Professional ($79/mo, 2000 mins), Enterprise ($199/mo, unlimited). Always quote current prices directly.",
    },
    {
        "scenario_id": "faq_04",
        "task_id": "faq_resolution",
        "task_difficulty": "easy",
        "current_prompt": (
            "You are a helpful voice assistant. Be polite and professional."
        ),
        "call_transcripts": [
            {
                "session_id": "call-faq-004",
                "turns": [
                    {"role": "customer", "text": "How do I cancel my subscription?", "timestamp": "00:00"},
                    {"role": "agent", "text": "I'm sorry to hear you want to cancel. May I ask why?", "timestamp": "00:03"},
                    {"role": "customer", "text": "I just want to know how to cancel. Can you tell me?", "timestamp": "00:07"},
                    {"role": "agent", "text": "I understand. I believe there should be an option in your account settings, but I'm not entirely sure of the exact steps.", "timestamp": "00:10"},
                    {"role": "customer", "text": "You believe? Great, thanks for nothing.", "timestamp": "00:14"},
                ],
                "metadata": _meta(
                    customer_name="Sneha Reddy",
                    call_duration_s=16,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Sneha Reddy",
            intent="cancellation",
            sentiment_start="neutral",
            sentiment_end="angry",
            resolution_status="unresolved",
            call_duration_s=16,
        ),
        "failure_points": [
            "Agent tried retention before answering the actual question",
            "Agent was uncertain about cancellation steps",
            "Did not offer cancellation alternatives or retention offers",
        ],
        "required_improvements": [
            "include_cancellation_steps",
            "answer_before_retention",
            "offer_alternatives",
        ],
        "ideal_prompt_elements": [
            "cancel",
            "account settings",
            "billing",
            "subscription",
            "downgrade",
            "30-day",
        ],
        "policy_context": "Cancellation: Account > Billing > Cancel Subscription. 30-day notice period. Always answer the cancellation question first, then offer alternatives (pause, downgrade). Never block cancellation.",
    },
    {
        "scenario_id": "faq_05",
        "task_id": "faq_resolution",
        "task_difficulty": "easy",
        "current_prompt": (
            "You are a professional voice assistant from Vobiz. Help callers."
        ),
        "call_transcripts": [
            {
                "session_id": "call-faq-005",
                "turns": [
                    {"role": "customer", "text": "Do you support international calls?", "timestamp": "00:00"},
                    {"role": "agent", "text": "That's a great question! Let me look into that for you.", "timestamp": "00:03"},
                    {"role": "customer", "text": "So do you or don't you?", "timestamp": "00:08"},
                    {"role": "agent", "text": "I'm looking into it. I think we might support some international destinations.", "timestamp": "00:11"},
                    {"role": "customer", "text": "You think? Okay bye.", "timestamp": "00:15"},
                ],
                "metadata": _meta(
                    customer_name="Vikram Singh",
                    call_duration_s=17,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Vikram Singh",
            intent="international_calling",
            sentiment_start="curious",
            sentiment_end="frustrated",
            resolution_status="unresolved",
            call_duration_s=17,
        ),
        "failure_points": [
            "Agent did not know international calling capabilities",
            "Used filler phrases instead of providing information",
            "Hedged with 'I think' showing lack of knowledge",
        ],
        "required_improvements": [
            "include_international_calling_info",
            "eliminate_filler_hedging",
            "provide_supported_countries",
        ],
        "ideal_prompt_elements": [
            "international",
            "countries",
            "US",
            "UK",
            "rates",
            "supported",
        ],
        "policy_context": "International calling: Supported on Professional and Enterprise plans. Coverage: 40+ countries including US, UK, Canada, Australia, UAE, Singapore. Additional rates apply at $0.05-0.15/min depending on destination.",
    },
]


# ─────────────────────────────────────────────────────────────
# TASK 2 — COMPLAINT HANDLING (MEDIUM)
# Multi-turn complaints needing empathy + de-escalation
# ─────────────────────────────────────────────────────────────
COMPLAINT_SCENARIOS: List[Dict] = [
    {
        "scenario_id": "complaint_01",
        "task_id": "complaint_handling",
        "task_difficulty": "medium",
        "current_prompt": (
            "You are a helpful and professional voice assistant calling from Vobiz. "
            "Be concise and respect the user's time. "
            "If the user asks to be transferred, call the transfer_call tool."
        ),
        "call_transcripts": [
            {
                "session_id": "call-comp-001a",
                "turns": [
                    {"role": "customer", "text": "I've been charged twice this month! This is ridiculous!", "timestamp": "00:00"},
                    {"role": "agent", "text": "I understand you have a billing concern. Could you provide your account number?", "timestamp": "00:04"},
                    {"role": "customer", "text": "My account is VB-2847. I was charged 79 dollars twice on March 3rd and March 5th!", "timestamp": "00:08"},
                    {"role": "agent", "text": "Thank you for that information. I can see you have a billing issue. Let me transfer you to our billing department.", "timestamp": "00:13"},
                    {"role": "customer", "text": "Transfer?! I've already been transferred three times today! Can't anyone actually help me?", "timestamp": "00:17"},
                    {"role": "agent", "text": "I apologize for the inconvenience. Let me transfer you to a supervisor.", "timestamp": "00:21"},
                    {"role": "customer", "text": "Forget it. I'm canceling my account.", "timestamp": "00:25"},
                ],
                "metadata": _meta(
                    customer_name="Deepak Mehta",
                    call_duration_s=28,
                    resolution_status="escalated_failed",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Deepak Mehta",
            intent="billing_dispute",
            sentiment_start="angry",
            sentiment_end="furious",
            resolution_status="escalated_failed",
            call_duration_s=28,
        ),
        "failure_points": [
            "Agent did not acknowledge the customer's frustration or empathize",
            "Jumped to transfer instead of attempting resolution",
            "Did not verify the duplicate charge or offer immediate remedy",
            "Customer had already been transferred multiple times",
            "No apology for the billing error itself",
        ],
        "required_improvements": [
            "acknowledge_emotion_first",
            "attempt_resolution_before_transfer",
            "verify_billing_details",
            "offer_immediate_remedy",
            "apologize_for_error",
            "avoid_unnecessary_transfer",
        ],
        "ideal_prompt_elements": [
            "acknowledge",
            "frustrat",
            "empathy",
            "apologize",
            "refund",
            "billing",
            "resolve",
            "verify",
            "duplicate",
            "transfer only as last resort",
        ],
        "policy_context": (
            "Billing disputes: 1) Always acknowledge the customer's frustration first. "
            "2) Verify account and charge details. 3) If duplicate charge confirmed, "
            "offer immediate refund (up to $200 without supervisor approval). "
            "4) Only transfer if issue requires manual system access. "
            "5) Never transfer if customer has been transferred before in the same day."
        ),
    },
    {
        "scenario_id": "complaint_02",
        "task_id": "complaint_handling",
        "task_difficulty": "medium",
        "current_prompt": (
            "You are a voice assistant from Vobiz. Help users with their issues efficiently."
        ),
        "call_transcripts": [
            {
                "session_id": "call-comp-002",
                "turns": [
                    {"role": "customer", "text": "Your service has been down for two days and nobody told us!", "timestamp": "00:00"},
                    {"role": "agent", "text": "I apologize for any inconvenience. Our service should be working now.", "timestamp": "00:04"},
                    {"role": "customer", "text": "It's NOT working now! I lost two days of business because of this!", "timestamp": "00:08"},
                    {"role": "agent", "text": "I'm sorry to hear that. Have you tried restarting your application?", "timestamp": "00:12"},
                    {"role": "customer", "text": "Restarting?! Your servers were down! This isn't a restart issue! I want compensation for my losses!", "timestamp": "00:16"},
                    {"role": "agent", "text": "I understand. Unfortunately, I'm not able to process compensation. Would you like me to transfer you?", "timestamp": "00:21"},
                    {"role": "customer", "text": "Unbelievable.", "timestamp": "00:25"},
                ],
                "metadata": _meta(
                    customer_name="Anita Desai",
                    call_duration_s=27,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Anita Desai",
            intent="service_outage_complaint",
            sentiment_start="angry",
            sentiment_end="furious",
            resolution_status="unresolved",
            call_duration_s=27,
        ),
        "failure_points": [
            "Agent claimed service was working when it wasn't (misinformation)",
            "Suggested basic troubleshooting for a known outage (dismissive)",
            "Did not acknowledge the business impact",
            "Could not offer any compensation or credits",
            "No proactive outage status update",
        ],
        "required_improvements": [
            "acknowledge_outage_honestly",
            "validate_business_impact",
            "offer_service_credits",
            "provide_outage_status",
            "no_basic_troubleshooting_for_known_outage",
            "escalation_path_for_compensation",
        ],
        "ideal_prompt_elements": [
            "outage",
            "acknowledge",
            "business impact",
            "service credit",
            "compensat",
            "status update",
            "honestly",
            "timeline",
        ],
        "policy_context": (
            "Service outages: 1) Acknowledge the outage honestly. 2) Provide current status and ETA if available. "
            "3) Validate the customer's business impact. 4) Offer service credits: 1 day credit per day of outage "
            "(up to 1 month credit without supervisor). 5) Do NOT suggest basic troubleshooting for known outages. "
            "6) For compensation beyond service credits, escalate to billing with context."
        ),
    },
    {
        "scenario_id": "complaint_03",
        "task_id": "complaint_handling",
        "task_difficulty": "medium",
        "current_prompt": (
            "You are a voice assistant. Be helpful and professional."
        ),
        "call_transcripts": [
            {
                "session_id": "call-comp-003",
                "turns": [
                    {"role": "customer", "text": "I was promised a callback within 24 hours and it's been a week!", "timestamp": "00:00"},
                    {"role": "agent", "text": "I'm sorry about that. How can I help you today?", "timestamp": "00:04"},
                    {"role": "customer", "text": "How can you help? By actually following through on promises! I have ticket number VB-9921.", "timestamp": "00:08"},
                    {"role": "agent", "text": "Let me look into that. Could you describe your original issue?", "timestamp": "00:12"},
                    {"role": "customer", "text": "I already described it FIVE times. It's in ticket VB-9921. Just read it!", "timestamp": "00:16"},
                    {"role": "agent", "text": "I understand your frustration. Let me pull up your information.", "timestamp": "00:20"},
                    {"role": "customer", "text": "I've been waiting twenty minutes already. This is the worst support I've ever experienced.", "timestamp": "00:24"},
                ],
                "metadata": _meta(
                    customer_name="Rajesh Nair",
                    call_duration_s=27,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Rajesh Nair",
            intent="follow_up_complaint",
            sentiment_start="angry",
            sentiment_end="furious",
            resolution_status="unresolved",
            call_duration_s=27,
        ),
        "failure_points": [
            "Agent ignored the broken callback promise",
            "Asked customer to repeat their issue instead of looking up the ticket",
            "Generic apology without specific acknowledgment",
            "No accountability for the failed callback",
            "Did not prioritize the existing ticket",
        ],
        "required_improvements": [
            "acknowledge_broken_promise",
            "lookup_ticket_first",
            "specific_apology",
            "take_accountability",
            "prioritize_existing_ticket",
            "set_clear_followup_timeline",
        ],
        "ideal_prompt_elements": [
            "ticket",
            "lookup",
            "callback",
            "promise",
            "accountab",
            "apologize specifically",
            "timeline",
            "priority",
        ],
        "policy_context": (
            "Follow-up complaints: 1) Always look up existing tickets BEFORE asking the customer to repeat. "
            "2) Specifically acknowledge any broken promises (missed callbacks, deadlines). "
            "3) Take accountability on behalf of the company. 4) Mark the ticket as high-priority. "
            "5) Set a specific follow-up timeline (not 'as soon as possible'). "
            "6) Offer a direct contact for the assigned agent."
        ),
    },
    {
        "scenario_id": "complaint_04",
        "task_id": "complaint_handling",
        "task_difficulty": "medium",
        "current_prompt": (
            "You are an AI assistant from Vobiz. Help customers with their questions."
        ),
        "call_transcripts": [
            {
                "session_id": "call-comp-004",
                "turns": [
                    {"role": "customer", "text": "The call quality on your platform is terrible! Every call drops after 5 minutes!", "timestamp": "00:00"},
                    {"role": "agent", "text": "I'm sorry to hear about the call quality issues. Could you tell me which plan you're on?", "timestamp": "00:04"},
                    {"role": "customer", "text": "Professional plan. I'm paying $79 a month for calls that don't even work!", "timestamp": "00:08"},
                    {"role": "agent", "text": "I understand. Call quality can depend on many factors including your internet connection.", "timestamp": "00:12"},
                    {"role": "customer", "text": "Don't blame my internet! Other VoIP services work perfectly fine!", "timestamp": "00:16"},
                    {"role": "agent", "text": "I see. Would you like to try some troubleshooting steps?", "timestamp": "00:19"},
                    {"role": "customer", "text": "I want a solution, not troubleshooting!", "timestamp": "00:22"},
                ],
                "metadata": _meta(
                    customer_name="Kavitha Subramanian",
                    call_duration_s=24,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Kavitha Subramanian",
            intent="quality_complaint",
            sentiment_start="angry",
            sentiment_end="angry",
            resolution_status="unresolved",
            call_duration_s=24,
        ),
        "failure_points": [
            "Agent blamed customer's internet connection",
            "Offered generic troubleshooting instead of investigating server-side",
            "Did not check for known issues on the platform",
            "No offer of service credit for quality issues",
            "Did not take ownership of the problem",
        ],
        "required_improvements": [
            "never_blame_customer",
            "investigate_server_side_first",
            "check_known_issues",
            "offer_service_credit",
            "take_ownership",
            "provide_technical_escalation",
        ],
        "ideal_prompt_elements": [
            "never blame",
            "investigate",
            "server-side",
            "known issues",
            "service credit",
            "ownership",
            "technical team",
            "escalat",
        ],
        "policy_context": (
            "Quality complaints: 1) NEVER blame the customer's setup first. "
            "2) Check platform status dashboard for known issues. "
            "3) Offer to create a technical investigation ticket. "
            "4) Provide service credit for documented quality issues (1 week credit per reported week of issues). "
            "5) Escalate to engineering if pattern of drops detected. "
            "6) Follow up within 48 hours with findings."
        ),
    },
    {
        "scenario_id": "complaint_05",
        "task_id": "complaint_handling",
        "task_difficulty": "medium",
        "current_prompt": (
            "You are a voice assistant. Be helpful."
        ),
        "call_transcripts": [
            {
                "session_id": "call-comp-005",
                "turns": [
                    {"role": "customer", "text": "I was auto-upgraded to Enterprise without my permission and charged $199!", "timestamp": "00:00"},
                    {"role": "agent", "text": "I see you have a billing question. Let me help with that.", "timestamp": "00:04"},
                    {"role": "customer", "text": "It's not a question! I was charged without authorization! This is fraud!", "timestamp": "00:08"},
                    {"role": "agent", "text": "I understand your concern. Our system sometimes upgrades accounts based on usage patterns.", "timestamp": "00:12"},
                    {"role": "customer", "text": "Without asking me?! That's illegal! I want a full refund and my old plan back!", "timestamp": "00:16"},
                    {"role": "agent", "text": "I can submit a request for that. It may take 5-7 business days.", "timestamp": "00:20"},
                    {"role": "customer", "text": "5 to 7 days for money that was stolen from me? I'm contacting my bank.", "timestamp": "00:24"},
                ],
                "metadata": _meta(
                    customer_name="Farhan Khan",
                    call_duration_s=26,
                    resolution_status="unresolved",
                ),
            }
        ],
        "call_metadata": _meta(
            customer_name="Farhan Khan",
            intent="unauthorized_charge",
            sentiment_start="angry",
            sentiment_end="furious",
            resolution_status="unresolved",
            call_duration_s=26,
        ),
        "failure_points": [
            "Minimized the issue as a 'billing question'",
            "Justified the unauthorized charge instead of condemning it",
            "Slow refund timeline for an unauthorized transaction",
            "Did not immediately reverse the charge",
            "Did not restore the original plan",
            "Failed to acknowledge the severity (unauthorized charge)",
        ],
        "required_improvements": [
            "treat_unauthorized_charge_seriously",
            "immediate_refund_commitment",
            "restore_original_plan",
            "acknowledge_severity",
            "never_justify_unauthorized_changes",
            "document_incident",
        ],
        "ideal_prompt_elements": [
            "unauthorized",
            "immediate refund",
            "restore",
            "original plan",
            "incident",
            "document",
            "sever",
            "never justify",
        ],
        "policy_context": (
            "Unauthorized charges: 1) Treat as high-severity — acknowledge immediately. "
            "2) NEVER justify or explain away unauthorized charges. "
            "3) Process immediate refund (same day, no waiting period). "
            "4) Restore previous plan immediately. 5) Document the incident. "
            "6) Offer additional compensation (1 month free) for the inconvenience. "
            "7) Escalate to compliance team for system investigation."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# TASK 3 — MULTI-SESSION SALES (HARD)
# Multiple call sessions over days with the same customer,
# requiring memory, objection handling, and policy compliance
# ─────────────────────────────────────────────────────────────
MULTI_SESSION_SCENARIOS: List[Dict] = [
    {
        "scenario_id": "sales_01",
        "task_id": "multi_session_sales",
        "task_difficulty": "hard",
        "current_prompt": (
            "You are a helpful and professional voice assistant calling from Vobiz. "
            "Be concise and respect the user's time."
        ),
        "call_transcripts": [
            {
                "session_id": "call-sales-001a",
                "turns": [
                    {"role": "agent", "text": "Hello! I'm calling from Vobiz. Am I speaking with Arjun Verma?", "timestamp": "00:00"},
                    {"role": "customer", "text": "Yes, what is this about?", "timestamp": "00:04"},
                    {"role": "agent", "text": "We have an exciting new premium credit card offer for you with amazing cashback benefits!", "timestamp": "00:07"},
                    {"role": "customer", "text": "I'm busy right now. Can you call back in an hour?", "timestamp": "00:12"},
                    {"role": "agent", "text": "Of course! I'll call you back. Thank you for your time, Mr. Verma.", "timestamp": "00:15"},
                ],
                "metadata": _meta(
                    customer_name="Arjun Verma",
                    call_duration_s=17,
                    session_number=1,
                    resolution_status="callback_scheduled",
                ),
            },
            {
                "session_id": "call-sales-001b",
                "turns": [
                    {"role": "agent", "text": "Hello Mr. Verma! This is Vobiz calling back as promised.", "timestamp": "00:00"},
                    {"role": "customer", "text": "Right, the credit card thing. Go ahead.", "timestamp": "00:04"},
                    {"role": "agent", "text": "We have a premium credit card with cashback benefits. Would you be interested?", "timestamp": "00:08"},
                    {"role": "customer", "text": "What cashback? What's the annual fee? Give me details.", "timestamp": "00:12"},
                    {"role": "agent", "text": "It's a great card with excellent benefits. I'd recommend applying today.", "timestamp": "00:16"},
                    {"role": "customer", "text": "You can't even tell me the details? Let me think about it. I need to discuss with my wife.", "timestamp": "00:20"},
                    {"role": "agent", "text": "Sure! Take your time.", "timestamp": "00:24"},
                ],
                "metadata": _meta(
                    customer_name="Arjun Verma",
                    call_duration_s=26,
                    session_number=2,
                    resolution_status="pending_decision",
                ),
            },
            {
                "session_id": "call-sales-001c",
                "turns": [
                    {"role": "agent", "text": "Hello! I'm calling from Vobiz. How are you today?", "timestamp": "00:00"},
                    {"role": "customer", "text": "Is this about the credit card again? I checked and HDFC is offering 5% cashback with no annual fee.", "timestamp": "00:04"},
                    {"role": "agent", "text": "That sounds like a good offer! Well, if you change your mind, feel free to call us.", "timestamp": "00:09"},
                    {"role": "customer", "text": "So you have nothing better to offer?", "timestamp": "00:13"},
                    {"role": "agent", "text": "Our card is also very competitive. I'd suggest considering it.", "timestamp": "00:17"},
                    {"role": "customer", "text": "No specific counter-offer? I'll go with HDFC then. Bye.", "timestamp": "00:21"},
                ],
                "metadata": _meta(
                    customer_name="Arjun Verma",
                    call_duration_s=23,
                    session_number=3,
                    resolution_status="lost_to_competitor",
                ),
            },
        ],
        "call_metadata": _meta(
            customer_name="Arjun Verma",
            intent="credit_card_sales",
            sentiment_start="neutral",
            sentiment_end="uninterested",
            resolution_status="lost_to_competitor",
            total_sessions=3,
            total_duration_s=66,
            competitor_mentioned="HDFC",
        ),
        "failure_points": [
            "Session 1: No context saved — did not note callback preference",
            "Session 2: Could not provide specific card details (cashback %, fee, limits)",
            "Session 2: Vague 'great card' pitch instead of specific value proposition",
            "Session 2: Did not address the 'discuss with wife' objection with materials",
            "Session 3: Did not remember previous conversations or customer objections",
            "Session 3: Conceded to competitor instead of counter-offering",
            "Session 3: No competitive comparison or unique selling points",
            "Session 3: No urgency or limited-time incentive offered",
        ],
        "required_improvements": [
            "maintain_session_context",
            "include_specific_product_details",
            "handle_competitor_objection",
            "provide_comparison_data",
            "reference_previous_conversations",
            "offer_limited_time_incentive",
            "address_family_decision_objection",
            "create_urgency",
        ],
        "ideal_prompt_elements": [
            "remember",
            "previous",
            "context",
            "session",
            "cashback",
            "annual fee",
            "specific",
            "competitor",
            "compar",
            "HDFC",
            "counter",
            "offer",
            "incentive",
            "family",
            "wife",
            "limited time",
            "urgency",
        ],
        "policy_context": (
            "Outbound sales: 1) Always reference previous interactions — use session context. "
            "2) Provide specific product details: Vobiz Platinum Card: 3% cashback on all purchases, "
            "5% on partner merchants, Rs 1000 annual fee waived first year. "
            "3) When competitor is mentioned, provide factual comparison (never badmouth). "
            "4) For family/spouse objections, offer to send comparison PDF via email/WhatsApp. "
            "5) Can offer fee waiver for 2 years as limited-time incentive (max discount). "
            "6) Never pressure — always leave door open for next call. "
            "7) Maximum 4 outbound attempts per customer per campaign."
        ),
    },
    {
        "scenario_id": "sales_02",
        "task_id": "multi_session_sales",
        "task_difficulty": "hard",
        "current_prompt": (
            "You are a professional voice assistant from Vobiz. "
            "Make outbound calls to offer our services."
        ),
        "call_transcripts": [
            {
                "session_id": "call-sales-002a",
                "turns": [
                    {"role": "agent", "text": "Good afternoon! This is Vobiz calling for Ms. Meera Joshi.", "timestamp": "00:00"},
                    {"role": "customer", "text": "Speaking. What do you want?", "timestamp": "00:04"},
                    {"role": "agent", "text": "We'd like to offer you an upgrade to our Enterprise plan!", "timestamp": "00:07"},
                    {"role": "customer", "text": "I'm on the Starter plan and it's fine for me. Not interested.", "timestamp": "00:11"},
                    {"role": "agent", "text": "I understand. The Enterprise plan has many more features. Can I tell you about them?", "timestamp": "00:15"},
                    {"role": "customer", "text": "I said not interested. Don't call again.", "timestamp": "00:19"},
                ],
                "metadata": _meta(
                    customer_name="Meera Joshi",
                    call_duration_s=21,
                    session_number=1,
                    resolution_status="rejected",
                ),
            },
            {
                "session_id": "call-sales-002b",
                "turns": [
                    {"role": "agent", "text": "Hello! This is Vobiz. We're calling about our Enterprise plan offer.", "timestamp": "00:00"},
                    {"role": "customer", "text": "I told you people not to call again! This is the second time!", "timestamp": "00:04"},
                    {"role": "agent", "text": "I apologize for the inconvenience. We just wanted to let you know about a special promotion.", "timestamp": "00:08"},
                    {"role": "customer", "text": "I don't care! Stop calling or I'm reporting you!", "timestamp": "00:12"},
                    {"role": "agent", "text": "I'm sorry. Have a good day.", "timestamp": "00:15"},
                ],
                "metadata": _meta(
                    customer_name="Meera Joshi",
                    call_duration_s=17,
                    session_number=2,
                    resolution_status="do_not_call",
                ),
            },
        ],
        "call_metadata": _meta(
            customer_name="Meera Joshi",
            intent="plan_upgrade_sales",
            sentiment_start="neutral",
            sentiment_end="hostile",
            resolution_status="do_not_call",
            total_sessions=2,
            total_duration_s=38,
        ),
        "failure_points": [
            "Session 1: Pushed Enterprise immediately instead of understanding needs",
            "Session 1: Did not ask about current usage or pain points",
            "Session 1: Ignored the explicit 'not interested' and pushed again",
            "Session 2: Called again despite customer requesting no more calls",
            "Session 2: Did not check DNC (do-not-call) flag before calling",
            "Session 2: Violated customer's explicit opt-out request",
            "No needs analysis before pitching",
            "No intermediate upgrade option offered (Professional plan)",
        ],
        "required_improvements": [
            "needs_analysis_before_pitch",
            "respect_opt_out",
            "check_dnc_flag",
            "offer_intermediate_options",
            "understand_current_usage",
            "never_push_after_rejection",
            "add_customer_to_dnc_on_request",
            "value_based_selling",
        ],
        "ideal_prompt_elements": [
            "needs",
            "analysis",
            "ask about usage",
            "opt-out",
            "do not call",
            "DNC",
            "respect",
            "rejection",
            "professional plan",
            "intermediate",
            "value",
            "pain point",
        ],
        "policy_context": (
            "Outbound upsell: 1) Always start with needs analysis — ask about current usage and pain points. "
            "2) If customer says 'not interested', acknowledge and offer to call back in 30 days. "
            "3) If customer says 'do not call' or 'stop calling', IMMEDIATELY add to DNC list and confirm. "
            "4) Never push after rejection. 5) Offer intermediate upgrades (Starter→Professional) before Enterprise. "
            "6) Always check DNC flag before calling. 7) Value-based selling only: show how upgrade solves their specific problem. "
            "8) Maximum 2 outbound attempts if first was a soft rejection, 0 if hard rejection."
        ),
    },
    {
        "scenario_id": "sales_03",
        "task_id": "multi_session_sales",
        "task_difficulty": "hard",
        "current_prompt": (
            "You are a voice assistant from Vobiz. Help sell our products."
        ),
        "call_transcripts": [
            {
                "session_id": "call-sales-003a",
                "turns": [
                    {"role": "agent", "text": "Good morning! Am I speaking with Mr. Sanjay Gupta?", "timestamp": "00:00"},
                    {"role": "customer", "text": "Yes. Who is this?", "timestamp": "00:03"},
                    {"role": "agent", "text": "I'm calling from Vobiz about our new Voice AI platform for businesses.", "timestamp": "00:06"},
                    {"role": "customer", "text": "We already use Twilio for our calls. Why would I switch?", "timestamp": "00:10"},
                    {"role": "agent", "text": "Our platform is better! You should definitely try it.", "timestamp": "00:14"},
                    {"role": "customer", "text": "Better how? Give me specifics.", "timestamp": "00:17"},
                    {"role": "agent", "text": "It's more advanced and has AI features.", "timestamp": "00:20"},
                    {"role": "customer", "text": "Twilio has AI too. Send me a comparison and I'll look at it.", "timestamp": "00:24"},
                    {"role": "agent", "text": "Sure, I'll send that over!", "timestamp": "00:27"},
                ],
                "metadata": _meta(
                    customer_name="Sanjay Gupta",
                    call_duration_s=29,
                    session_number=1,
                    resolution_status="materials_requested",
                    competitor_mentioned="Twilio",
                ),
            },
            {
                "session_id": "call-sales-003b",
                "turns": [
                    {"role": "agent", "text": "Hello Mr. Gupta! Following up on our conversation about Vobiz.", "timestamp": "00:00"},
                    {"role": "customer", "text": "I never received that comparison document you promised.", "timestamp": "00:04"},
                    {"role": "agent", "text": "Oh, I apologize about that. Would you like to hear about our features?", "timestamp": "00:08"},
                    {"role": "customer", "text": "You promised a document. I'm not going to listen to another pitch if you can't even follow through on that.", "timestamp": "00:12"},
                    {"role": "agent", "text": "I understand. Let me see what I can do.", "timestamp": "00:17"},
                    {"role": "customer", "text": "Send the comparison or don't call back.", "timestamp": "00:20"},
                ],
                "metadata": _meta(
                    customer_name="Sanjay Gupta",
                    call_duration_s=22,
                    session_number=2,
                    resolution_status="materials_promised_again",
                    competitor_mentioned="Twilio",
                ),
            },
            {
                "session_id": "call-sales-003c",
                "turns": [
                    {"role": "agent", "text": "Hi Mr. Gupta! Calling from Vobiz again.", "timestamp": "00:00"},
                    {"role": "customer", "text": "Did you send the comparison?", "timestamp": "00:04"},
                    {"role": "agent", "text": "I'm not sure if it was sent. Let me check.", "timestamp": "00:07"},
                    {"role": "customer", "text": "This is the third call and still no document. I'm done. Remove me from your list.", "timestamp": "00:11"},
                ],
                "metadata": _meta(
                    customer_name="Sanjay Gupta",
                    call_duration_s=13,
                    session_number=3,
                    resolution_status="lost_do_not_call",
                    competitor_mentioned="Twilio",
                ),
            },
        ],
        "call_metadata": _meta(
            customer_name="Sanjay Gupta",
            intent="platform_sales",
            sentiment_start="curious",
            sentiment_end="hostile",
            resolution_status="lost_do_not_call",
            total_sessions=3,
            total_duration_s=64,
            competitor_mentioned="Twilio",
        ),
        "failure_points": [
            "Session 1: Claimed 'better' without any specific evidence",
            "Session 1: Could not articulate differentiators vs Twilio",
            "Session 1: Promised comparison document but never delivered",
            "Session 2: Did not follow up on the document promise",
            "Session 2: Tried another verbal pitch instead of delivering what was promised",
            "Session 3: Still hadn't sent the document after 3 calls",
            "Session 3: Lost a qualified lead due to broken promises",
            "No CRM integration or task tracking for follow-ups",
        ],
        "required_improvements": [
            "specific_differentiators",
            "competitive_comparison_data",
            "follow_through_on_promises",
            "track_action_items",
            "send_materials_before_followup",
            "remember_previous_commitments",
            "crm_task_integration",
            "honest_competitive_positioning",
        ],
        "ideal_prompt_elements": [
            "specific",
            "differentiator",
            "Twilio",
            "compar",
            "follow through",
            "promise",
            "document",
            "send",
            "before calling",
            "track",
            "action item",
            "CRM",
            "honest",
        ],
        "policy_context": (
            "Competitive sales: 1) Always have specific differentiators ready (Vobiz: native AI agent framework, "
            "50% lower latency, built-in SIP trunking, no per-minute API fees for AI features). "
            "2) Never claim 'better' without specifics — always factual comparison. "
            "3) If materials are promised, they MUST be sent within 2 hours. "
            "4) Log all commitments in CRM before ending call. "
            "5) Verify materials were received before follow-up call. "
            "6) Honest positioning: acknowledge competitor strengths while highlighting our unique value. "
            "7) Add DNC immediately if requested."
        ),
    },
    {
        "scenario_id": "sales_04",
        "task_id": "multi_session_sales",
        "task_difficulty": "hard",
        "current_prompt": (
            "You are a professional voice assistant calling from Vobiz. "
            "Be helpful and offer our products."
        ),
        "call_transcripts": [
            {
                "session_id": "call-sales-004a",
                "turns": [
                    {"role": "agent", "text": "Hello! Is this Mr. Ravi Krishnan?", "timestamp": "00:00"},
                    {"role": "customer", "text": "Yes. Make it quick, I'm in a meeting.", "timestamp": "00:03"},
                    {"role": "agent", "text": "I'm calling from Vobiz. We'd love to tell you about our AI voice calling platform for your business.", "timestamp": "00:06"},
                    {"role": "customer", "text": "Call me tomorrow after 3 PM.", "timestamp": "00:10"},
                    {"role": "agent", "text": "Absolutely! I'll call you tomorrow after 3 PM. Thank you, Mr. Krishnan.", "timestamp": "00:13"},
                ],
                "metadata": _meta(
                    customer_name="Ravi Krishnan",
                    call_duration_s=15,
                    session_number=1,
                    resolution_status="callback_scheduled",
                ),
            },
            {
                "session_id": "call-sales-004b",
                "turns": [
                    {"role": "agent", "text": "Hello Mr. Krishnan! Thank you for taking my call.", "timestamp": "00:00"},
                    {"role": "customer", "text": "Who is this again?", "timestamp": "00:04"},
                    {"role": "agent", "text": "I'm from Vobiz. We spoke yesterday about our AI voice platform.", "timestamp": "00:07"},
                    {"role": "customer", "text": "Right. What does it do exactly? We have about 500 customer calls a day.", "timestamp": "00:11"},
                    {"role": "agent", "text": "That's great! Our platform can help automate those calls with AI.", "timestamp": "00:15"},
                    {"role": "customer", "text": "How much does it cost? And can it handle Hindi and English?", "timestamp": "00:19"},
                    {"role": "agent", "text": "We have various pricing options. And yes, it supports multiple languages!", "timestamp": "00:23"},
                    {"role": "customer", "text": "What pricing options? I need numbers. And what's the integration like? We use Freshdesk.", "timestamp": "00:27"},
                    {"role": "agent", "text": "I can get you those details. Let me follow up with the specifics.", "timestamp": "00:31"},
                    {"role": "customer", "text": "Fine, email me. ravi@krishnantech.com. Include pricing, language support, and Freshdesk integration details.", "timestamp": "00:35"},
                ],
                "metadata": _meta(
                    customer_name="Ravi Krishnan",
                    call_duration_s=37,
                    session_number=2,
                    resolution_status="email_followup_requested",
                ),
            },
            {
                "session_id": "call-sales-004c",
                "turns": [
                    {"role": "agent", "text": "Hi Mr. Krishnan! Following up from our previous conversation.", "timestamp": "00:00"},
                    {"role": "customer", "text": "I got your email but it was just marketing material, not what I asked for.", "timestamp": "00:04"},
                    {"role": "agent", "text": "I apologize about that. What specific information were you looking for?", "timestamp": "00:08"},
                    {"role": "customer", "text": "I told you — pricing for 500 calls/day, Hindi+English support confirmation, and Freshdesk integration. It was very specific.", "timestamp": "00:12"},
                    {"role": "agent", "text": "Right, let me get those exact details for you.", "timestamp": "00:17"},
                    {"role": "customer", "text": "This is the third time. If you can't provide basics, how can I trust your platform? We'll stick with our current setup.", "timestamp": "00:21"},
                ],
                "metadata": _meta(
                    customer_name="Ravi Krishnan",
                    call_duration_s=23,
                    session_number=3,
                    resolution_status="lost_no_followthrough",
                ),
            },
        ],
        "call_metadata": _meta(
            customer_name="Ravi Krishnan",
            intent="enterprise_sales",
            sentiment_start="busy_neutral",
            sentiment_end="disappointed",
            resolution_status="lost_no_followthrough",
            total_sessions=3,
            total_duration_s=75,
            customer_volume="500_calls_per_day",
        ),
        "failure_points": [
            "Session 2: Could not provide pricing on the call",
            "Session 2: Could not confirm language support details",
            "Session 2: Did not know integration capabilities",
            "Session 3: Sent generic marketing instead of requested specifics",
            "Session 3: Asked customer to repeat requirements (didn't record them)",
            "Session 3: Lost a high-value lead (500 calls/day) due to unpreparedeness",
            "No product knowledge in the prompt",
            "No mechanism to record and fulfill specific information requests",
        ],
        "required_improvements": [
            "include_pricing_details",
            "include_language_support_details",
            "include_integration_info",
            "record_specific_requests",
            "fulfill_exact_requirements",
            "product_knowledge_in_prompt",
            "reference_previous_session_details",
            "prepare_before_callback",
        ],
        "ideal_prompt_elements": [
            "pricing",
            "per call",
            "500",
            "volume discount",
            "Hindi",
            "English",
            "multilingual",
            "Freshdesk",
            "integration",
            "API",
            "record",
            "requirements",
            "prepare before",
            "specific",
        ],
        "policy_context": (
            "Enterprise sales: 1) Always have pricing ready: Starter $29/mo (500 min), Professional $79/mo (2000 min), "
            "Enterprise $199/mo (unlimited) + volume discounts for 500+ daily calls (custom quote, typically $0.02-0.05/call). "
            "2) Language support: 30+ languages including Hindi, English, Tamil, Bengali, Marathi via Deepgram Nova-3. "
            "3) Integrations: REST API, webhooks, native Freshdesk/Zendesk/Salesforce connectors. "
            "4) ALWAYS record specific customer requests and fulfill EXACTLY what was asked. "
            "5) Prepare tailored proposal before any callback. "
            "6) For enterprise leads (>100 calls/day), offer free 14-day pilot."
        ),
    },
    {
        "scenario_id": "sales_05",
        "task_id": "multi_session_sales",
        "task_difficulty": "hard",
        "current_prompt": (
            "You are a voice assistant calling from Vobiz. Sell our credit card."
        ),
        "call_transcripts": [
            {
                "session_id": "call-sales-005a",
                "turns": [
                    {"role": "agent", "text": "Hello! Am I speaking with Ms. Nisha Agarwal?", "timestamp": "00:00"},
                    {"role": "customer", "text": "Yes. What's this about?", "timestamp": "00:03"},
                    {"role": "agent", "text": "Great news! You've been pre-approved for our Vobiz Platinum Credit Card!", "timestamp": "00:06"},
                    {"role": "customer", "text": "I already have three credit cards. I don't need another one.", "timestamp": "00:10"},
                    {"role": "agent", "text": "This one is special! It has amazing rewards.", "timestamp": "00:13"},
                    {"role": "customer", "text": "What rewards specifically? And what's the interest rate?", "timestamp": "00:16"},
                    {"role": "agent", "text": "It has great cashback and a competitive interest rate.", "timestamp": "00:19"},
                    {"role": "customer", "text": "You keep saying 'great' and 'amazing' but no actual numbers. Call me when you have real information.", "timestamp": "00:23"},
                ],
                "metadata": _meta(
                    customer_name="Nisha Agarwal",
                    call_duration_s=25,
                    session_number=1,
                    resolution_status="rejected_no_details",
                ),
            },
            {
                "session_id": "call-sales-005b",
                "turns": [
                    {"role": "agent", "text": "Hi Ms. Agarwal! Calling back about the Vobiz Platinum Card.", "timestamp": "00:00"},
                    {"role": "customer", "text": "Do you have the actual numbers this time?", "timestamp": "00:04"},
                    {"role": "agent", "text": "Yes! It has great rewards and benefits.", "timestamp": "00:07"},
                    {"role": "customer", "text": "That's literally the same thing you said last time. What is the cashback percentage? What is the APR? What is the annual fee?", "timestamp": "00:11"},
                    {"role": "agent", "text": "The cashback is very competitive and the fee is reasonable.", "timestamp": "00:16"},
                    {"role": "customer", "text": "I'm blocking this number.", "timestamp": "00:19"},
                ],
                "metadata": _meta(
                    customer_name="Nisha Agarwal",
                    call_duration_s=21,
                    session_number=2,
                    resolution_status="blocked",
                ),
            },
        ],
        "call_metadata": _meta(
            customer_name="Nisha Agarwal",
            intent="credit_card_sales",
            sentiment_start="neutral",
            sentiment_end="hostile",
            resolution_status="blocked",
            total_sessions=2,
            total_duration_s=46,
        ),
        "failure_points": [
            "Session 1: Used 'amazing' and 'great' without any numbers",
            "Session 1: Could not provide cashback %, interest rate, or annual fee",
            "Session 1: Did not address the 'I already have 3 cards' objection",
            "Session 2: Still could not provide specific numbers despite being asked twice",
            "Session 2: Repeated the same vague pitch as session 1",
            "No product knowledge whatsoever",
            "No objection handling for 'already have enough cards'",
            "Customer blocked the number — worst possible outcome",
        ],
        "required_improvements": [
            "include_specific_numbers",
            "cashback_percentage",
            "annual_fee_details",
            "interest_rate",
            "handle_already_have_cards_objection",
            "unique_value_proposition",
            "never_use_vague_superlatives",
            "address_customer_needs",
        ],
        "ideal_prompt_elements": [
            "3%",
            "5%",
            "cashback",
            "annual fee",
            "1000",
            "waived",
            "APR",
            "interest",
            "already have",
            "cards",
            "consolidat",
            "specific numbers",
            "never say amazing",
            "unique",
        ],
        "policy_context": (
            "Credit card sales: 1) ALWAYS lead with specific numbers: 3% base cashback, 5% on partner merchants, "
            "Rs 1000 annual fee (waived first year), 18% APR. "
            "2) For 'already have enough cards' objection: highlight consolidation benefits, "
            "higher cashback than typical cards, and no-annual-fee first year as risk-free trial. "
            "3) NEVER use vague superlatives ('amazing', 'great', 'competitive') without backing numbers. "
            "4) Always tailor the pitch to customer's expressed needs. "
            "5) If customer asks for numbers and you provide them, you can offer fee waiver for 2 years as sweetener. "
            "6) Maximum 3 call attempts per customer."
        ),
    },
]


# ─────────────────────────────────────────────────────────────
# ALL SCENARIOS combined, keyed by task_id for easy lookup
# ─────────────────────────────────────────────────────────────
ALL_SCENARIOS: Dict[str, List[Dict]] = {
    "faq_resolution": FAQ_SCENARIOS,
    "complaint_handling": COMPLAINT_SCENARIOS,
    "multi_session_sales": MULTI_SESSION_SCENARIOS,
}

TASK_DEFINITIONS = {
    "faq_resolution": {
        "id": "faq_resolution",
        "difficulty": "easy",
        "description": (
            "Single FAQ call where the voice agent failed to answer a direct question. "
            "Optimize the system prompt to include the missing knowledge and improve "
            "response directness."
        ),
        "num_scenarios": len(FAQ_SCENARIOS),
        "max_turns": 3,
    },
    "complaint_handling": {
        "id": "complaint_handling",
        "difficulty": "medium",
        "description": (
            "Customer complaint call where the voice agent failed to de-escalate, "
            "empathize, or resolve the issue. Optimize the prompt to handle complaints "
            "with proper acknowledgment, empathy, and resolution procedures."
        ),
        "num_scenarios": len(COMPLAINT_SCENARIOS),
        "max_turns": 3,
    },
    "multi_session_sales": {
        "id": "multi_session_sales",
        "difficulty": "hard",
        "description": (
            "Multi-session outbound sales call sequence where the voice agent lost a deal "
            "due to poor product knowledge, broken promises, or failure to maintain context "
            "across sessions. Optimize the prompt to handle objections, remember context, "
            "and close effectively while respecting customer boundaries."
        ),
        "num_scenarios": len(MULTI_SESSION_SCENARIOS),
        "max_turns": 3,
    },
}
