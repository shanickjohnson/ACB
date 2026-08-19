"""
Supervisor node.

Unlike the old single-shot router (one classification call, one specialist,
done), the supervisor can be re-entered mid-turn: a specialist can hand
control back to it via Command(goto="supervisor") when it wants a different
lane to take over (e.g. loans handing an uncertain rate to compliance). The
supervisor is a silent dispatcher — its own output is never shown to the
customer, only its routing decision.

Guardrail-blocked messages never reach this node at all: input_guardrail's
own conditional edge sends them straight to output_guardrail (see
graph.py). That's stricter than routing them through the supervisor to an
"escalation" target, which is what a literal reading of "if an upstream
guardrail flagged the message, route to escalation" would do — an
injection/jailbreak attempt shouldn't reach any LLM call, including this
one, and the existing bypass already guarantees that.
"""

import json

from google.genai import types

from .. import rag
from ..config import CHAT_MODEL
from .state import ACBState

SPECIALIST_ROUTES = [
    "payments",
    "cards",
    "loans",
    "mortgage",
    "onboarding",
    "faq",
    "compliance",
    "escalation",
]

MAX_HANDOFFS = 3

SUPERVISOR_SYSTEM_PROMPT = """
You are the silent dispatcher for ACB Caribbean's banking assistant. You
never speak to the customer directly — you only decide which specialist
handles the message next.

Specialists:
- payments: sending/receiving money, transfers, wires, bill pay, payment issues
- cards: debit/credit cards, activation, blocks, lost/stolen reports, card fees, PINs
- loans: personal/auto/business (non-mortgage) loans, loan rates, applications, payments
- mortgage: home loans, mortgage rates, applications, refinancing
- onboarding: opening a new account, required documents, becoming a customer
- faq: branch hours/locations, general bank info — the catch-all ONLY when
  nothing above clearly fits
- compliance: specific fee/rate/policy questions that need to be answered
  strictly from official reference data, or a number another specialist
  wasn't confident about
- escalation: the customer explicitly wants a human, is angry or
  complaining, is reporting fraud or a lost/stolen card, or nothing else
  can help

Rules:
- Prefer the most specific matching specialist. Use faq only when no other
  lane clearly applies.
- If this is a re-route (a specialist already responded this turn) and
  nothing further is needed, respond with "FINISH".
- Never respond with "FINISH" before any specialist has responded this turn.

Respond with a JSON object: {"next": "<one of the specialist names%s>", "reasoning": "<one short sentence>"}.
"""


def _build_schema(allow_finish: bool) -> dict:
    enum = list(SPECIALIST_ROUTES) + (["FINISH"] if allow_finish else [])
    return {
        "type": "OBJECT",
        "properties": {
            "next": {"type": "STRING", "enum": enum},
            "reasoning": {"type": "STRING"},
        },
        "required": ["next", "reasoning"],
    }


def supervisor_node(state: ACBState, genai_client):
    from langgraph.types import Command

    message = state["message"]
    handoff_count = state.get("handoff_count", 0)
    has_responded = bool(state.get("agent_reply"))

    jurisdiction = state.get("jurisdiction")
    if jurisdiction is None:
        jurisdiction = rag.detect_jurisdiction(message)

    # Deterministic safety net: don't even ask the model once the hop cap
    # is hit, so a stubborn/confused supervisor can't loop forever.
    if handoff_count >= MAX_HANDOFFS:
        return Command(
            goto="escalation",
            update={"jurisdiction": jurisdiction, "escalation_reason": "handoff limit reached"},
        )

    allow_finish = has_responded
    prompt_suffix = ' or "FINISH" if the turn is complete' if allow_finish else ""

    context_lines = [f'Customer message: "{message}"']
    if state.get("route_suggestion"):
        context_lines.append(f"Previous specialist suggested handing off to: {state['route_suggestion']}")
    if jurisdiction:
        context_lines.append(f"Detected jurisdiction: {jurisdiction}")

    response = genai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=[{"role": "user", "parts": [{"text": "\n".join(context_lines)}]}],
        config=types.GenerateContentConfig(
            system_instruction=SUPERVISOR_SYSTEM_PROMPT % prompt_suffix,
            max_output_tokens=200,
            temperature=0,
            response_mime_type="application/json",
            response_schema=_build_schema(allow_finish),
        ),
    )

    next_route = "faq"
    try:
        decision = json.loads(response.text or "{}")
        candidate = str(decision.get("next", "")).strip()
        if candidate == "FINISH" and allow_finish:
            next_route = "FINISH"
        elif candidate in SPECIALIST_ROUTES:
            next_route = candidate
    except (json.JSONDecodeError, AttributeError):
        pass  # keep the safe faq fallback

    update = {"jurisdiction": jurisdiction, "route_suggestion": None}
    if next_route == "FINISH":
        return Command(goto="output_guardrail", update=update)

    update["route"] = next_route
    return Command(goto=next_route, update=update)


def route_after_guardrail(state: ACBState) -> str:
    """Conditional edge out of the input guardrail node. Blocked messages
    go straight to output_guardrail — they never reach the supervisor or
    any specialist, guardrail or not."""
    return "blocked" if state.get("blocked") else "supervisor"
