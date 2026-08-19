"""
Router / supervisor node.

One structured-output Gemini call classifies the message into exactly
one of the domain agents. This replaces keyword/regex intent detection
with a single cheap, reliable classification call — the CSV fast path
in tools.py still runs first inside each domain agent so exact-match
questions never even need this call to result in an LLM answer.
"""

from google.genai import types

from ..config import CHAT_MODEL
from .state import ACBState

AGENT_NAMES = [
    "payments",
    "cards",
    "loans",
    "mortgage",
    "onboarding",
    "faq",
]

ROUTER_SYSTEM_PROMPT = """
You are the intent router for ACB Caribbean's banking assistant.
Classify the customer's message into exactly one category:

- payments: sending/receiving money, transfers, wires, bill pay, payment issues
- cards: debit/credit cards, card activation, card blocks, card fees, PINs
- loans: personal/business loans, loan rates, loan applications, loan payments
- mortgage: home loans, mortgage rates, mortgage applications, refinancing
- onboarding: opening a new account, required documents, becoming a customer
- faq: branch hours/locations, general bank info, anything that doesn't fit
  the categories above, or anything you're unsure about

Respond with ONLY the category name, nothing else.
"""


def router_node(state: ACBState, genai_client) -> dict:
    message = state["message"]

    response = genai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=[{"role": "user", "parts": [{"text": message}]}],
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM_PROMPT,
            max_output_tokens=10,
            temperature=0,
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": AGENT_NAMES},
        ),
    )

    route = (response.text or "").strip().lower()
    if route not in AGENT_NAMES:
        route = "faq"  # safest default: general-knowledge agent, not a dead end

    return {"route": route}


def route_after_guardrail(state: ACBState) -> str:
    """Conditional edge out of the input guardrail node."""
    return "blocked" if state.get("blocked") else "router"


def route_to_agent(state: ACBState) -> str:
    """Conditional edge out of the router node."""
    return state.get("route", "faq")
