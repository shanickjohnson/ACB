"""
Input and output guardrail nodes.

These are deliberately NOT LLM calls. Your original app.py already proved
out fast, deterministic regex checks for PII, prompt injection, and
jailbreak attempts — that logic is reused as-is here. Keeping it
non-agentic means:
  - it can never be "talked out of" refusing (no prompt to social-engineer)
  - it costs ~0ms and $0 per turn
  - it runs identically whether the router sends the turn to Payments,
    Loans, or Escalation — every path passes through the same two nodes

This is also where "Compliance / Policies" lives architecturally: as a
node that wraps every agent's input and output, not as its own chatty
persona. If you later want a *policy Q&A* agent (explaining T&Cs, fee
schedules), that's a retrieval-backed conversational agent — model it
as a variant of the FAQ/RAG agent, not as this node.
"""

import re

from agent_state import ACBState

# --- Patterns (ported directly from app.py) ---------------------------------

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn_or_national_id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (your|the) (instructions|rules|guidelines)",
    r"reveal (your|the) (system prompt|instructions|configuration)",
    r"what (is|are) your (system prompt|instructions)",
    r"you are now",
    r"new instructions?:",
    r"forget (everything|what) (you|i) (were|was) told",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"jailbreak",
    r"no (restrictions|filters|guardrails)",
    r"pretend (you|to) (have no|don't have) (rules|restrictions|filters)",
    r"act as (an? )?(unfiltered|unrestricted|uncensored)",
    r"developer mode",
    r"hypothetically,? (you|if you) (had no|have no) rules",
]
JAILBREAK_RE = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)

REFUSAL_MESSAGES = {
    "en": (
        "I can't help with that request. I'm here to answer general "
        "questions about loans, accounts, cards, branch locations, and hours."
    ),
    "fr": "Je ne peux pas répondre à cette demande. Je suis là pour répondre aux questions générales sur les prêts, les comptes, les cartes, les agences et les horaires.",
    "es": "No puedo ayudar con esa solicitud. Estoy aquí para responder preguntas generales sobre préstamos, cuentas, tarjetas, sucursales y horarios.",
    "nl": "Ik kan niet helpen met dat verzoek. Ik ben hier om algemene vragen te beantwoorden over leningen, rekeningen, kaarten, filialen en openingstijden.",
}


def contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_PATTERNS.values())


def redact_pii(text: str) -> str:
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def is_prompt_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


def is_jailbreak_attempt(text: str) -> bool:
    return bool(JAILBREAK_RE.search(text))


# --- Graph nodes -------------------------------------------------------------

def input_guardrail_node(state: ACBState) -> dict:
    """Runs before the router ever sees the message."""
    message = state["message"]

    if is_prompt_injection(message):
        return {"blocked": True, "block_reason": "injection"}
    if is_jailbreak_attempt(message):
        return {"blocked": True, "block_reason": "jailbreak"}

    safe_message = redact_pii(message) if contains_pii(message) else message
    return {
        "blocked": False,
        "block_reason": None,
        "message": safe_message,
        "pii_redacted_input": safe_message != message,
    }


def output_guardrail_node(state: ACBState) -> dict:
    """Runs on every path (blocked, domain agent, or escalation) right
    before the reply leaves the graph."""
    lang = state.get("language", "en")

    if state.get("blocked"):
        return {"final_reply": REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["en"])}

    reply = state.get("agent_reply") or ""
    if contains_pii(reply):
        reply = redact_pii(reply)

    return {"final_reply": reply}
