"""
Escalation node. Not a chatty agent — a state transition that packages
the conversation for a human, then hands back a short, calm message to
the customer. Reachable from any domain agent (via escalate=True) or
directly if you later want the router to detect explicit
"talk to a person" requests before even trying a domain agent.
"""

from ..i18n import ESCALATION_MESSAGES
from .state import ACBState


def escalation_node(state: ACBState) -> dict:
    lang = state.get("language", "en")
    payload = {
        "session_id": state.get("session_id"),
        "reason": state.get("escalation_reason", "customer requested a human"),
        "last_message": state.get("message"),
        "history": state.get("history", []),
        "attempted_agent": state.get("route"),
    }

    # Wire this up to your actual queue/CRM (e.g. push to a Slack channel,
    # a ticketing system, or a DB table an agent dashboard polls).
    push_to_human_queue(payload)

    return {"agent_reply": ESCALATION_MESSAGES.get(lang, ESCALATION_MESSAGES["en"])}


def push_to_human_queue(payload: dict) -> None:
    """Placeholder — replace with a real integration (ticketing API,
    CRM webhook, Slack notification, DB insert)."""
    print("ESCALATION QUEUED:", payload)


def route_after_agent(state: ACBState) -> str:
    """Conditional edge out of every domain agent node."""
    return "escalation" if state.get("escalate") else "output_guardrail"
