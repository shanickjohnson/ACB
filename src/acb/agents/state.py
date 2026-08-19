"""
Shared state passed between every node in the ACB LangGraph agent.

Keep this small and serializable — it gets written to the checkpointer
on every node transition, so anything heavy (full retrieved documents,
big tool payloads) should be summarized before it lands here rather
than stored verbatim turn over turn.
"""

from __future__ import annotations

from typing import TypedDict


class Turn(TypedDict):
    role: str  # "user" | "model"
    text: str


class ACBState(TypedDict, total=False):
    # --- input ---
    message: str  # the current raw user message
    history: list[Turn]  # prior turns for this session (bounded, see graph.py)
    session_id: str
    language: str  # "en" | "fr" | "es" | "nl"

    # --- guardrails ---
    blocked: bool  # True if input guardrail refused the request outright
    block_reason: str | None  # "injection" | "jailbreak" | None
    pii_redacted_input: bool

    # --- routing ---
    route: str | None  # one of the ROUTE_NAMES in supervisor.py
    jurisdiction: str | None  # "Antigua & Barbuda" | "Grenada" | None (unknown)
    handoff_count: int  # number of times a specialist has handed back to the
    # supervisor for re-routing this turn (e.g. loans -> compliance).
    # Capped in supervisor.py to prevent re-routing loops.
    route_suggestion: str | None  # e.g. "compliance", set by a specialist
    # handing back to the supervisor; the supervisor is not bound to it,
    # but treats it as a strong signal.

    # --- domain agent output ---
    agent_reply: str | None
    agent_name: str | None

    # --- escalation ---
    escalation_reason: str | None

    # --- output ---
    final_reply: str | None
