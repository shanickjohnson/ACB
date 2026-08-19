"""
Compiles the full ACB agent graph:

  START -> input_guardrail --(blocked)--> output_guardrail -> END
                            \\-(ok)------> supervisor

  supervisor dispatches (Command(goto=...)) to exactly one of:
    payments, cards, loans, mortgage, onboarding, faq, compliance, escalation

  Each specialist finishes the turn directly (Command(goto="output_guardrail"))
  or escalates (Command(goto="escalation")). loans/mortgage have one more
  option: hand back to the supervisor (Command(goto="supervisor")) when
  they want a compliance check — the supervisor then re-dispatches, capped
  at MAX_HANDOFFS hops (see supervisor.py) to guarantee termination.

  escalation -> output_guardrail -> END

Run with `run_graph(...)` from the API layer's /chat endpoint. State
persists per session via the checkpointer (thread_id-keyed), so no
separate in-memory session dict is needed.
"""

import os
import sqlite3
from functools import partial

import google.genai as genai
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ..guardrails import input_guardrail_node, output_guardrail_node
from .domain_agents import (
    cards_agent,
    compliance_agent,
    faq_agent,
    loans_agent,
    mortgage_agent,
    onboarding_agent,
    payments_agent,
)
from .escalation import escalation_node
from .state import ACBState
from .supervisor import SPECIALIST_ROUTES, route_after_guardrail, supervisor_node

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Every specialist finishes by either escalating or going straight to
# output_guardrail; loans/mortgage can additionally hand back to the
# supervisor for a compliance hop.
SIMPLE_SPECIALISTS = {
    "payments": payments_agent,
    "cards": cards_agent,
    "onboarding": onboarding_agent,
    "faq": faq_agent,
    "compliance": compliance_agent,
}
REROUTABLE_SPECIALISTS = {
    "loans": loans_agent,
    "mortgage": mortgage_agent,
}


def build_graph():
    graph = StateGraph(ACBState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node(
        "supervisor",
        partial(supervisor_node, genai_client=_client),
        destinations=tuple(SPECIALIST_ROUTES) + ("output_guardrail",),
    )
    for name, fn in SIMPLE_SPECIALISTS.items():
        graph.add_node(name, partial(fn, genai_client=_client), destinations=("escalation", "output_guardrail"))
    for name, fn in REROUTABLE_SPECIALISTS.items():
        graph.add_node(
            name,
            partial(fn, genai_client=_client),
            destinations=("escalation", "output_guardrail", "supervisor"),
        )
    graph.add_node("escalation", escalation_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"blocked": "output_guardrail", "supervisor": "supervisor"},
    )
    graph.add_edge("escalation", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    conn = sqlite3.connect("acb_sessions.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_graph(message: str, session_id: str, language: str = "en") -> dict:
    """Drop-in reply function for the /chat endpoint. Returns the same
    shape the endpoint responds with."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke(
        {"message": message, "session_id": session_id, "language": language, "handoff_count": 0},
        config=config,
    )

    return {
        "reply": result.get("final_reply", ""),
        "session_id": session_id,
        "language": language,
        "route": result.get("route"),
        "escalated": result.get("agent_name") == "escalation",
    }
