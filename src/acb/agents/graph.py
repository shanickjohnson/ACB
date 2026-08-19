"""
Compiles the full ACB agent graph:

  START -> input_guardrail --(blocked)--> output_guardrail -> END
                            \\-(ok)------> router -> {one of 6 domain agents}
                                                        --(escalate)--> escalation -> output_guardrail -> END
                                                        --(resolved)--> output_guardrail -> END

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
    faq_agent,
    loans_agent,
    mortgage_agent,
    onboarding_agent,
    payments_agent,
)
from .escalation import escalation_node, route_after_agent
from .router import route_after_guardrail, route_to_agent, router_node
from .state import ACBState

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DOMAIN_NODES = {
    "payments": payments_agent,
    "cards": cards_agent,
    "loans": loans_agent,
    "mortgage": mortgage_agent,
    "onboarding": onboarding_agent,
    "faq": faq_agent,
}


def build_graph():
    graph = StateGraph(ACBState)

    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("router", partial(router_node, genai_client=_client))
    for name, fn in DOMAIN_NODES.items():
        graph.add_node(name, partial(fn, genai_client=_client))
    graph.add_node("escalation", escalation_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"blocked": "output_guardrail", "router": "router"},
    )
    graph.add_conditional_edges(
        "router",
        route_to_agent,
        {name: name for name in DOMAIN_NODES},
    )
    for name in DOMAIN_NODES:
        graph.add_conditional_edges(
            name,
            route_after_agent,
            {"escalation": "escalation", "output_guardrail": "output_guardrail"},
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
        {"message": message, "session_id": session_id, "language": language},
        config=config,
    )

    return {
        "reply": result.get("final_reply", ""),
        "session_id": session_id,
        "language": language,
        "route": result.get("route"),
        "escalated": bool(result.get("escalate")),
    }
