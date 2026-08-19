"""
Tools shared across domain agents: the fixed-rate amortization calculator,
the CSV fast-path lookup, JSON reference-data loading, and the FAQ
retrieval hook into rag.py.
"""

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def amortize(principal: float, annual_rate: float, term_months: int) -> dict:
    """Fixed-rate amortization: fixed monthly payment, computed from
    principal, annual interest rate (percent), and term in months."""
    if principal <= 0:
        raise ValueError("Loan amount must be greater than zero")
    if annual_rate < 0:
        raise ValueError("Interest rate can't be negative")
    if term_months <= 0:
        raise ValueError("Term must be greater than zero")

    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        monthly_payment = principal / term_months
    else:
        factor = (1 + monthly_rate) ** term_months
        monthly_payment = principal * monthly_rate * factor / (factor - 1)

    total_payment = monthly_payment * term_months
    return {
        "principal": round(principal, 2),
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_payment - principal, 2),
        "term_months": term_months,
        "annual_rate": annual_rate,
    }


def load_csv_data(filename: str = "qa_data.csv") -> dict:
    """Fast-path Q&A lookup, checked before any LLM call — free and
    deterministic. filename is resolved relative to data/ unless it's
    already an absolute/relative path that exists as given."""
    path = filename if os.path.exists(filename) else os.path.join(DATA_DIR, filename)
    data = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["User_Questions"].lower().strip()] = row["Bot_Response"]
    except FileNotFoundError:
        print(f"Warning: {path} not found. Starting with empty CSV data.")
    return data


CSV_REPLIES = load_csv_data()


def csv_fast_path(message: str) -> str | None:
    return CSV_REPLIES.get(message.lower().strip())


def retrieve_faq_context(query: str, top_k: int = 4) -> str:
    """Retrieves grounding context for the FAQ agent via rag.py's
    retriever, returned as a short string to inject into the prompt."""
    try:
        from . import rag

        results = rag.retrieve(query, top_k=top_k)
        if isinstance(results, list):
            return "\n---\n".join(str(r) for r in results)
        return str(results)
    except Exception as e:
        print("RAG retrieval error:", e)
        return ""


def load_json_reference(path: str) -> dict:
    """Loads a reference JSON file (fees, business services) for a
    domain agent to ground its answer in. Returns {} on failure so a
    missing file degrades gracefully instead of crashing the node.
    path is resolved relative to data/ unless it's already a valid path."""
    resolved = path if os.path.exists(path) else os.path.join(DATA_DIR, path)
    try:
        with open(resolved, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {resolved} not found.")
        return {}
