"""
Tools shared across domain agents: the fixed-rate amortization calculator,
the CSV fast-path lookup, JSON reference-data loading, and the FAQ
retrieval hook into rag.py.
"""

import csv
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

_AMOUNT_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand)?\b", re.IGNORECASE)
_RATE_RE = re.compile(r"([\d.]+)\s*%")
_TERM_RE = re.compile(r"(\d+)\s*[- ]?\s*(?:year|yr)s?\b", re.IGNORECASE)


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


def calculate_loan_from_message(message: str, default_term_years: float) -> dict | None:
    """Deterministic calculate_loan/calculate_mortgage tool: extracts a
    dollar amount, an interest rate, and (optionally) a term directly from
    the customer's own message and runs them through amortize(). Returns
    None if an amount or a rate isn't clearly present, rather than
    guessing — the agent never invents a number the customer didn't supply
    or that wasn't in retrieved reference data.

    Kept as plain regex + arithmetic (no LLM call) rather than a live
    function-calling round trip, since this only needs to parse numbers
    already in the message, not reasoning.
    """
    amount_match = _AMOUNT_RE.search(message)
    rate_match = _RATE_RE.search(message)
    if not amount_match or not rate_match:
        return None

    amount_str, suffix = amount_match.groups()
    try:
        amount = float(amount_str.replace(",", ""))
    except ValueError:
        return None
    if suffix and suffix.lower() in ("k", "thousand"):
        amount *= 1000
    rate = float(rate_match.group(1))

    term_match = _TERM_RE.search(message)
    term_years = float(term_match.group(1)) if term_match else default_term_years

    try:
        result = amortize(amount, rate, round(term_years * 12))
    except ValueError:
        return None
    result["is_estimate"] = True
    return result


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


def retrieve_scoped_context(
    query: str,
    top_k: int = 4,
    jurisdiction: str | None = None,
    chunk_type: str | None = None,
    keywords: list[str] | None = None,
) -> str:
    """Domain-scoped variant of retrieve_faq_context for specialists that
    want grounding context narrowed to their own lane, without touching
    rag.py's retrieval/chunking logic itself:
      - chunk_type: keep only chunks of this type ("fee" | "service" | "web"),
        e.g. onboarding wants "service" chunks (business_services.json).
      - keywords: keep only chunks whose text mentions at least one keyword,
        e.g. payments wants chunks mentioning "transfer"/"wire"/"bill pay".
    Both filters are applied over a wider candidate pool than top_k so
    narrowing doesn't starve the result; if a filter empties the pool
    entirely, falls back to the unfiltered top_k rather than returning no
    context at all — some grounding beats none.
    """
    try:
        from . import rag

        pool = rag.retrieve(query, top_k=max(top_k * 4, 12), jurisdiction=jurisdiction)

        filtered = pool
        if chunk_type:
            filtered = [c for c in filtered if c.get("type") == chunk_type]
        if keywords:
            lowered = [k.lower() for k in keywords]
            filtered = [c for c in filtered if any(k in c["text"].lower() for k in lowered)]

        chunks = filtered[:top_k] if filtered else pool[:top_k]
        return rag.format_context(chunks)
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
