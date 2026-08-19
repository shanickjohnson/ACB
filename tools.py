"""
Tools shared across domain agents. Ported/wrapped from your existing
app.py and rag.py so the agents call the same math and the same
retrieval corpus your monolith already used — no behavior drift.
"""

import csv


def amortize(principal: float, annual_rate: float, term_months: int) -> dict:
    """Fixed-rate amortization — identical to app.py's amortize()."""
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
    """Same fast-path lookup your monolith used before falling back to
    the LLM. Kept here so any domain agent (not just FAQ) can check it
    first — it's free and deterministic."""
    data = {}
    try:
        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["User_Questions"].lower().strip()] = row["Bot_Response"]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Starting with empty CSV data.")
    return data


CSV_REPLIES = load_csv_data()


def csv_fast_path(message: str) -> str | None:
    return CSV_REPLIES.get(message.lower().strip())


def retrieve_faq_context(query: str, top_k: int = 4) -> str:
    """
    Hook into your existing rag.py retriever.

    Replace the body of this function with a call into whatever
    retrieval interface rag.py exposes (e.g. `rag.retrieve(query, top_k)`
    or a vector-store `.similarity_search(query)`), since that file
    wasn't available to read while generating this scaffold. This
    function's contract is: take a query, return a short string of
    retrieved context to inject into the FAQ agent's prompt.
    """
    try:
        import rag  # your existing module

        if hasattr(rag, "retrieve"):
            results = rag.retrieve(query, top_k=top_k)
        elif hasattr(rag, "search"):
            results = rag.search(query, top_k=top_k)
        else:
            raise AttributeError(
                "rag.py has no retrieve()/search() — wire this up to your "
                "actual retriever function name."
            )
        if isinstance(results, list):
            return "\n---\n".join(str(r) for r in results)
        return str(results)
    except Exception as e:
        print("RAG retrieval error:", e)
        return ""


def load_json_reference(path: str) -> dict:
    """Loads a reference JSON file (fees, business services) for a
    domain agent to ground its answer in. Returns {} on failure so a
    missing file degrades gracefully instead of crashing the node."""
    import json

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return {}
