import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..agents.graph import run_graph
from ..i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from ..tools import amortize
from .schemas import ChatMessage, LoanCalcRequest, MortgageCalcRequest
from .voice import router as voice_router

load_dotenv()

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "frontend")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(voice_router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Languages (English + French, Spanish, Dutch)
# ---------------------------------------------------------------------------
@app.get("/languages")
def list_languages():
    """Language list for the sidebar globe selector."""
    return {
        "default": DEFAULT_LANGUAGE,
        "languages": [
            {
                "code": code,
                "name": info["name"],
                "native_name": info["native_name"],
                "flag": info["flag"],
            }
            for code, info in SUPPORTED_LANGUAGES.items()
        ],
    }


# ---------------------------------------------------------------------------
# Chat — delegates to the LangGraph agent pipeline
# ---------------------------------------------------------------------------
@app.post("/chat")
def chat(chat_message: ChatMessage):
    session_id = chat_message.session_id or str(uuid.uuid4())
    language = chat_message.language if chat_message.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    result = run_graph(chat_message.message, session_id, language)
    return result


# ---------------------------------------------------------------------------
# Mortgage / loan calculators
# ---------------------------------------------------------------------------
@app.post("/calculate/mortgage")
def calculate_mortgage(payload: MortgageCalcRequest):
    principal = payload.home_price - payload.down_payment
    if principal <= 0:
        raise HTTPException(status_code=400, detail="Down payment must be less than the home price")
    try:
        result = amortize(principal, payload.annual_rate, round(payload.term_years * 12))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result["home_price"] = round(payload.home_price, 2)
    result["down_payment"] = round(payload.down_payment, 2)
    return result


@app.post("/calculate/loan")
def calculate_loan(payload: LoanCalcRequest):
    try:
        result = amortize(payload.loan_amount, payload.annual_rate, round(payload.term_years * 12))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
