"""Pydantic request bodies for the FastAPI endpoints."""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    message: str
    session_id: str | None = None  # returned from a previous /chat call to continue that conversation
    language: str = "en"  # selected in the sidebar globe menu (en/fr/es/nl)


class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None  # defaults to ELEVENLABS_VOICE_ID (fx5le4FFKvx12m8z2cAr)


class MortgageCalcRequest(BaseModel):
    home_price: float
    down_payment: float = 0
    annual_rate: float  # percent, e.g. 6.5
    term_years: float = 30


class LoanCalcRequest(BaseModel):
    loan_amount: float
    annual_rate: float  # percent, e.g. 8.5
    term_years: float = 5
