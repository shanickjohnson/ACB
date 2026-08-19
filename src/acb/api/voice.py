"""ElevenLabs voice glue: /tts and /stt endpoints, plus the Markdown
stripping helper so bot replies aren't read aloud with '**'/'-' literally."""

import os
import re

import requests
from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from .schemas import TTSRequest

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ACB assistant voice — this exact voice ID is used for all text-to-speech.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "fx5le4FFKvx12m8z2cAr")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

router = APIRouter()


def strip_markdown_for_speech(text: str) -> str:
    """Removes Markdown syntax so it isn't read aloud literally (e.g. '**', '-')."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\s{0,3}[-*#+>]+\s", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def elevenlabs_tts(text: str, voice_id: str = ELEVENLABS_VOICE_ID) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    resp = requests.post(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30,
    )
    _raise_for_status_with_body(resp)
    return resp.content


def elevenlabs_stt(audio_bytes: bytes, content_type: str) -> str:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    resp = requests.post(
        ELEVENLABS_STT_URL,
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        data={"model_id": "scribe_v1"},
        files={"file": ("recording", audio_bytes, content_type or "audio/webm")},
        timeout=30,
    )
    _raise_for_status_with_body(resp)
    return resp.json().get("text", "")


def _raise_for_status_with_body(resp: requests.Response) -> None:
    """Like resp.raise_for_status(), but keeps ElevenLabs' response body
    (e.g. 'invalid_api_key' vs 'voice_not_found') instead of just the status line."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} — body: {resp.text[:500]}", response=resp) from e


# ElevenLabs status codes that mean "our account/key is the problem", mapped to
# a detail the frontend can show instead of a generic "unavailable" message.
_ELEVENLABS_ACCOUNT_ERRORS = {
    401: "ElevenLabs API key is invalid or expired",
    402: "ElevenLabs account is out of credits",
    429: "ElevenLabs rate limit exceeded, try again shortly",
}


def _upstream_error_detail(e: Exception, fallback: str) -> str:
    if isinstance(e, requests.HTTPError) and e.response is not None:
        return _ELEVENLABS_ACCOUNT_ERRORS.get(e.response.status_code, fallback)
    return fallback


@router.post("/tts")
def tts(payload: TTSRequest):
    """Turns a bot reply into speech using ElevenLabs voice fx5le4FFKvx12m8z2cAr."""
    text = strip_markdown_for_speech(payload.text or "")
    if not text:
        raise HTTPException(status_code=400, detail="No text to speak")
    voice_id = payload.voice_id or ELEVENLABS_VOICE_ID
    try:
        audio_bytes = elevenlabs_tts(text[:2000], voice_id)
    except Exception as e:
        print("ElevenLabs TTS error:", e)
        detail = _upstream_error_detail(e, "Text-to-speech is unavailable right now")
        raise HTTPException(status_code=502, detail=detail)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@router.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """Transcribes a recorded voice clip using ElevenLabs Scribe."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio received")
    try:
        text = elevenlabs_stt(audio_bytes, audio.content_type or "audio/webm")
    except Exception as e:
        print("ElevenLabs STT error:", e)
        detail = _upstream_error_detail(e, "Speech-to-text is unavailable right now")
        raise HTTPException(status_code=502, detail=detail)
    return {"text": text}
