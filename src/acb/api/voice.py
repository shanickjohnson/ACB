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
    resp.raise_for_status()
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
    resp.raise_for_status()
    return resp.json().get("text", "")


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
        raise HTTPException(status_code=502, detail="Text-to-speech is unavailable right now")
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
        raise HTTPException(status_code=502, detail="Speech-to-text is unavailable right now")
    return {"text": text}
