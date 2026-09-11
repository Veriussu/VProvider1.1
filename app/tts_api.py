# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 13 - OpenAI Uyumlu Ses (TTS) API'si
#  Dosya:    app/tts_api.py
#  Amaç:     OpenAI /v1/audio/speech ile uyumlu metin-sesleme uçları sunar.
#            OpenAI bu isteklerde sesi ham bayt (audio/mpeg) döner; aynı
#            davranış korunur. kimlik doğrulama (Bearer) ve hata yapısı
#            (detail.error) tüm /v1/* uçlarıyla aynıdır.
#  Mekanik:  - POST /v1/audio/speech  -> {model, input, voice, response_format}
#              ham MP3 döner.
#            - GET  /v1/audio/voices   -> desteklenen Türkçe sesler (panel için)
#  Kullanım: app/main.py içinde app.include_router(tts_router)
# ─────────────────────────────────────────────────────────────

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.routing import APIRouter as _Router
from pydantic import BaseModel, Field

from app import tts_backend
from app.auth import require_api_key
from app.config import settings
from app.openai_api import _openai_error
from app.tts_backend import TTSError, tts_status

router: _Router = APIRouter(prefix="/v1/audio", dependencies=[Depends(require_api_key)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class SpeechRequest(BaseModel):
    """POST /v1/audio/speech istek gövdesi (OpenAI ile aynı alanlar)."""

    model: str = "edge-tts"       # motor adı (edge | edge-tts → edge)
    input: str = Field(..., min_length=1)
    voice: str = ""               # boşsa TTS_VOICE kullanılır
    response_format: str = "mp3"  # mp3 (edge-tts çıktısı)
    speed: float = Field(1.0, ge=0.5, le=2.0)  # 1.0 = normal hız


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.post("/speech")
async def create_speech(req: SpeechRequest):
    """Metni seslendirir; ham ses (audio/mpeg) döner (OpenAI ile aynı)."""
    if req.response_format != "mp3":
        raise _openai_error(
            400,
            "edge-tts yalnızca mp3 üretir; response_format='mp3' kullanın.",
            param="response_format",
        )
    engine = "edge" if "edge" in req.model else req.model
    try:
        audio, mime = await tts_backend.synthesize(
            text=req.input,
            voice=req.voice,
            engine=engine,
            rate=tts_backend._edge_speed_rate(req.speed),
        )
    except TTSError as exc:
        raise _openai_error(400, str(exc), code="speech_failed")
    except Exception as exc:
        raise _openai_error(502, f"Ses motoru bağlantı hatası: {exc}", code="tts_unreachable")

    return Response(content=audio, media_type=mime)


@router.get("/voices")
async def list_voices():
    """Desteklenen Türkçe sesleri listeler (OpenAI'de yok; VProvider eklentisi)."""
    return {
        "object": "list",
        "data": [
            {"id": vid, "label": label, "locale": "tr-TR"}
            for vid, label in tts_backend.TR_VOICES.items()
        ],
    }


@router.get("/status")
async def status_endpoint():
    """Motor durumu (açık mı, kurulu mu, hangi ses) — panel ve teşhis için."""
    return await tts_status()