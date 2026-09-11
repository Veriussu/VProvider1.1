# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 13 - Ses Üretimi (TTS) Motoru
#  Dosya:    app/tts_backend.py
#  Amaç:     Metinden konuşma (TTS) üretir. Varsayılan motor edge-tts
#            (Microsoft Edge çevrimiçi motoru): hafif, saf Python, MP3
#            üretir; GPU gerekmez. Panel üretimleri in-memory depoda
#            tutulur (çoktan-azan düzen).
#  Not:      Modül kapalıysa ya da metin çok uzunsa TTSError verilir;
#            API katmanı bunu OpenAI uyumlu hataya çevirir.
#  Kullanım: rez = await synthesize("merhaba", voice="tr-TR-EmelNeural")
#            rez -> (ses_baytları, mime)
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import threading
import time
from typing import Optional

from app.config import settings

logger = logging.getLogger("vprovider")

MAX_TEXT_CHARS = 3000          # edge-tts üst sınırı (~3000 karakter)
TTS_TIMEOUT = 120              # tek konuşma için üst süre (saniye)
AUDIO_STORE_CAP = 20           # depoda tutulacak son üretim sayısı

# Panel seçiminde sunulan Türkçe sesler (edge-tts'te en yaygın kullanılanlar)
TR_VOICES: dict[str, str] = {
    "tr-TR-EmelNeural": "Emel (kadın)",
    "tr-TR-AhmetNeural": "Ahmet (erkek)",
}


class TTSError(RuntimeError):
    """Ses üretim hataları (kapalı modül, kurulu olmayan motor, uzun metin...)."""


# ------------------------------------------------------------------
# Ses deposu (in-memory): panel arayüzünce sunulan son üretimler
# ------------------------------------------------------------------

_audio_store: dict[str, dict] = {}
_audio_store_lock = threading.Lock()


def store_audio(key: str, audio: bytes, mime: str) -> None:
    """Üretilen sesi çoktan-azan düzen (cap) ile saklar."""
    with _audio_store_lock:
        _audio_store[key] = {"bytes": audio, "mime": mime}
        while len(_audio_store) > AUDIO_STORE_CAP:
            _audio_store.pop(next(iter(_audio_store)))


def get_stored_audio(key: str) -> Optional[dict]:
    """Depodan ses baytlarını döner (yoksa None)."""
    with _audio_store_lock:
        return _audio_store.get(key)


# ------------------------------------------------------------------
# Motor
# ------------------------------------------------------------------

def _edge_communicate(text: str, voice: str, rate: str):
    """edge-tts.Communicate örneği üretir.

    Ayrı fonksiyona çıkarıldı ki testler sahte modülle yamalayabilsin.
    """
    import edge_tts  # işlev anında yüklenir; bağımlılık isteğe bağlıdır

    return edge_tts.Communicate(text, voice, rate=rate)


def _import_edge():
    import edge_tts  # noqa: F401

    return edge_tts


async def _synthesize_edge(text: str, voice: str, rate: str) -> tuple[bytes, str]:
    """edge-tts ile konuşmayı üretir -> (mp3_baytları, "audio/mpeg")."""
    try:
        communicate = await asyncio.to_thread(_edge_communicate, text, voice, rate)
    except ImportError:
        raise TTSError(
            "edge-tts motoru kurulu değil. Kurun: .venv/bin/pip install edge-tts "
            "(veya requirements.txt'ten)."
        )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    if buf.tell() == 0:
        raise TTSError("edge-tts ses üretemedi (ağ/geçersiz ses adı).")
    return buf.getvalue(), "audio/mpeg"


async def synthesize(
    text: str,
    voice: str = "",
    engine: str = "",
    rate: str = "",
) -> tuple[bytes, str]:
    """Metni seslendirir.

    Dönüş: (ses_baytları, mime). Yanlış/kapalı kurulumda TTSError yükseltir.
    """
    if not settings.tts_enabled:
        raise TTSError("Ses üretim modülü kapalı (TTS_ENABLED=false).")

    text = text.strip()
    if not text:
        raise TTSError("Konuşulacak metin boş olamaz.")
    if len(text) > MAX_TEXT_CHARS:
        raise TTSError(f"Metin çok uzun: en fazla {MAX_TEXT_CHARS} karakter olabilir.")

    engine = engine or settings.tts_engine
    if engine == "edge":
        voice = voice or settings.tts_voice
        rate = rate or settings.tts_voice_rate
        return await _synthesize_edge(text, voice, rate)

    raise TTSError(f"Bilinmeyen ses motoru: {engine} (kullanılabilir: edge).")


def engine_available() -> bool:
    """Seçili motorun bellekte yüklü olup olmadığını döner (kurulum denetimi)."""
    try:
        _import_edge()
        return True
    except ImportError:
        return False


def _edge_speed_rate(speed: float) -> str:
    """OpenAI speed değerini (1.0 = normal) edge-tts 'rate' biçimine çevirir."""
    pct = int(round((speed - 1.0) * 100))
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct}%"


async def tts_status() -> dict:
    """Panel için durum bilgisi: açık mı, motor kurulu mu, hangi ses kullanılıyor."""
    return {
        "enabled": settings.tts_enabled,
        "engine": settings.tts_engine,
        "voice": settings.tts_voice,
        "engine_installed": engine_available() if settings.tts_enabled else None,
        "rate": settings.tts_voice_rate,
    }