# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 13 - Ses Üretim Motoru Testleri
#  Dosya:    tests/test_tts_backend.py
#  Amaç:     tts_backend.synthesize + depo akışını sahte edge-tts ile
#            doğrular; gerçek Microsoft servisine istek atılmaz.
#  Mekanik:  - edge_tts modülü sys.modules'a sahte nesneyle enjekte edilir.
#            - Modül kapalı / boş metin / uzun metin / bilinmeyen motor
#              hata senaryoları ayrı ayrı denetlenir.
# ─────────────────────────────────────────────────────────────

import asyncio
import sys

import pytest

from app import tts_backend as tts
from app.tts_backend import TTSError


class FakeAudioChunk:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return {"type": "audio", "data": self._data}[key]


class FakeCommunicate:
    def __init__(self, text, voice, rate=""):
        self.text = text
        self.voice = voice
        self.rate = rate

    async def stream(self):
        yield {"type": "audio", "data": b"ID3-sahte-mp3-1"}
        yield {"type": "word-boundary", "text": "x", "offset": 0, "duration": 1}
        yield {"type": "audio", "data": b"-2"}


class FakeEdgeModule:
    Communicate = FakeCommunicate


@pytest.fixture()
def fake_edge(monkeypatch):
    """edge_tts modülünü sahte sürümle değiştirir."""
    monkeypatch.setitem(sys.modules, "edge_tts", FakeEdgeModule)
    monkeypatch.setattr(tts.settings, "tts_enabled", True)


def test_synthesize_edge_returns_bytes(fake_edge):
    """Sahte edge-tts ile konuşma baytları üretilir."""
    data, mime = asyncio.run(tts.synthesize("merhaba dünya", voice="tr-TR-EmelNeural"))
    assert data == b"ID3-sahte-mp3-1-2"
    assert mime == "audio/mpeg"


def test_synthesize_disabled_raises():
    """Modül kapalıyken anlaşılır hata verilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba"))
    assert "kapalı" in str(exc.value)


def test_synthesize_empty_text_raises(fake_edge):
    """Boş metin reddedilir."""
    with pytest.raises(TTSError):
        asyncio.run(tts.synthesize("   "))


def test_synthesize_too_long_raises(fake_edge):
    """Karakter üst sınırı aşılınca reddedilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("a" * (tts.MAX_TEXT_CHARS + 1)))
    assert "çok uzun" in str(exc.value)


def test_synthesize_unknown_engine_raises(fake_edge):
    """Bilinmeyen motor reddedilir."""
    with pytest.raises(TTSError) as exc:
        asyncio.run(tts.synthesize("merhaba", engine="hayalet-motor"))
    assert "hayalet-motor" in str(exc.value)


def test_engine_available_with_fake(fake_edge):
    """Sahte modül varken motor kurulu kabul edilir."""
    assert tts.engine_available() is True


def test_edge_speed_rate():
    """OpenAI speed değeri edge-tts rate biçimine çevrilir."""
    assert tts._edge_speed_rate(1.0) == "+0%"
    assert tts._edge_speed_rate(1.1) == "+10%"
    assert tts._edge_speed_rate(0.85) == "-15%"


def test_audio_store_roundtrip():
    """Depoya yazılan ses okunabilir ve cap uygulanır."""
    tts.store_audio("k1", b"AAA", "audio/mpeg")
    assert tts.get_stored_audio("k1") == {"bytes": b"AAA", "mime": "audio/mpeg"}
    assert tts.get_stored_audio("yok") is None
    # cap: en son üretimler korunur
    for i in range(tts.AUDIO_STORE_CAP + 5):
        tts.store_audio(f"k{i}", b"X", "audio/mpeg")
    assert tts.get_stored_audio("k1") is None  # eski silindi
    assert tts.get_stored_audio(f"k{tts.AUDIO_STORE_CAP + 4}") is not None