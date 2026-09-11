# ─────────────────────────────────────────────────────────────
#  Bölüm:    Ses (TTS) API Uç Noktası Testleri
#  Dosya:    tests/test_tts_api.py
#  Amaç:     OpenAI uyumlu /v1/audio/speech ve panel /panel/tts/* uçlarını
#            sahte motorla doğrular (gerçek edge-tts isteği atılmaz).
#  Mekanik:  - API anahtarı zorunluluğu (401) denetlenir.
#            - Modül kapalıyken 400 (speech_failed).
#            - Başarılı üretim ham ses (audio/mpeg) döner.
#            - Panel uçları oturum (session) korumalıdır.
# ─────────────────────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient

from app import admin_api, auth, tts_backend
from app import main as main_mod
from app.config import settings
from app.user_store import UserStore

TEST_API_KEY = "ses-test-anahtar"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API anahtarlı hazır test ortamı."""
    store = UserStore(tmp_path / "tts.db")
    store.init()
    store.set_api_key(TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    return TestClient(main_mod.app)


def _auth():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


def _enable_tts(monkeypatch):
    monkeypatch.setattr(settings, "tts_enabled", True)


# ------------------------------------------------------------------
# OpenAI uyumlu /v1/audio/*
# ------------------------------------------------------------------

def test_speech_requires_api_key(client):
    """API anahtarı olmadan ses uçlarına erişilemez (401)."""
    assert client.post("/v1/audio/speech", json={"input": "merhaba"}).status_code == 401
    assert client.get("/v1/audio/voices").status_code == 401


def test_speech_disabled_returns_400(client):
    """TTS_ENABLED=false iken 400 (speech_failed) döner."""
    r = client.post("/v1/audio/speech", json={"input": "merhaba"}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "speech_failed"
    assert "kapalı" in r.json()["error"]["message"]


def test_speech_rejects_non_mp3_format(client):
    """Yalnızca mp3 desteklenir (edge-tts çıktısı)."""
    _enable_tts(monkeypatch := pytest.MonkeyPatch())
    r = client.post("/v1/audio/speech",
                    json={"input": "merhaba", "response_format": "wav"},
                    headers=_auth())
    monkeypatch.undo()
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "response_format"


def test_speech_success_returns_audio_bytes(client, monkeypatch):
    """Başarılı üretim ham ses döner (OpenAI davranışı)."""
    _enable_tts(monkeypatch)

    async def fake_synth(**kwargs):
        assert kwargs["engine"] == "edge"
        return b"ID3-gercek-mp3-baytlari", "audio/mpeg"

    monkeypatch.setattr(tts_backend, "synthesize", fake_synth)

    r = client.post("/v1/audio/speech",
                    json={"input": "Merhaba dünya", "voice": "tr-TR-EmelNeural"},
                    headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/mpeg")
    assert r.content == b"ID3-gercek-mp3-baytlari"


def test_voices_endpoint(client):
    """Ses listesi Türkçe sesleri döner."""
    r = client.get("/v1/audio/voices", headers=_auth())
    assert r.status_code == 200
    ids = [v["id"] for v in r.json()["data"]]
    assert "tr-TR-EmelNeural" in ids and "tr-TR-AhmetNeural" in ids


def test_audio_status_endpoint(client):
    """Durum ucu motor bilgisi döner."""
    r = client.get("/v1/audio/status", headers=_auth())
    assert r.status_code == 200
    assert r.json()["enabled"] is False
    assert r.json()["engine"] == "edge"


# ------------------------------------------------------------------
# Panel uçları (/panel/tts/*) — oturum korumalı
# ------------------------------------------------------------------

@pytest.fixture()
def panel(tmp_path, monkeypatch):
    """Kullanıcı + giriş içeren panel müşterisi."""
    store = UserStore(tmp_path / "tts-panel.db")
    store.init()
    store.mark_setup_done()
    store.set_api_key(TEST_API_KEY)
    from app.auth import hash_password

    store.create_user("yonetici", hash_password("sifre-1234"))
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    monkeypatch.setattr(admin_api, "get_store", lambda: store)
    c = TestClient(main_mod.app)
    c.post("/panel/login", json={"username": "yonetici", "password": "sifre-1234"})
    return c


def test_panel_tts_requires_session(client):
    """Girişsiz panel TTS uçlarına erişilemez (401)."""
    assert client.post("/panel/tts/generate", json={"text": "merhaba"}).status_code == 401
    assert client.get("/panel/tts/audio/x").status_code == 401


def test_panel_tts_generate_and_play(panel, monkeypatch):
    """Girişli panel: ses üretilir, URL ile oynatılır."""
    _enable_tts(monkeypatch)

    async def fake_synth(**kwargs):
        return b"PANEL-MP3", "audio/mpeg"

    monkeypatch.setattr(tts_backend, "synthesize", fake_synth)

    r = panel.post("/panel/tts/generate", json={"text": "merhaba panel"})
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    assert r.json()["mime_type"] == "audio/mpeg"

    audio = panel.get(url)
    assert audio.status_code == 200
    assert audio.content == b"PANEL-MP3"


def test_panel_tts_disabled_error(panel, monkeypatch):
    """Modül kapalıyken panel üretimi 400 verir."""
    monkeypatch.setattr(settings, "tts_enabled", False)
    r = panel.post("/panel/tts/generate", json={"text": "merhaba"})
    assert r.status_code == 400
    assert "kapalı" in r.json()["detail"]