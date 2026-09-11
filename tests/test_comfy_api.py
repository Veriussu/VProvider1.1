# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 11 - ComfyUI API Uç Noktası Testleri
#  Dosya:    tests/test_comfy_api.py
#  Amaç:     OpenAI uyumlu /v1/images/* ve panel /panel/comfy/* uçlarını
#            sahte istemciyle doğrular (gerçek ComfyUI gerekmez).
#  Mekanik:  - API anahtarı zorunluluğu (auth) doğrulanır.
#            - Modül kapalıyken 400, checkpoint eksikken 400, üretim
#              hatasında 500 döner.
#            - Başarılı üretim url ve b64_json modlarında test edilir.
#            - Panel uçları oturum (session) korumalıdır.
# ─────────────────────────────────────────────────────────────

import base64

import pytest
from fastapi.testclient import TestClient

from app import admin_api, auth, comfy_api, comfy_client
from app import main as main_mod
from app.comfy_client import ComfyUnavailableError
from app.config import settings
from app.user_store import UserStore

TEST_API_KEY = "gorsel-test-anahtar"


class FakeComfyClient:
    """get_client() yerine geçen sahte istemci."""

    def checkpoints(self):
        return ["v1-5.safetensors", "sdxl.safetensors"]

    def status(self):
        return {"ok": True, "gpu": "NVIDIA GeForce RTX 4060", "vram_mb": 8192}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API anahtarlı + sahte model içeren hazır test ortamı."""
    store = UserStore(tmp_path / "comfy.db")
    store.init()
    store.set_api_key(TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    return TestClient(main_mod.app)


def _enable_comfy(monkeypatch, default_ckpt="v1-5.safetensors"):
    monkeypatch.setattr(settings, "comfyui_enabled", True)
    monkeypatch.setattr(settings, "comfyui_default_checkpoint", default_ckpt)


def _auth():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


# ------------------------------------------------------------------
# OpenAI uyumlu /v1/images/*
# ------------------------------------------------------------------

def test_images_requires_api_key(client):
    """API anahtarı olmadan görsel uçlarına erişilemez (401)."""
    assert client.get("/v1/images/comfy/checkpoints").status_code == 401
    assert client.post("/v1/images/generations", json={"prompt": "kedi"}).status_code == 401


def test_generation_disabled_returns_400(client):
    """COMFYUI_ENABLED=false iken anlaşılır 400 döner."""
    r = client.post("/v1/images/generations", json={"prompt": "kedi"}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "comfy_disabled"


def test_generation_url_mode(client, monkeypatch):
    """Url modunda /v1/images/file/{id}/{i} adresine işaret edilir ve sunulur."""
    _enable_comfy(monkeypatch)

    async def fake_gen(**kwargs):
        assert kwargs["checkpoint"] == "v1-5.safetensors"
        return {"prompt_id": "pid-a", "images": [{"bytes": b"PKL data", "mime": "image/png"}]}

    monkeypatch.setattr(comfy_client, "generate_image", fake_gen)
    monkeypatch.setattr(comfy_client, "get_stored_image",
                        lambda pid, i: {"bytes": b"PKL data", "mime": "image/png"})

    r = client.post("/v1/images/generations",
                    json={"prompt": "kedi", "model": "v1-5.safetensors", "size": "512x512"},
                    headers=_auth())
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data[0]["url"] == "/v1/images/file/pid-a/0"

    img = client.get("/v1/images/file/pid-a/0", headers=_auth())
    assert img.status_code == 200
    assert img.content == b"PKL data"


def test_generation_b64_mode(client, monkeypatch):
    """b64_json modunda base64 kodlu görsel döner."""
    _enable_comfy(monkeypatch)

    async def fake_gen(**kwargs):
        return {"prompt_id": "pid-b", "images": [{"bytes": b"AB", "mime": "image/png"}]}

    monkeypatch.setattr(comfy_client, "generate_image", fake_gen)

    r = client.post("/v1/images/generations",
                    json={"prompt": "kedi", "response_format": "b64_json",
                          "model": "v1-5.safetensors"},
                    headers=_auth())
    assert r.status_code == 200
    assert r.json()["data"][0]["b64_json"] == base64.b64encode(b"AB").decode("ascii")


def test_generation_missing_checkpoint_400(client, monkeypatch):
    """Checkpoint ne istekte ne varsayılanda varsa 400 döner."""
    _enable_comfy(monkeypatch, default_ckpt="")
    r = client.post("/v1/images/generations", json={"prompt": "kedi"}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "checkpoint_required"


def test_generation_comfy_error_maps_500(client, monkeypatch):
    """Üretim hatası OpenAI biçiminde 500 ile döner."""
    _enable_comfy(monkeypatch)

    async def fake_gen(**kwargs):
        from app.comfy_client import ComfyGenerationError
        raise ComfyGenerationError("CUDA out of memory")

    monkeypatch.setattr(comfy_client, "generate_image", fake_gen)

    r = client.post("/v1/images/generations", json={"prompt": "kedi"}, headers=_auth())
    assert r.status_code == 500
    assert "CUDA" in r.json()["error"]["message"]


def test_checkpoints_endpoint(client, monkeypatch):
    """Checkpoint listesi OpenAI biçiminde döner."""
    _enable_comfy(monkeypatch)
    monkeypatch.setattr(comfy_api, "get_client", lambda: FakeComfyClient())
    r = client.get("/v1/images/comfy/checkpoints", headers=_auth())
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["data"]] == ["v1-5.safetensors", "sdxl.safetensors"]


# ------------------------------------------------------------------
# OpenAI uyumlu /v1/videos/* (Faz 12)
# ------------------------------------------------------------------

def test_videos_requires_api_key(client):
    """API anahtarı olmadan video uçlarına erişilemez (401)."""
    assert client.post("/v1/videos/generations", json={"prompt": "bulut"}).status_code == 401


def test_video_generation_disabled_returns_400(client):
    """Modül kapalıyken video üretimi 400 verir."""
    r = client.post("/v1/videos/generations", json={"prompt": "bulut"}, headers=_auth())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "comfy_disabled"


def test_video_generation_url_mode(client, monkeypatch):
    """Url modunda video üretilir ve /v1/videos/file/{id} sunulur."""
    _enable_comfy(monkeypatch)

    async def fake_vid(**kwargs):
        assert kwargs["frames"] == 16
        return {"prompt_id": "pid-v", "video": {
            "bytes": b"GIF89a-video-verisi", "mime": "image/gif",
            "width": 512, "height": 512, "frames": 2,
        }}

    monkeypatch.setattr(comfy_client, "generate_video", fake_vid)
    monkeypatch.setattr(comfy_client, "get_stored_video",
                        lambda pid: {"bytes": b"GIF89a-video-verisi", "mime": "image/gif",
                                     "width": 512, "height": 512, "frames": 2})

    r = client.post("/v1/videos/generations",
                    json={"prompt": "bulut", "model": "v1-5.safetensors",
                          "size": "512x512", "frames": 16},
                    headers=_auth())
    assert r.status_code == 200, r.text
    data = r.json()["data"][0]
    assert data["url"] == "/v1/videos/file/pid-v"
    assert data["mime_type"] == "image/gif"
    assert data["width"] == 512 and data["frames"] == 2

    vid = client.get("/v1/videos/file/pid-v", headers=_auth())
    assert vid.status_code == 200
    assert vid.content == b"GIF89a-video-verisi"


def test_video_generation_frames_validation(client, monkeypatch):
    """Kare sayısı sınır dışıysa pydantic 422 döner."""
    _enable_comfy(monkeypatch)
    r = client.post("/v1/videos/generations", json={"prompt": "bulut", "frames": 999},
                    headers=_auth())
    assert r.status_code == 422


# ------------------------------------------------------------------
# Panel uçları (/panel/comfy/*) — oturum korumalı
# ------------------------------------------------------------------

@pytest.fixture()
def panel(client, tmp_path, monkeypatch):
    """Panel için kullanıcı + giriş ekler."""
    store = UserStore(tmp_path / "comfy-panel.db")
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


def test_panel_comfy_requires_session(client):
    """Girişsiz panel ComfyUI uçlarına erişilemez (401)."""
    assert client.get("/panel/comfy/status").status_code == 401
    assert client.post("/panel/comfy/generate", json={"prompt": "kedi"}).status_code == 401


def test_panel_comfy_status_and_generate(panel, monkeypatch):
    """Girişli panel: durum ve üretim akışı çalışır."""
    _enable_comfy(monkeypatch)
    monkeypatch.setattr(admin_api, "get_client", lambda: FakeComfyClient())

    status = panel.get("/panel/comfy/status")
    assert status.status_code == 200
    assert status.json()["ok"] is True
    assert status.json()["gpu"].startswith("NVIDIA")

    async def fake_gen(**kwargs):
        return {"prompt_id": "pid-p", "images": [{"bytes": b"PNGDATA", "mime": "image/png"}]}

    monkeypatch.setattr(comfy_client, "generate_image", fake_gen)
    monkeypatch.setattr(comfy_client, "get_stored_image",
                        lambda pid, i: {"bytes": b"PNGDATA", "mime": "image/png"})

    r = panel.post("/panel/comfy/generate", json={"prompt": "deniz", "checkpoint": "v1-5.safetensors"})
    assert r.status_code == 200, r.text
    url = r.json()["images"][0]["url"]
    img = panel.get(url)
    assert img.status_code == 200
    assert img.content == b"PNGDATA"


def test_panel_comfy_disabled_error(panel, monkeypatch):
    """Modül kapalıyken panel üretimi 400 ile yanıt verir."""
    monkeypatch.setattr(settings, "comfyui_enabled", False)

    async def fake_gen(**kwargs):
        raise ComfyUnavailableError("ComfyUI köprüsü kapalı")

    monkeypatch.setattr(comfy_client, "generate_image", fake_gen)
    r = panel.post("/panel/comfy/generate", json={"prompt": "kedi"})
    assert r.status_code == 400
    assert "kapalı" in r.json()["detail"]


def test_panel_video_requires_session(client):
    """Girişsiz panel video uçlarına erişilemez (401)."""
    assert client.post("/panel/video/generate", json={"prompt": "kedi"}).status_code == 401
    assert client.get("/panel/comfy/video/x").status_code == 401


def test_panel_video_generate(panel, monkeypatch):
    """Girişli panel: video üretimi GIF URL'siyle döner ve sunulur."""
    _enable_comfy(monkeypatch)

    async def fake_vid(**kwargs):
        return {"prompt_id": "pid-pv", "video": {
            "bytes": b"GIF89a-panel", "mime": "image/gif",
            "width": 512, "height": 512, "frames": 4,
        }}

    monkeypatch.setattr(comfy_client, "generate_video", fake_vid)
    monkeypatch.setattr(comfy_client, "get_stored_video",
                        lambda pid: {"bytes": b"GIF89a-panel", "mime": "image/gif",
                                     "width": 512, "height": 512, "frames": 4})

    r = panel.post("/panel/video/generate", json={"prompt": "uçan kuş", "frames": 16})
    assert r.status_code == 200, r.text
    assert r.json()["frames"] == 4
    vid = panel.get(r.json()["url"])
    assert vid.status_code == 200
    assert vid.content == b"GIF89a-panel"