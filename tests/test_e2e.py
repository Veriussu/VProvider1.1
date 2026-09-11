# ─────────────────────────────────────────────────────────────
#  Bölüm:    Uçtan Uca Testler
#  Dosya:    tests/test_e2e.py
#  Amaç:     Gerçek GGUF modeli + gerçek llama.cpp motoruyla tam akışı
#            doğrular: panel oturumu, model listesi, yükleme, chat
#            (akışsız + SSE stream), completion, panel durumu.
#  Mekanik:  - Gerçek models/ klasöründeki en küçük .gguf seçilir
#              (GPU varsa GPU'ya taşınır; VPROVIDER_E2E_GPU_LAYERS=0
#              ile CPU moduna zorlanabilir).
#            - Model bir kez yüklenir, tüm testler aynı bağlamı paylaşır.
#            - Uygulamanın tüm singleton erişimleri geçici örneklere
#              yönlendirilir: gerçek veritabanına dokunulmaz.
#  Kullanım: pytest tests/test_e2e.py -v
#            pytest tests/ -m e2e
# ─────────────────────────────────────────────────────────────

import json
import os

import pytest
from fastapi.testclient import TestClient

from app import admin_api, auth, openai_api
from app import main as main_mod
from app.config import settings
from app.llama_backend import create_engine
from app.model_manager import ModelManager
from app.user_store import UserStore

# Yalnızca gerçek model ve gerçek motor varken anlamlı
E2E_GPU_LAYERS = int(os.environ.get("VPROVIDER_E2E_GPU_LAYERS", "-1"))

API_KEY = "e2e-dogrulama-anahtari"


def _real_models() -> list:
    """models/ klasöründeki gerçek .gguf dosyalarını boyutla birlikte döner."""
    models = list(settings.models_path.rglob("*.gguf"))
    return sorted(models, key=lambda p: p.stat().st_size)


def _has_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _real_models() or not _has_llama_cpp(),
    reason="Gerçek GGUF modeli veya llama-cpp-python yok (önce install.sh kurun)",
)


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """E2E için geçici veritabanı: kullanıcı + API anahtarı + kurulum hazır."""
    store = UserStore(tmp_path_factory.mktemp("e2e") / "e2e.db")
    store.init()
    store.save_site_info("E2E Test", "", "", "", "e2e@ornek.com")
    store.mark_setup_done()
    store.set_api_key(API_KEY)
    from app.auth import hash_password

    store.create_user("e2e", hash_password("e2e-sifre-123"))
    return store


@pytest.fixture(scope="module")
def model_id():
    """Gerçek modeller arasından en küçük olanın kimliği (hız için)."""
    return _real_models()[0].stem


@pytest.fixture(scope="module")
def client(real_store, model_id):
    """Gerçek motorlu uçtan uca istemci: singleton'lar geçici örneklere bağlı."""
    manager = ModelManager(
        models_dir=settings.models_path,
        engine_factory=lambda info: create_engine(
            str(info.path),
            context_size=512,
            gpu_layers=E2E_GPU_LAYERS,
            threads=2,
        ),
        memory_mode="dynamic",
        idle_timeout_minutes=1,
    )

    # Singleton erişimlerini geçici örneklere yönlendir (manuel, modül ömürlü)
    originals = {}
    for module in (auth, main_mod, admin_api):
        originals[(id(module), "get_store")] = module.get_store
        module.get_store = lambda: real_store
    for module in (openai_api, admin_api):
        originals[(id(module), "get_manager")] = module.get_manager
        module.get_manager = lambda: manager

    client = TestClient(main_mod.app)
    login = client.post("/panel/login", json={"username": "e2e", "password": "e2e-sifre-123"})
    assert login.status_code == 200
    yield client

    # Temizlik: modeli bellekten boşalt (gerçek model dosyasına dokunma)
    for info in manager.list_models():
        engine = manager._engines.get(info.model_id)
        if engine is not None:
            try:
                engine.unload()
            except Exception:
                pass

    # Orijinalleri geri yükle (sonraki test dosyalarını kirletme)
    for module in (auth, main_mod, admin_api):
        module.get_store = originals[(id(module), "get_store")]
    for module in (openai_api, admin_api):
        module.get_manager = originals[(id(module), "get_manager")]


# ------------------------------------------------------------------
# 1) Model listesi (panel + OpenAI API)
# ------------------------------------------------------------------

def test_e2e_models_visible_in_panel_and_v1(client, model_id):
    """Diskteki gerçek model hem panelde hem /v1/models'te görünür."""
    panel = client.get("/panel/models").json()
    assert any(m["id"] == model_id for m in panel["models"])

    v1 = client.get("/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    assert v1.status_code == 200
    ids = [m["id"] for m in v1.json()["data"]]
    assert model_id in ids


def test_e2e_v1_requires_valid_key(client):
    """API anahtarı yoksa /v1 erişimi reddedilir."""
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    assert "Bearer" in resp.json()["detail"]


# ------------------------------------------------------------------
# 2) Yükleme + chat (akışsız ve SSE)
# ------------------------------------------------------------------

def test_e2e_chat_completion_non_stream(client, model_id):
    """Gerçek modelle akışsız chat yanıtı ve usage bilgisi üretir."""
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "1 ve 1 kaç eder?"}],
            "max_tokens": 24,
            "temperature": 0.2,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert content.strip()
    assert body["object"] == "chat.completion"
    assert body["usage"]["completion_tokens"] > 0


def test_e2e_chat_stream_uses_single_id(client, model_id):
    """SSE akışı tek sabit id ile parçaları getirir ve [DONE] ile biter."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "2 rakamını üç kelimeyle öv."}],
        "stream": True,
        "max_tokens": 32,
    }
    with client.stream(
        "POST", "/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
    ) as resp:
        assert resp.status_code == 200
        lines = [ln for ln in resp.iter_lines() if ln]
        events = [l for l in lines if l.startswith("data: ")]
        assert events[-1] == "data: [DONE]"
        chunk_ids = {
            json.loads(l[6:])["id"]
            for l in events[:-1]
            if json.loads(l[6:]).get("choices")
        }
        assert len(chunk_ids) == 1, f"Akış boyunca id değişti: {chunk_ids}"
        # En az bir gerçek içerik parçası üretilmeli
        content_parts = ["".join(c.get("delta", {}).get("content", "")) for c in
                         (json.loads(l[6:])["choices"][0] for l in events[:-1])]
        assert any(content_parts)


def test_e2e_completion_non_stream(client, model_id):
    """Gerçek modelle akışsız completion yanıtı üretir."""
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model_id, "prompt": "Merhaba", "max_tokens": 24},
    )
    assert resp.status_code == 200, resp.text
    text = resp.json()["choices"][0]["text"]
    assert text.strip()


# ------------------------------------------------------------------
# 3) Panel akışı: yükle -> durum -> boşalt -> bellek modu
# ------------------------------------------------------------------

def test_e2e_panel_load_status_unload_mode(client, model_id):
    """Panel yükleme/boşaltma ve bellek modu API'leri gerçek modelle çalışır."""
    r = client.post(f"/panel/models/{model_id}/load")
    assert r.status_code == 200 and r.json()["loaded"] is True

    listed = client.get("/panel/models").json()
    entry = next(m for m in listed["models"] if m["id"] == model_id)
    assert entry["loaded"] is True

    r = client.post(f"/panel/models/{model_id}/mode", json={"memory_mode": "keep"})
    assert r.status_code == 200 and r.json()["memory_mode"] == "keep"
    listed = client.get("/panel/models").json()
    entry = next(m for m in listed["models"] if m["id"] == model_id)
    assert entry["memory_mode"] == "keep"

    r = client.post(f"/panel/models/{model_id}/unload")
    assert r.status_code == 200 and r.json()["loaded"] is False
    listed = client.get("/panel/models").json()
    entry = next(m for m in listed["models"] if m["id"] == model_id)
    assert entry["loaded"] is False