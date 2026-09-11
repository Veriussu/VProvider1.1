# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Paneli API Testleri
#  Dosya:    tests/test_admin_api.py
#  Amaç:     Kurulum, giriş/çıkış, oturum koruması, model yönetimi ve
#            isimlendirilmiş API anahtarı akışlarını sahte motor + geçici
#            veritabanıyla doğrular.
#  Mekanik:  - user_store/auth singleton'ları geçici örneklere yönlendirilir.
#            - HuggingFace araması ve indirme sahte fonksiyonlarla yamalanır.
# ─────────────────────────────────────────────────────────────

import asyncio

from fastapi.testclient import TestClient

from app import admin_api, auth, hf_downloader
from app import main as main_mod
from app import openai_api as openai_api_mod
from app.model_manager import ModelManager
from app.user_store import UserStore

TEST_API_KEY = "panel-test-anahtar"


class FakeEngine:
    """Panel model listesi testleri için sahte motor."""

    def __init__(self, path):
        self._loaded = False
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run_chat(self, messages, **params):
        user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
        return "yanıt: " + user

    def stream_chat(self, messages, **params):
        text = self.run_chat(messages, **params)
        for ch in text:
            yield ch

    def usage_info(self):
        return dict(self.last_usage)


def _make_client(tmp_path, monkeypatch, with_user=True):
    """Ortak test kurulumu: geçici DB + sahte model + oturumlu müşteri."""
    store = UserStore(tmp_path / "panel.db")
    store.init()
    if with_user:
        store.mark_setup_done()
        store.set_api_key(TEST_API_KEY)
        _create_user(store)

    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    monkeypatch.setattr(admin_api, "get_store", lambda: store)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "ornek-model.gguf").write_bytes(b"data")

    manager = ModelManager(models_dir=models_dir, engine_factory=lambda i: FakeEngine(str(i.path)), memory_mode="keep")
    monkeypatch.setattr(openai_api_mod, "get_manager", lambda: manager)
    monkeypatch.setattr(admin_api, "get_manager", lambda: manager)

    client = TestClient(main_mod.app)
    return client, store


def _create_user(store):
    from app.auth import hash_password

    store.create_user("yonetici", hash_password("sifre-1234"))


def _login(client):
    resp = client.post("/panel/login", json={"username": "yonetici", "password": "sifre-1234"})
    assert resp.status_code == 200
    return resp


# ------------------------------------------------------------------
# Kurulum ve giriş
# ------------------------------------------------------------------

def test_setup_creates_user_and_cookie(tmp_path, monkeypatch):
    """İlk kurulum kullanıcıyı oluşturur, oturumu çereze yazar, API anahtarı üretir."""
    client, store = _make_client(tmp_path, monkeypatch, with_user=False)
    resp = client.post("/panel/setup", json={
        "username": "admin", "password": "guclu-sifre-1",
    })
    assert resp.status_code == 200
    assert store.is_setup_done()
    keys = store.list_api_keys()
    assert keys and keys[0]["name"] == "Varsayılan" and keys[0]["key"]  # varsayılan anahtar üretildi
    assert store.get_user_by_username("admin") is not None
    assert "vprovider_session" in client.cookies
    info = store.get_site_info()
    assert info["project_name"] == "VProvider"
    assert info["github_url"] == "https://github.com/Veriussu/"


def test_setup_rejected_when_done(tmp_path, monkeypatch):
    """Kurulum bir kez yapıldıysa ikinci deneme reddedilir."""
    client, _ = _make_client(tmp_path, monkeypatch)
    resp = client.post("/panel/setup", json={"username": "x", "password": "y"})
    assert resp.status_code == 400


def test_setup_weak_password_rejected(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch, with_user=False)
    resp = client.post("/panel/setup", json={"username": "admin", "password": "kisa"})
    assert resp.status_code == 422


def test_login_wrong_password(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    resp = client.post("/panel/login", json={"username": "yonetici", "password": "yanlis"})
    assert resp.status_code == 401


def test_login_and_logout_flow(tmp_path, monkeypatch):
    client, store = _make_client(tmp_path, monkeypatch)
    login = _login(client)
    assert "vprovider_session" in client.cookies
    me = client.get("/panel/me")
    assert me.status_code == 200
    assert me.json()["user"] == "yonetici"
    assert me.json()["setup_done"] is True

    logout = client.post("/panel/logout")
    assert logout.status_code == 200
    assert client.get("/panel/me").status_code == 401


# ------------------------------------------------------------------
# Oturum koruması
# ------------------------------------------------------------------

def test_panel_requires_session(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    assert client.get("/panel/models").status_code == 401
    assert client.get("/panel/apis").status_code == 401
    assert client.post("/panel/models/ornek-model/load").status_code == 401


def test_panel_status_public(tmp_path, monkeypatch):
    """Durum şifresiz okunabilir (panel açılışı için)."""
    client, _ = _make_client(tmp_path, monkeypatch)
    body = client.get("/panel/status").json()
    assert body["needs_setup"] is False
    assert body["setup_done"] is True
    assert body["app"] == "VProvider"
    assert body["site"]["project_name"] == "VProvider"


def test_panel_status_needs_setup_when_no_user(tmp_path, monkeypatch):
    """Sistemde kullanıcı yokken kayıt sayfası gösterilmesi gerektiğini bildirir."""
    client, _ = _make_client(tmp_path, monkeypatch, with_user=False)
    assert client.get("/panel/status").json()["needs_setup"] is True


def test_login_rejected_when_no_user(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch, with_user=False)
    resp = client.post("/panel/login", json={"username": "x", "password": "şifre-1234"})
    assert resp.status_code == 400
    assert "kayıtlı kullanıcı" in resp.json()["detail"]


def test_register_rejected_when_user_exists(tmp_path, monkeypatch):
    """Kullanıcı varken kayıt sayfası asla açılmamalı; /setup ikinci kez reddedilmeli."""
    client, _ = _make_client(tmp_path, monkeypatch)
    resp = client.post("/panel/setup", json={"username": "ikinci", "password": "sifre-1234"})
    assert resp.status_code == 400
    assert "kayıtlı" in resp.json()["detail"]


# ------------------------------------------------------------------
# Model yönetimi
# ------------------------------------------------------------------

def test_models_list_and_actions(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)

    models = client.get("/panel/models").json()["models"]
    model_ids = [m["id"] for m in models]
    assert "ornek-model" in model_ids
    item = [m for m in models if m["id"] == "ornek-model"][0]
    assert item["loaded"] is False

    r = client.post("/panel/models/ornek-model/load")
    assert r.status_code == 200
    assert [m for m in client.get("/panel/models").json()["models"] if m["id"] == "ornek-model"][0]["loaded"] is True

    r = client.post("/panel/models/ornek-model/mode", json={"memory_mode": "dynamic"})
    assert r.status_code == 200
    assert [m for m in client.get("/panel/models").json()["models"] if m["id"] == "ornek-model"][0]["memory_mode"] == "dynamic"


def test_unknown_model_404(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)
    assert client.post("/panel/models/yok-model/load").status_code == 404


def test_delete_model(tmp_path, monkeypatch):
    client, store = _make_client(tmp_path, monkeypatch)
    _login(client)
    resp = client.post("/panel/models/ornek-model/delete")
    assert resp.status_code == 200
    assert resp.json()["removed"] == ["ornek-model.gguf"]
    assert client.post("/panel/models/ornek-model/load").status_code == 404


# ------------------------------------------------------------------
# HuggingFace arama + indirme
# ------------------------------------------------------------------

def test_browse_models(tmp_path, monkeypatch):
    """Katalog (browse) uç noktası geniş GGUF listesini döner; oturumsuz kullanılamaz."""
    client, _ = _make_client(tmp_path, monkeypatch)

    anon = client.get("/panel/models/browse")
    assert anon.status_code == 401

    _login(client)
    monkeypatch.setattr(hf_downloader, "browse_models", lambda: [
        {"repo_id": "org/buyuk", "downloads": 999, "likes": 3,
         "last_modified": "", "gguf_count": 6, "pipeline": "text-generation"},
        {"repo_id": "org/kucuk", "downloads": 5, "likes": 1,
         "last_modified": "", "gguf_count": 1, "pipeline": "text2text-generation"},
    ])

    r = client.get("/panel/models/browse")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 2
    assert models[0]["repo_id"] == "org/buyuk"
    assert models[0]["pipeline"] == "text-generation"


def test_search_and_download_flow(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)

    fake_files = [{"filename": "kucuk-Q4_K_M.gguf", "size_bytes": 1234}]

    class RFile:
        def __init__(self, filename, size_bytes):
            self.filename = filename
            self.size_bytes = size_bytes

    async def fake_start(repo, fname, revision="main", models_dir=None, token=None, progress_cb=None):
        return type("R", (), {"path": None})()

    monkeypatch.setattr(hf_downloader, "search_models", lambda q, cat, lim: [
        type("R", (), {"repo_id": "org/kucuk", "downloads": 5, "likes": 1,
                        "last_modified": "", "gguf_count": 1})()
    ])
    monkeypatch.setattr(hf_downloader, "get_repo_files", lambda repo: [RFile("kucuk-Q4_K_M.gguf", 1234)])
    monkeypatch.setattr(hf_downloader, "start_download", fake_start)
    monkeypatch.setattr(hf_downloader, "get_download_status", lambda repo: {
        "received": 1234, "total": 1234, "status": "tamam", "error": ""})
    monkeypatch.setattr(
        hf_downloader, "list_active_downloads",
        lambda: {"org/kucuk": {"received": 1234, "total": 1234, "status": "tamam", "error": ""}})

    s = client.get("/panel/models/search", params={"q": "kucuk"})
    assert s.status_code == 200
    assert s.json()["models"][0]["repo_id"] == "org/kucuk"

    f = client.get("/panel/models/repo-files", params={"repo": "org/kucuk"})
    assert f.status_code == 200
    assert f.json()["files"][0]["filename"] == "kucuk-Q4_K_M.gguf"

    d = client.post("/panel/models/download", json={"repo_id": "org/kucuk", "filename": "kucuk-Q4_K_M.gguf"})
    assert d.status_code == 200

    status = client.get("/panel/models/download/status", params={"repo_id": "org/kucuk"})
    assert status.json()["status"] == "tamam"

    # Çoklu indirme görünürlüğü: tüm (aktif+biten) kayıtlar listelenir
    all_dl = client.get("/panel/models/downloads")
    assert all_dl.status_code == 200
    assert all_dl.json()["org/kucuk"]["status"] == "tamam"


def test_cancel_download_endpoint(tmp_path, monkeypatch):
    """İptal uç noktası: oturumsuz red, aktif indirme yoksa 404, etkinse ok."""
    client, _ = _make_client(tmp_path, monkeypatch)

    anon = client.post("/panel/models/download/cancel", json={"repo_id": "org/x", "filename": "a.gguf"})
    assert anon.status_code == 401

    _login(client)

    # Olmayan/bitmiş indirme için iptal -> 404
    r = client.post("/panel/models/download/cancel", json={"repo_id": "org/x", "filename": "a.gguf"})
    assert r.status_code == 404

    # Aktif indirme varken iptal -> 200
    hf_downloader._active.add("org/x")
    try:
        r2 = client.post("/panel/models/download/cancel", json={"repo_id": "org/x", "filename": "a.gguf"})
        assert r2.status_code == 200
        assert r2.json()["ok"] is True
    finally:
        hf_downloader._active.discard("org/x")
        hf_downloader._cancelled.discard("org/x")


# ------------------------------------------------------------------
# API anahtarları (isimlendirilmiş, birden çok)
# ------------------------------------------------------------------

def test_apis_list_create_delete_flow(tmp_path, monkeypatch):
    client, store = _make_client(tmp_path, monkeypatch)
    _login(client)

    # Başlangıçta isimlendirilmiş anahtar yok (legacy ayrı tutulur)
    assert client.get("/panel/apis").json()["keys"] == []

    resp = client.post("/panel/apis", json={"name": "Chatbox"})
    assert resp.status_code == 200
    key1 = resp.json()["key"]
    assert key1["name"] == "Chatbox"
    assert len(key1["key"]) == 64

    resp = client.post("/panel/apis", json={"name": "Sunucu-1"})
    assert resp.status_code == 200
    key2 = resp.json()["key"]

    keys = client.get("/panel/apis").json()["keys"]
    assert [k["name"] for k in keys] == ["Chatbox", "Sunucu-1"]
    assert store.get_api_key_by_id(key1["id"])["key"] == key1["key"]

    # Silme: ilk anahtar gider, diğeri kalır
    resp = client.delete(f"/panel/apis/{key1['id']}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == key1["id"]
    names = [k["name"] for k in client.get("/panel/apis").json()["keys"]]
    assert names == ["Sunucu-1"]
    assert store.get_api_key_by_id(key1["id"]) is None

    # Olmayan anahtarı silmek -> 404
    assert client.delete("/panel/apis/9999").status_code == 404


def test_create_api_key_validation(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)
    assert client.post("/panel/apis", json={"name": "   "}).status_code == 400
    assert client.post("/panel/apis", json={"name": "x" * 51}).status_code == 400


def test_named_api_key_accepted_for_v1(tmp_path, monkeypatch):
    """Oluşturulan isimlendirilmiş anahtar /v1/* uç noktalarında kabul edilir."""
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)
    created = client.post("/panel/apis", json={"name": "Test"}).json()["key"]

    ok = client.get("/v1/models", headers={"Authorization": "Bearer " + created["key"]})
    assert ok.status_code == 200

    # Anahtar silinince o anahtara gelen istekler reddedilir
    client.delete(f"/panel/apis/{created['id']}")
    bad = client.get("/v1/models", headers={"Authorization": "Bearer " + created["key"]})
    assert bad.status_code == 401


def test_delete_last_key_clears_legacy(tmp_path, monkeypatch):
    """Son isimlendirilmiş anahtar silinince miras settings.api_key de temizlenir."""
    client, store = _make_client(tmp_path, monkeypatch)
    _login(client)
    assert store.get_api_key() == TEST_API_KEY  # miras anahtar mevcut

    created = client.post("/panel/apis", json={"name": "Tek"}).json()["key"]
    client.delete(f"/panel/apis/{created['id']}")

    assert store.get_api_key() == ""  # miras temizlendi
    assert not client.get("/panel/apis").json()["keys"]
    assert client.get(
        "/v1/models", headers={"Authorization": "Bearer " + created["key"]}
    ).status_code == 401


# ------------------------------------------------------------------
# Panel sohbet (modeli arayüzden test etme)
# ------------------------------------------------------------------

def test_panel_chat_without_login_requires_session(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    resp = client.post("/panel/chat", json={"model": "ornek-model", "messages": [{"role": "user", "content": "merhaba"}]})
    assert resp.status_code == 401


def test_panel_chat_gets_answer_and_usage(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)

    resp = client.post("/panel/chat", json={
        "model": "ornek-model",
        "messages": [
            {"role": "user", "content": "merhaba dünya"},
        ],
        "temperature": 0.3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["model"] == "ornek-model"
    assert body["message"] == "yanıt: merhaba dünya"
    assert body["usage"]["total_tokens"] == 8


def test_panel_chat_invalid_role_rejected(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)

    resp = client.post("/panel/chat", json={
        "model": "ornek-model",
        "messages": [{"role": "moderatör", "content": "x"}],
    })
    assert resp.status_code == 400


def test_panel_chat_unknown_model_400(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)

    resp = client.post("/panel/chat", json={
        "model": "yok-model",
        "messages": [{"role": "user", "content": "merhaba"}],
    })
    assert resp.status_code == 400
    assert "bulunamadı" in resp.json()["detail"]


def test_panel_chat_full_history_sent(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    _login(client)
    messages = [
        {"role": "user", "content": "ilk soru"},
        {"role": "assistant", "content": "ilk yanıt"},
        {"role": "user", "content": "ikinci soru"},
    ]
    resp = client.post("/panel/chat", json={"model": "ornek-model", "messages": messages})
    assert resp.status_code == 200
    assert resp.json()["message"] == "yanıt: ikinci soru"


# ------------------------------------------------------------------
# Panel ön yüz (kök dizin)
# ------------------------------------------------------------------

def test_index_served(tmp_path, monkeypatch):
    client, _ = _make_client(tmp_path, monkeypatch)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()