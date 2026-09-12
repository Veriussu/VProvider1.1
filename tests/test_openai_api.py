# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu API Testleri
#  Dosya:    tests/test_openai_api.py
#  Amaç:     /v1/* uç noktalarının OpenAI yapısını (auth, liste,
#            chat, completion, akış/SSE) sahte motorla doğrular.
#  Mekanik:  - Test, gerçek veritabanına dokunmaz: user_store ve
#              model_manager singleton'lari geçici örneklere yönlendirilir.
#            - API anahtarı doğrulaması, geçersiz/eksik anahtar senaryoları
#              da denetlenir.
# ─────────────────────────────────────────────────────────────

import json

import pytest
from fastapi.testclient import TestClient

from app import auth
from app import main as main_mod
from app import openai_api as openai_api_mod
from app.model_manager import ModelManager
from app.user_store import UserStore

TEST_API_KEY = "test-anahtar-123"


class FakeEngine:
    """/v1 endpoint testleri için sahte motor (gerçek llama.cpp gerekmez)."""

    def __init__(self, path):
        self.path = path
        self._loaded = False
        self.last_usage = {}

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def run_chat(self, messages, **params):
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
        return "Merhaba! Ben test modeli."

    def stream_chat(self, messages, **params):
        yield "Mer"
        yield "haba"

    def run_completion(self, prompt, **params):
        self.last_usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        return "tamamlanmış metin"

    def stream_completion(self, prompt, **params):
        yield "tamam"
        yield "lanmış"

    def usage_info(self):
        return self.last_usage


class ToolFakeEngine(FakeEngine):
    """Araç (tool) çağrısı üretebilen sahte motor; /v1/responses testleri."""

    def __init__(self, path, mode="content"):
        super().__init__(path)
        self.mode = mode  # "content" | "toolcall"
        self.last_messages = None
        self.last_tool_calls = None
        self.last_finish_reason = "stop"

    def tool_call_info(self):
        return self.last_tool_calls, self.last_finish_reason

    def run_chat(self, messages, **params):
        self.last_messages = messages
        self.last_usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        if self.mode == "toolcall":
            self.last_tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ]
            self.last_finish_reason = "tool_calls"
            return ""
        self.last_tool_calls = None
        self.last_finish_reason = "stop"
        return "Tamamlandı."

    def stream_chat(self, messages, _yield_events=False, **params):
        self.last_messages = messages
        if self.mode == "toolcall":
            if _yield_events:
                yield {"type": "tool_args", "index": 0, "id": "call_x", "name": "bash", "delta": '{"co'}
                yield {"type": "tool_args", "index": 0, "id": "call_x", "name": "bash", "delta": 'mmand":"ls"}'}
                yield {"type": "finish", "reason": "tool_calls"}
            else:
                yield ""
            return
        if _yield_events:
            yield {"type": "content", "text": "Mer"}
            yield {"type": "content", "text": "haba"}
            yield {"type": "finish", "reason": "stop"}
            yield {"type": "usage", "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}
            return
        yield "Mer"
        yield "haba"


def _kur_client(tmp_path, monkeypatch, mode="content"):
    """API anahtarı + sahte motor içeren ortam kurar; TestClient döner."""
    store = UserStore(tmp_path / "test.db")
    store.init()
    store.create_api_key("Test", TEST_API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "test-model.gguf").write_bytes(b"data")

    manager = ModelManager(
        models_dir=models_dir,
        engine_factory=lambda info: ToolFakeEngine(str(info.path), mode=mode),
        memory_mode="keep",
    )
    monkeypatch.setattr(openai_api_mod, "get_manager", lambda: manager)
    return TestClient(main_mod.app)


@pytest.fixture()
def tool_client(tmp_path, monkeypatch):
    """Araç çağrısı üretebilen sahte motor içeren test ortamı."""
    return _kur_client(tmp_path, monkeypatch, mode="toolcall")


@pytest.fixture()
def content_tool_client(tmp_path, monkeypatch):
    """İçerik üreten sahte motor (Responses testleri için)."""
    return _kur_client(tmp_path, monkeypatch, mode="content")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API anahtarı + sahte model içeren hazır test ortamı."""
    # Geçici veritabanı: API anahtarı tanımla ve anahtarları test ortamına bağla
    store = UserStore(tmp_path / "test.db")
    store.init()
    store.create_api_key("Test", TEST_API_KEY)

    # Tekil örnekleri geçici nesnelere yönlendir (gerçek veriye dokunmaz)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    # main uygulamasının açılışında gerçek veritabanı oluşmasın
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    # Sahte model içeren yönetici
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "test-model.gguf").write_bytes(b"data")
    (models_dir / "diger-model.gguf").write_bytes(b"data")

    manager = ModelManager(
        models_dir=models_dir,
        engine_factory=lambda info: FakeEngine(str(info.path)),
        memory_mode="keep",
    )
    monkeypatch.setattr(openai_api_mod, "get_manager", lambda: manager)

    return TestClient(main_mod.app)


# ------------------------------------------------------------------
# Kimlik doğrulama
# ------------------------------------------------------------------

def test_models_requires_api_key(client):
    """API anahtarı olmadan /v1/models erişilemez (401)."""
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_wrong_api_key_rejected(client):
    """Yanlış API anahtarı reddedilir (401)."""
    resp = client.get("/v1/models", headers={"Authorization": "Bearer yanlis-anahtar"})
    assert resp.status_code == 401


def test_invalid_auth_scheme(client):
    """Bearer dışı yetkilendirme reddedilir (401)."""
    resp = client.get("/v1/models", headers={"Authorization": "Basic xyz"})
    assert resp.status_code == 401


# ------------------------------------------------------------------
# Model listesi
# ------------------------------------------------------------------

def test_list_models(client):
    """Doğru anahtarla model listesi OpenAI biçiminde döner."""
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"test-model", "diger-model"}
    assert all(m["object"] == "model" for m in body["data"])


# ------------------------------------------------------------------
# Chat completions
# ------------------------------------------------------------------

def test_chat_completion_unknown_model(client):
    """Var olmayan model için OpenAI uyumlu 404 hatası döner."""
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={"model": "yok-model", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_chat_completion_ok(client):
    """Akışsız chat yanıtı OpenAI biçiminde ve içerik doğru gelir."""
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Merhaba!"}],
            "temperature": 0.3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "Merhaba" in msg["content"]
    assert body["usage"]["total_tokens"] == 13


def test_chat_completion_stream(client):
    """Akışlı chat yanıtı SSE parçaları ve [DONE] işareti içerir."""
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Merhaba!"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    cizgiler = resp.text.strip().split("\n\n")
    hayir_done = [c for c in cizgiler if c != "data: [DONE]"]
    assert len(hayir_done) >= 2  # en az iki içerik parçası + bitiş
    parcalar = []
    kimlikler = set()
    for c in hayir_done[:-1]:  # son içerik parçası boş + finish_reason
        veri = json.loads(c[6:])
        kimlikler.add(veri["id"])
        parcalar.append(veri["choices"][0]["delta"]["content"])
    assert parcalar == ["Mer", "haba"]
    assert len(kimlikler) == 1  # tüm parçalar aynı akış kimliğini taşır
    assert cizgiler[-1] == "data: [DONE]"


# ------------------------------------------------------------------
# Completions
# ------------------------------------------------------------------

def test_completion_ok(client):
    """Akışsız completion yanıtı OpenAI biçiminde döner."""
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={"model": "test-model", "prompt": "Bir gün", "max_tokens": 16},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "tamamlanmış metin"
    assert body["usage"]["total_tokens"] == 7


def test_completion_stream(client):
    """Akışlı completion yanıtı SSE biçiminde döner."""
    resp = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={"model": "test-model", "prompt": "Bir gün", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert resp.text.endswith("data: [DONE]\n\n")
    assert "tamam" in resp.text


# ------------------------------------------------------------------
# Araç (tool) çağrısı — chat completions
# ------------------------------------------------------------------

def test_chat_completion_with_tools_nonstream(tool_client):
    """Tool'lu chat yanıtı tool_calls ve finish_reason='tool_calls' içerir."""
    resp = tool_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "ls çalıştır"}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "bash"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    msg = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["content"] is None or msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert '"command": "ls"' in msg["tool_calls"][0]["function"]["arguments"]


def test_chat_completion_max_completion_tokens_alias(tool_client):
    """max_completion_tokens kabul edilir ve motor parametresi max_tokens'a çevrilir."""
    resp = tool_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 42,
        },
    )
    assert resp.status_code == 200


def test_chat_completion_tool_stream(tool_client):
    """Tool'lu chat akışı SSE'de tool_calls delta'ları ve tool_calls bitişi içerir."""
    resp = tool_client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "ls"}],
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}],
        },
    )
    assert resp.status_code == 200
    cizgiler = [c for c in resp.text.strip().split("\n\n") if c != "data: [DONE]"]
    tool_deltasi_geldi = False
    bitis = None
    for c in cizgiler:
        veri = json.loads(c[6:])
        if veri["choices"][0]["delta"].get("tool_calls"):
            tool_deltasi_geldi = True
        if veri["choices"][0].get("finish_reason"):
            bitis = veri["choices"][0]["finish_reason"]
    assert tool_deltasi_geldi
    assert bitis == "tool_calls"


# ------------------------------------------------------------------
# Responses API
# ------------------------------------------------------------------

def test_responses_requires_auth(content_tool_client):
    """Anahtar olmadan /v1/responses erişilemez (401)."""
    resp = content_tool_client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "merhaba"},
    )
    assert resp.status_code == 401


def test_responses_nonstream(content_tool_client):
    """Akışsız responses yanıtı mesaj kalemi ve usage ile döner."""
    resp = content_tool_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={"model": "test-model", "input": "Nasılsın?", "instructions": "Kısa cevap ver."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "test-model"
    msg = body["output"][0]
    assert msg["type"] == "message"
    assert msg["content"][0]["type"] == "output_text"
    assert "Tamamlandı" in msg["content"][0]["text"]
    assert body["usage"]["total_tokens"] == 12


def test_responses_stream(content_tool_client):
    """Responses akışı response.* olaylarını ve [DONE] işaretini içerir."""
    resp = content_tool_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={"model": "test-model", "input": "Selam", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    cizgiler = [c for c in resp.text.strip().split("\n\n") if c != "data: [DONE]"]
    tipler = [json.loads(c[6:])["type"] for c in cizgiler]
    for gerekli in (
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
        "response.done",
    ):
        assert gerekli in tipler, f"eksik olay: {gerekli}"
    delta_metin = "".join(
        json.loads(c[6:])["delta"] for c in cizgiler if json.loads(c[6:])["type"] == "response.output_text.delta"
    )
    assert delta_metin == "Merhaba"


def test_responses_input_items_mapping(content_tool_client, monkeypatch):
    """input kalemleri (message/function_call_output) doğru mesajlara çevrilir."""
    resp = content_tool_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "instructions": "Sen yardım asistanısın.",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "merhaba"}]},
                {"type": "function_call_output", "call_id": "call_1", "output": "ls çıktısı"},
            ],
        },
    )
    assert resp.status_code == 200
    # Sahte motor son aldığı mesajları kaydetti
    manager = openai_api_mod.get_manager()
    engine = manager._engines["test-model"]
    roles = [m["role"] for m in engine.last_messages]
    assert roles == ["system", "user", "tool"]
    assert engine.last_messages[2]["tool_call_id"] == "call_1"
    assert engine.last_messages[2]["content"] == "ls çıktısı"


def test_responses_toolcall(tool_client):
    """Tool çağrısıyla responses yanıtı function_call kalemi döner."""
    resp = tool_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "input": "ls çalıştır",
            "tools": [{"type": "function", "name": "bash", "description": "Kod çalıştır",
                       "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}],
            "tool_choice": {"type": "function", "name": "bash"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    call = body["output"][0]
    assert call["type"] == "function_call"
    assert call["name"] == "bash"
    assert "ls" in call["arguments"]
    assert call["status"] == "completed"


def test_responses_tool_stream(tool_client):
    """Tool'lu responses akışı function_call kalem olaylarını yayınlar."""
    resp = tool_client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        json={
            "model": "test-model",
            "input": "ls",
            "stream": True,
            "tools": [{"type": "function", "name": "bash", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "bash"},
        },
    )
    assert resp.status_code == 200
    cizgiler = [c for c in resp.text.strip().split("\n\n") if c != "data: [DONE]"]
    tipler = [json.loads(c[6:])["type"] for c in cizgiler]
    assert "response.output_item.added" in tipler
    assert "response.function_call_arguments.delta" in tipler
    assert "response.function_call_arguments.done" in tipler
    # Son function_call kalemi bash adını ve arguments içeriğini taşır
    done_items = [
        json.loads(c[6:])["item"]
        for c in cizgiler
        if json.loads(c[6:])["type"] == "response.output_item.done"
    ]
    calls = [i for i in done_items if i["type"] == "function_call"]
    assert calls and calls[0]["name"] == "bash"
    assert calls[0]["arguments"] == '{"command":"ls"}'