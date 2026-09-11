# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 6 - Model İndirici Testleri
#  Dosya:    tests/test_hf_downloader.py
#  Amaç:     Arama, dosya seçimi (nicelik), parça parça sürdürülebilir
#            indirme ve silme işlevlerini çevrimdışı (fake/sahte) doğrular.
#  Mekanik:  - HuggingFace API çağrıları sahte sınıflarla yamalanır.
#            - İndirme çekirdeği, Range destekli yerel HTTP sunucusuyla
#              gerçek ağ trafiği olmadan test edilir.
#            - PyTest'in %tp fazlalığı yok; asyncio.run() ile kullanılır.
# ─────────────────────────────────────────────────────────────

import asyncio
import http.server
import socket
import threading

import pytest

from app import hf_downloader as hf


# ------------------------------------------------------------------
# Sahte HuggingFace nesneleri
# ------------------------------------------------------------------

class FakeListModel:
    """HfApi.list_models'in döndürdüğü sahte ModelInfo."""

    def __init__(self, model_id, downloads=0, likes=0, tags=None):
        self.modelId = model_id
        self.downloads = downloads
        self.likes = likes
        self.lastModified = "2026-01-01T00:00:00.000Z"
        self.tags = tags or ["gguf"]


class FakeSibling:
    """model_info içindeki sahte dosya (sibling)."""

    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


class FakeApi:
    """HfApi'nin yeterli yüzeyi olan sahte sınıf."""

    recorded_kwargs = {}          # list_models çağrısının kwargs'ı
    repo_files = []               # model_info için [FakeSibling]
    all_models = []               # list_models için [FakeListModel]
    repo_file_map = {}            # repo_id -> dosya adı listesi (list_repo_files)

    def __init__(self, token=None):
        self.token = token

    def list_models(self, **kwargs):
        type(self).recorded_kwargs = kwargs
        return list(type(self).all_models)

    def model_info(self, repo_id, **kwargs):
        return type("Info", (), {"siblings": type(self).repo_files})()

    def list_repo_files(self, repo_id, **kwargs):
        return type(self).repo_file_map.get(repo_id, [])


# ------------------------------------------------------------------
# Arama
# ------------------------------------------------------------------

def test_search_models_filters_by_gguf_files(monkeypatch):
    """Arama metin üretimi kategorisinde yapılır; yalnızca .gguf içeren
    repo'lar sonuca girer."""
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: FakeApi())
    FakeApi.all_models = [
        FakeListModel("org/gguf-repo", downloads=123, likes=4),
        FakeListModel("org/safetensors-repo", downloads=999, likes=50),
        FakeListModel("org/ikinci-gguf", downloads=50, likes=1),
    ]
    FakeApi.repo_file_map = {
        "org/gguf-repo": ["model.Q4_K_M.gguf"],
        "org/ikinci-gguf": ["model-f16.gguf"],
    }

    results = hf.search_models(query="demo", limit=10, scan_max=10)

    assert FakeApi.recorded_kwargs.get("pipeline_tag") == "text-generation"
    assert FakeApi.recorded_kwargs.get("search") == "demo"
    assert FakeApi.recorded_kwargs.get("limit") == 10
    # safetensors repo'su atlanır; iki gguf repo kazanılır
    assert [r.repo_id for r in results] == ["org/gguf-repo", "org/ikinci-gguf"]
    assert results[0].downloads == 123
    assert results[0].gguf_count == 1


def test_search_models_respects_limit(monkeypatch):
    """İstenen limit aşılmaz."""
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: FakeApi())
    FakeApi.all_models = [
        FakeListModel(f"org/r{i}", downloads=i) for i in range(5)
    ]
    FakeApi.repo_file_map = {f"org/r{i}": ["m.gguf"] for i in range(5)}

    results = hf.search_models(limit=2, scan_max=5)
    assert len(results) == 2


# ------------------------------------------------------------------
# Dosya listeleme ve nicelik seçimi
# ------------------------------------------------------------------

def test_pick_gguf_prefers_requested_quant(monkeypatch):
    """İstenen nicelik (Q4_K_M) varsa o seçilir."""
    FakeApi.repo_files = [
        FakeSibling("model.Q8_0.gguf", 8000),
        FakeSibling("model.Q4_K_M.gguf", 4000),
        FakeSibling("model.Q4_0.gguf", 3500),
    ]
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: FakeApi())

    picked = hf.pick_gguf("org/multi")
    assert picked.filename == "model.Q4_K_M.gguf"


def test_pick_gguf_falls_back_to_largest(monkeypatch):
    """İstenen nicelik yoksa en büyük dosya seçilir."""
    FakeApi.repo_files = [
        FakeSibling("model.Q8_0.gguf", 5000),
        FakeSibling("model.Q3_K_S.gguf", 2000),
    ]
    monkeypatch.setattr(hf, "HfApi", lambda *a, **k: FakeApi())

    picked = hf.pick_gguf("org/multi")
    assert picked.filename == "model.Q8_0.gguf"


# ------------------------------------------------------------------
# Parça parça sürdürülebilir indirme (yerel HTTP sunucusu)
# ------------------------------------------------------------------

def _start_ranged_server(payload):
    """Range destekli küçük bir HTTP sunucusunu arka planda başlatır."""
    server_state = {"payload": payload, "last_range": None, "requests": 0}

    class RangedHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            server_state["requests"] += 1
            server_state["last_range"] = self.headers.get("Range")
            data = server_state["payload"]
            start = 0
            raw = self.headers.get("Range")
            if raw:
                start = int(raw.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangedHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/model.gguf"
    return url, server_state, srv


def test_resumable_get_fresh_download(tmp_path):
    """Yeni indirme tüm payload'ı yazar ve ilerlemeyi bildirir."""
    payload = bytes(range(256)) * 4  # 1024 bayt
    url, state, srv = _start_ranged_server(payload)
    dest = tmp_path / "model.gguf.part"
    progress = []

    try:
        result = hf._resumable_get(url, dest, None, lambda r, t: progress.append((r, t)))
        assert dest.read_bytes() == payload
        assert result["total"] == 1024
        assert progress and progress[-1] == (1024, 1024)
        assert state["requests"] == 1
    finally:
        srv.shutdown()


def test_resumable_get_continues_from_existing(tmp_path):
    """Kaldığı yerden (Range) devam eder; içerik eksiksiz tamamlanır."""
    payload = bytes(range(256)) * 4
    url, state, srv = _start_ranged_server(payload)
    dest = tmp_path / "model.gguf.part"
    dest.write_bytes(payload[:400])  # ilk 400 bayt zaten inmiş

    try:
        result = hf._resumable_get(url, dest, None)
        assert dest.read_bytes() == payload
        assert result["total"] == 1024
        assert result["received"] == 1024
        assert state["last_range"] == "bytes=400-"
    finally:
        srv.shutdown()


def test_resumable_get_completed_file_returns_416(tmp_path):
    """Dosya eksiksizken sunucu 416 verir; indirme tamam kabul edilir."""
    payload = bytes(range(256)) * 4
    url, state, srv = _start_ranged_server(payload)
    dest = tmp_path / "model.gguf.part"
    dest.write_bytes(payload)

    try:
        result = hf._resumable_get(url, dest, None)
        assert result == {"received": 1024, "total": 1024}
        assert state["requests"] == 1
    finally:
        srv.shutdown()


# ------------------------------------------------------------------
# download_model ve silme
# ------------------------------------------------------------------

def test_download_model_renames_part_to_gguf(monkeypatch, tmp_path):
    """İndirme tamamlanınca .part dosyası .gguf'a dönüştürülür."""
    calls = []

    def fake_resumable(url, dest, token, progress_cb, cancel_check=None):
        dest.write_bytes(b"gguf-data")
        progress_cb(9, 9)
        return {"received": 9, "total": 9}

    monkeypatch.setattr(hf, "_resumable_get", fake_resumable)
    monkeypatch.setattr(hf, "_repo_size", lambda repo, fname: 9)

    result = hf.download_model("org/rep", "model-Q4_K_M.gguf", models_dir=tmp_path)

    path = tmp_path / "model-Q4_K_M.gguf"
    assert result.path == path
    assert path.read_bytes() == b"gguf-data"
    assert not path.with_suffix(".gguf.part").exists()  # geçici dosya kalmadı
    status = hf.get_download_status("org/rep")
    assert status["status"] == "tamam"
    assert status["received"] == 9
    assert status["total"] == 9


def test_download_model_nested_filename_creates_subdir(monkeypatch, tmp_path):
    """İç içe dosya yolu (alt klasör) indirme sırasında otomatik oluşturulur."""
    def fake_resumable(url, dest, token, progress_cb, cancel_check=None):
        dest.write_bytes(b"veri")
        progress_cb(4, 4)
        return {"received": 4, "total": 4}

    monkeypatch.setattr(hf, "_resumable_get", fake_resumable)
    monkeypatch.setattr(hf, "_repo_size", lambda repo, fname: 4)

    result = hf.download_model("ggml-org/models-moved", "tinyllamas/stories260K.gguf", models_dir=tmp_path)

    assert (tmp_path / "tinyllamas").is_dir()
    assert (tmp_path / "tinyllamas" / "stories260K.gguf").read_bytes() == b"veri"
    assert not (tmp_path / "tinyllamas" / "stories260K.gguf.part").exists()


def test_download_model_raises_and_marks_error(monkeypatch, tmp_path):
    """Hata durumunda işlev hata fırlatır ve durum 'hata' olur."""
    def broken(url, dest, token, progress_cb, cancel_check=None):
        raise ConnectionError("ağ hatası")

    monkeypatch.setattr(hf, "_resumable_get", broken)
    with pytest.raises(ConnectionError):
        hf.download_model("org/rep", "model.gguf", models_dir=tmp_path)
    assert hf.get_download_status("org/rep")["status"] == "hata"


def test_download_model_cancel_cleans_part_and_marks_iptal(monkeypatch, tmp_path):
    """İptal sinyali indirmeyi durdurur, yarım dosyayı temizler, durum 'iptal' olur."""
    def cancel_on(cancel_check):
        assert cancel_check()  # iptal bayrağı _cancelled setinden okunur
        raise hf.DownloadCancelledError()

    def fake_resumable(url, dest, token, progress_cb, cancel_check=None):
        dest.write_bytes(b"yarim")
        return cancel_on(cancel_check)

    monkeypatch.setattr(hf, "_resumable_get", fake_resumable)
    monkeypatch.setattr(hf, "_repo_size", lambda repo, fname: 100)
    hf._cancelled.add("org/rep")
    try:
        with pytest.raises(hf.DownloadCancelledError):
            hf.download_model("org/rep", "model.gguf", models_dir=tmp_path)
        assert hf.get_download_status("org/rep")["status"] == "iptal"
        assert not (tmp_path / "model.gguf.part").exists()  # yarım dosya temizlendi
        assert not (tmp_path / "model.gguf").exists()  # final dosya oluşmadı
    finally:
        hf._cancelled.discard("org/rep")
        hf._active.discard("org/rep")


def test_cancel_download_returns_false_when_inactive(monkeypatch, tmp_path):
    """Aktif/başlamamış indirme üzerinde iptal isteği reddedilir."""
    assert hf.cancel_download("org/yok") is False


def test_cancel_download_succeeds_on_in_progress(monkeypatch, tmp_path):
    """Devam eden indirme için iptal çağrısı kabul edilir."""
    hf._active.add("org/x")
    try:
        assert hf.cancel_download("org/x") is True
        assert "org/x" in hf._cancelled
    finally:
        hf._active.discard("org/x")
        hf._cancelled.discard("org/x")


def test_start_download_runs_in_thread(monkeypatch, tmp_path):
    """start_download, asenkron çağrıdan thread üzerinde indirme yapar."""
    def fake_resumable(url, dest, token, progress_cb, cancel_check=None):
        dest.write_bytes(b"async-veri")
        progress_cb(10, 10)
        return {"received": 10, "total": 10}

    monkeypatch.setattr(hf, "_resumable_get", fake_resumable)
    monkeypatch.setattr(hf, "_repo_size", lambda repo, fname: 10)

    result = asyncio.run(
        hf.start_download("org/rep", "model.gguf", models_dir=tmp_path)
    )
    assert result.path.parent == tmp_path
    assert result.path.read_bytes() == b"async-veri"
    assert not (tmp_path / "model.gguf.part").exists()


def test_multiple_downloads_run_in_parallel(monkeypatch, tmp_path):
    """Farklı repoların indirmeleri aynı anda (paralel) yürür, birbirini
    beklemez: iki ayrı sunucunun istekleri doğrulayıcı bariyerde buluşur."""
    import concurrent.futures

    payload = bytes(range(256)) * 4  # 1024 bayt < tek parça -> tek istek/yükleme
    barrier = threading.Barrier(2)
    servers = []
    urls = []

    for _ in range(2):
        class BarrierHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                barrier.wait(timeout=5)  # iki indirme de burada buluşmalı
                body = payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BarrierHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        urls.append(f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")

    monkeypatch.setattr(hf, "_resolve_url", lambda repo, fname, revision="main": {
        "org/a": urls[0],
        "org/b": urls[1],
    }[repo])
    monkeypatch.setattr(
        hf,
        "get_repo_files",
        lambda repo_id, token=None: [hf.RepoFile("model.gguf", len(payload))],
    )

    def one(repo_id, filename):
        return hf.download_model(repo_id, filename, models_dir=tmp_path)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(one, "org/a", "a.gguf")
            f2 = pool.submit(one, "org/b", "b.gguf")
            r1, r2 = f1.result(timeout=10), f2.result(timeout=10)
    finally:
        for srv in servers:
            srv.shutdown()

    # Her iki reponun da tamamlandığı görülür ve dosyalar yerindedir
    assert r1.repo_id == "org/a" and r2.repo_id == "org/b"
    assert (tmp_path / "a.gguf").exists()
    assert (tmp_path / "b.gguf").exists()
    assert hf.get_download_status("org/a")["status"] == "tamam"
    assert hf.get_download_status("org/b")["status"] == "tamam"


def test_delete_model_files_removes_gguf_and_part(tmp_path):
    """silinen modelin ana dosyası ve yarım indirmesi temizlenir."""
    (tmp_path / "hedef-model.gguf").write_bytes(b"x")
    (tmp_path / "hedef-model.gguf.part").write_bytes(b"y")
    (tmp_path / "diger.gguf").write_bytes(b"z")

    removed = hf.delete_model_files("hedef-model", models_dir=tmp_path)

    assert set(removed) == {"hedef-model.gguf", "hedef-model.gguf.part"}
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"diger.gguf"}


def test_delete_model_unloads_then_removes(tmp_path):
    """Asenkron silme önce boşaltmayı, sonra dosya temizliğini yapar."""
    class FakeManager:
        def __init__(self):
            self.unloaded = []

        async def unload(self, model_id):
            self.unloaded.append(model_id)
            return True

    manager = FakeManager()
    (tmp_path / "silinecek.gguf").write_bytes(b"x")
    removed = asyncio.run(hf.delete_model("silinecek", manager=manager, models_dir=tmp_path))
    assert manager.unloaded == ["silinecek"]
    assert removed == ["silinecek.gguf"]