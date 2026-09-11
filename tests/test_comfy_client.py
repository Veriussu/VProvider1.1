# ─────────────────────────────────────────────────────────────
#  Bölüm:    ComfyUI İstemcisi Testleri
#  Dosya:    tests/test_comfy_client.py
#  Amaç:     ComfyUI köprüsünün HTTP davranışını sahte (MockTransport)
#            ağ ile doğrular; gerçek ComfyUI kurulumu gerekmez.

#  Mekanik:  - httpx.MockTransport ile /system_stats, /object_info,
#              /prompt, /history, /view uçları simüle edilir.
#            - generate_image bütünü (kuyruk -> bekle -> indir -> depo)
#              sahte istemciyle koşulur.
#            - generate_video: sahte PNG kareleri gerçek Pillow ile GIF'e
#              birleştirilir ve deposuna yazılır.
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import json

import httpx
import pytest
from PIL import Image

from app import comfy_client as cc
from app.comfy_client import ComfyUnavailableError


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _transport(handler):
    """handlter imzası: callable(request, httpx.MockTransport'un next handler'ı)."""
    def _h(request):
        body = handler(request)
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)
    return httpx.MockTransport(_h)


def _client(handler):
    """Sahte ağlı istemci."""
    return cc.ComfyClient(base_url="http://comfy.test", transport=_transport(handler))


# ------------------------------------------------------------------
# Workflow kurulumu
# ------------------------------------------------------------------

def test_build_workflow_has_expected_nodes():
    """txt2img workflow'u SD/FLUX ailesi için doğru düğümleri içerir."""
    wf = cc.ComfyClient().build_txt2img_workflow(
        checkpoint="v1-5-pruned-emaonly.safetensors",
        prompt="güneşli orman",
        negative_prompt="karanlık",
        width=512, height=768, steps=25, cfg=7.5, seed=42,
    )
    prompt = wf["prompt"]
    assert prompt["4"]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly.safetensors"
    assert prompt["5"]["inputs"] == {"width": 512, "height": 768, "batch_size": 1}
    assert prompt["8"]["inputs"]["seed"] == 42
    assert prompt["8"]["inputs"]["steps"] == 25
    assert prompt["8"]["inputs"]["cfg"] == 7.5
    assert prompt["10"]["class_type"] == "SaveImage"  # çıktı görseli üreten düğüm
    assert "client_id" in wf


# ------------------------------------------------------------------
# Durum ve checkpoint listesi
# ------------------------------------------------------------------

def test_status_ok_parses_gpu(tmp_path):
    """system_stats cevabı ok ve GPU adını çözer."""
    handler = lambda req: {
        "system": {},
        "devices": [{"name": "NVIDIA GeForce RTX 4060", "vram_total": 8 * 1024**3}],
    }
    c = _client(handler)
    out = c.status()
    assert out["ok"] is True
    assert out["gpu"] == "NVIDIA GeForce RTX 4060"
    assert out["vram_mb"] == 8192


def test_status_reports_reachability_error():
    """ComfyUI kapalıysa ok=False ve hata mesajı döner."""

    def handler(req):
        raise httpx.ConnectError("bağlantı reddedildi")

    c = _client(handler)
    out = c.status()
    assert out["ok"] is False
    assert "bağlantı" in out["error"]


def test_checkpoints_parses_options():
    """object_info cevabından checkpoint adları çözülür."""

    def handler(req):
        return {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [None, {"options": ["a.safetensors", "b.safetensors"]}]}}
            }
        }

    c = _client(handler)
    assert c.checkpoints() == ["a.safetensors", "b.safetensors"]


# ------------------------------------------------------------------
# Kuyruk ve yoklama (history)
# ------------------------------------------------------------------

def test_submit_returns_prompt_id():
    """POST /prompt prompt_id döner."""

    def handler(req):
        assert req.url.path == "/prompt"
        payload = json.loads(req.content)
        return {"prompt_id": "pid-xyz", "number": 1, "node_errors": {}}

    c = _client(handler)
    assert c.submit({"prompt": {}, "client_id": "x"}) == "pid-xyz"


def test_result_pending_when_history_empty():
    """History boşken done=False döner."""

    def handler(req):
        assert req.url.path == "/history/pid-1"
        return {}

    c = _client(handler)
    res = c.result("pid-1")
    assert res == {"done": False}


def test_result_done_extracts_saveimage():
    """Biten işte SaveImage çıktısı görsel listesine dönüşür."""

    def handler(req):
        return {
            "pid-1": {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "10": {
                        "class_type": "SaveImage",
                        "images": [{"filename": "x.png", "subfolder": "vprovider/ab", "type": "output"}],
                    }
                },
            }
        }

    c = _client(handler)
    res = c.result("pid-1")
    assert res["done"] is True
    assert res["images"] == [{"filename": "x.png", "subfolder": "vprovider/ab", "type": "output"}]


def test_result_reports_comfy_error():
    """Status_str=error olduğunda hata mesajı döner."""

    def handler(req):
        return {
            "pid-1": {
                "status": {
                    "status_str": "error",
                    "messages": [["execution_error", "CUDA out of memory"]],
                },
            }
        }

    c = _client(handler)
    res = c.result("pid-1")
    assert res["done"] is True
    assert "CUDA out of memory" in res["error"]


def test_fetch_image_returns_bytes():
    """GET /view görsel baytlarını döner."""

    def handler(req):
        assert req.url.path == "/view"
        assert req.url.params["filename"] == "x.png"
        return httpx.Response(200, content=b"\x89PNG-gercek")


    c = _client(handler)
    assert c.fetch_image("x.png", "sub", "output") == b"\x89PNG-gercek"


# ------------------------------------------------------------------
# Generate_image bütünü
# ------------------------------------------------------------------

class FakeComfy(cc.ComfyClient):
    """generate_image testi için: kuyruk/be üret/tamamla anında döner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, payload):
        return "pid-full"

    def result(self, prompt_id):
        return {"done": True, "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}

    def fetch_image(self, filename, subfolder="", img_type="output"):
        return b"\x89PNG-veri"


def test_generate_image_end_to_end(monkeypatch):
    """Üretim: kuyruk -> tamam -> görsel indir -> depoya yaz."""
    monkeypatch.setattr(cc.settings, "comfyui_enabled", True)
    monkeypatch.setattr(cc, "ComfyClient", FakeComfy)

    result = asyncio.run(cc.generate_image(
        prompt="kedi", checkpoint="cat.safetensors", size="512x512", seed=7
    ))
    assert result["prompt_id"] == "pid-full"
    assert len(result["images"]) == 1
    assert result["images"][0]["bytes"] == b"\x89PNG-veri"
    # depoya yazıldı (panel/API görseli buradan okur)
    depo = cc.get_stored_image("pid-full", 0)
    assert depo["bytes"] == b"\x89PNG-veri"


def test_generate_image_disabled_raises(monkeypatch):
    """Modül kapalıyken anlaşılır hata verilir."""
    monkeypatch.setattr(cc.settings, "comfyui_enabled", False)
    with pytest.raises(ComfyUnavailableError):
        asyncio.run(cc.generate_image(prompt="kedi", checkpoint="cat.safetensors"))


def test_get_stored_image_missing_returns_none():
    """Depoda olmayan görsel None döner."""
    assert cc.get_stored_image("yok-pid", 0) is None


# ------------------------------------------------------------------
# Video: AnimateDiff workflow
# ------------------------------------------------------------------

def _fake_png_bytes(width=64, height=64, red=100):
    """Gerçek (içi boş) PNG üretir — Pillow'un GIF'e dönüştürebilmesi için."""
    img = Image.new("RGB", (width, height), (red, 0, 0))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_build_video_workflow_has_expected_nodes():
    """txt2video workflow'u AnimateDiff (ADE) düğümleriyle kurulur."""
    wf = cc.ComfyClient().build_video_workflow(
        checkpoint="v1-5-pruned.safetensors",
        prompt="bulutlar süzülüyor",
        negative_prompt="durağan",
        width=512, height=512, frames=24, steps=25, cfg=7.0, seed=123,
    )
    prompt = wf["prompt"]
    assert prompt["2"]["inputs"]["ckpt_name"] == "v1-5-pruned.safetensors"
    assert prompt["3"]["inputs"]["text"] == "bulutlar süzülüyor"
    assert prompt["5"]["class_type"] == "ADE_AnimateDiffLoaderWithContext"
    assert prompt["5"]["inputs"]["model"] == ["2", 0]
    assert prompt["6"]["inputs"] == {"width": 512, "height": 512, "batch_size": 24}
    assert prompt["7"]["class_type"] == "KSampler"
    assert prompt["8"]["inputs"]["vae"] == ["2", 2]
    assert prompt["9"]["class_type"] == "SaveImage"


# ------------------------------------------------------------------
# GIF birleştirme
# ------------------------------------------------------------------

def test_build_gif_returns_animated_gif():
    """Birden çok kare animasyonlu GIF'e dönüşür."""
    frames = [_fake_png_bytes(red=r) for r in (200, 100, 50)]
    gif, w, h, n = cc._build_gif(frames, fps=8)
    assert n == 3
    assert (w, h) == (64, 64)
    assert gif[:6] in (b"GIF87a", b"GIF89a")


def test_build_gif_single_frame_returns_png():
    """Tek kare olduğunda orijinal PNG korunur."""
    frames = [_fake_png_bytes()]
    gif, w, h, n = cc._build_gif(frames)
    assert n == 1
    assert gif[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_gif_empty_raises():
    """Karesiz girdi ComfyGenerationError ile reddedilir."""
    with pytest.raises(cc.ComfyGenerationError):
        cc._build_gif([])


# ------------------------------------------------------------------
# generate_video bütünü
# ------------------------------------------------------------------

class FakeVideoComfy(cc.ComfyClient):
    """generate_video testi: 2 kare üretir ve hemen tamamlandı der."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, payload):
        return "pid-video"

    def result(self, prompt_id):
        return {
            "done": True,
            "images": [
                {"filename": f"f{i}.png", "subfolder": "v", "type": "output"}
                for i in range(2)
            ],
        }

    def fetch_image(self, filename, subfolder="", img_type="output"):
        red = 250 if filename == "f0.png" else 20
        return _fake_png_bytes(red=red)


def test_generate_video_end_to_end(monkeypatch):
    """Üretim: kuyruk -> kareler -> GIF -> depoya yaz."""
    monkeypatch.setattr(cc.settings, "comfyui_enabled", True)
    monkeypatch.setattr(cc, "ComfyClient", FakeVideoComfy)

    result = asyncio.run(cc.generate_video(
        prompt="uçan balon", checkpoint="v1.safetensors", size="512x512", frames=16, seed=1
    ))
    assert result["prompt_id"] == "pid-video"
    vid = result["video"]
    assert vid["frames"] == 2
    assert vid["mime"] == "image/gif"
    assert vid["bytes"][:6] in (b"GIF87a", b"GIF89a")
    # depodan da okunabiliyor
    depo = cc.get_stored_video("pid-video")
    assert depo is not None and depo["bytes"] == vid["bytes"]


def test_generate_video_disabled_raises(monkeypatch):
    """Modül kapalıyken video üretimi de engellenir."""
    monkeypatch.setattr(cc.settings, "comfyui_enabled", False)
    with pytest.raises(ComfyUnavailableError):
        asyncio.run(cc.generate_video(prompt="x", checkpoint="y.safetensors"))


def test_get_stored_video_missing_returns_none():
    """Depoda olmayan video None döner."""
    assert cc.get_stored_video("yok-pid") is None