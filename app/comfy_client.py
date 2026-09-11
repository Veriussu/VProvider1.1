# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 11-12 - ComfyUI Köprüsü (Görsel + Video Üretimi)
#  Dosya:    app/comfy_client.py
#  Amaç:     ComfyUI (üretim motoru) ile konuşan ince istemci.
#            VProvider metin LLM için llama.cpp kullanır; görsel/video üretimde
#            motor, ayrı bir süreçte kurulu ComfyUI'dir ve HTTP API'siyle
#            çalışır. Bu modül yalnızca o köprüyü kurar; modeli kendisi
#            çalıştırmaz.
#  Mekanik:  - /system_stats  : ComfyUI ayakta mı? (kart bilgisi)
#            - /object_info   : kullanılabilir checkpoint'ler
#            - /prompt        : txt2img / txt2video workflow'u kuyruğa alır
#            - /history/{id}  : üretim bitti mi / çıktı görselleri
#            - /view          : üretilen görselin baytları
#            - generate_image(): görsel üretir, in-memory depoda saklar.
#            - generate_video (): AnimateDiff karelerini üretir, kareleri
#              Pillow ile GIF'e birleştirir ve depoya yazar (Faz 12).
#  Kullanım: comfy_client.generate_image(prompt="kedi", ...)
#            comfy_client.generate_video(prompt="kedi koşuyor", ...)
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import threading
import time
from typing import Generator, Optional
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger("vprovider")

# txt2img varsayılanları (SD 1.5/SDXL ailesi için uygun değerler)
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "normal"
DEFAULT_BATCH = 1
# ComfyUI'de işin bitmesi için en çok beklenen süre (saniye)
GENERATE_TIMEOUT = 300
# Depoda tutulacak son görsel üretim sayısı (bellek korunur)
_IMAGE_STORE_CAP = 40

# txt2video varsayılanları (AnimateDiff - SD 1.5 tabanlı)
VIDEO_DEFAULT_STEPS = 25
VIDEO_DEFAULT_CFG = 7.0
VIDEO_DEFAULT_FRAMES = 16
VIDEO_FPS = 8                      # GIF kare hızı (fps)
VIDEO_MOTION_MODULE = "mm_sd_v15_v2.ckpt"  # SD 1.5 hareket modülü (ADE)
VIDEO_GENERATE_TIMEOUT = 600       # video daha uzun sürer
_VIDEO_STORE_CAP = 20


# ------------------------------------------------------------------
# Görsel deposu (in-memory): üretilen görseller kısa süreliğine tutulur.
# ------------------------------------------------------------------

_image_store: dict[str, list[dict]] = {}
_image_store_lock = threading.Lock()


def _store_images(prompt_id: str, images: list[dict]) -> None:
    """Üretilen görselleri çoktan-azan düzen (cap) ile saklar."""
    with _image_store_lock:
        _image_store[prompt_id] = images
        # Eski kayıtları temizle (sadece son N üretim korunur)
        while len(_image_store) > _IMAGE_STORE_CAP:
            _image_store.pop(next(iter(_image_store)))


def get_stored_image(prompt_id: str, index: int) -> Optional[dict]:
    """Depolardan tek görseli döner (yoksa None)."""
    with _image_store_lock:
        images = _image_store.get(prompt_id)
        if not images or index >= len(images):
            return None
        return images[index]


# ------------------------------------------------------------------
# Video deposu (in-memory): üretilen GIF'ler kısa süreliğine tutulur.
# ------------------------------------------------------------------

_video_store: dict[str, dict] = {}
_video_store_lock = threading.Lock()


def _store_video(prompt_id: str, video: dict) -> None:
    """Üretilen videoyu çoktan-azan düzen (cap) ile saklar."""
    with _video_store_lock:
        _video_store[prompt_id] = video
        while len(_video_store) > _VIDEO_STORE_CAP:
            _video_store.pop(next(iter(_video_store)))


def get_stored_video(prompt_id: str) -> Optional[dict]:
    """Depodan videoyu döner (yoksa None)."""
    with _video_store_lock:
        return _video_store.get(prompt_id)


# ------------------------------------------------------------------
# ComfyUI HTTP istemcisi
# ------------------------------------------------------------------

class ComfyUnavailableError(RuntimeError):
    """ComfyUI kapalı veya erişilemiyor."""


class ComfyGenerationError(RuntimeError):
    """ComfyUI içinde üretim hatası."""


class ComfyClient:
    """ComfyUI'nin HTTP API'siyle konuşan istemci.

    transport parametresi testlerde sahte (MockTransport) ağ için verilir;
    gerçek kullanımda boş bırakılır.
    """

    def __init__(self, base_url: Optional[str] = None, transport: Optional[httpx.BaseTransport] = None):
        self.base_url = base_url or f"http://{settings.comfyui_host}:{settings.comfyui_port}"
        self._transport = transport

    def _client(self) -> httpx.Client:
        if self._transport is not None:
            return httpx.Client(transport=self._transport, base_url=self.base_url, timeout=10.0)
        return httpx.Client(base_url=self.base_url, timeout=10.0)

    # ---------- durum / kapasite ----------

    def status(self) -> dict:
        """GET /system_stats -> ComfyUI ve GPU kartı bilgisi."""
        try:
            with self._client() as c:
                r = c.get("/system_stats")
                r.raise_for_status()
                data = r.json()
            devices = (data.get("devices") or [{}])
            gpu = devices[0].get("name") if devices else ""
            return {"ok": True, "gpu": gpu, "vram_mb": _gpu_vram_mb(devices[0])}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def checkpoints(self) -> list[str]:
        """GET /object_info/CheckpointLoaderSimple -> .safetensors adları."""
        with self._client() as c:
            r = c.get("/object_info/CheckpointLoaderSimple")
            r.raise_for_status()
            node = r.json().get("CheckpointLoaderSimple", {})
            required = node.get("input", {}).get("required", {})
            meta = (required.get("ckpt_name") or [None, {}])[1]
        return list(meta.get("options", []) or [])

    # ---------- workflow kurulumu ----------

    def build_txt2img_workflow(
        self,
        checkpoint: str,
        prompt: str,
        negative_prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = DEFAULT_STEPS,
        cfg: float = DEFAULT_CFG,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        seed: int = -1,
        batch_size: int = DEFAULT_BATCH,
    ) -> dict:
        """SD 1.5 / SDXL ailesi için klasik txt2img workflow'u kurar.

        Düğümler (ComfyUI API biçimi): checkpoint -> prompt/negatif kodlama
        -> KSampler -> VAEDecode -> SaveImage. seed=-1 rastgele olur.
        """
        return {
            "prompt": {
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch_size}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
                "8": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": seed, "steps": steps, "cfg": cfg,
                        "sampler_name": sampler, "scheduler": scheduler,
                        "denoise": 1.0,
                        "model": ["4", 0], "positive": ["6", 0],
                        "negative": ["7", 0], "latent_image": ["5", 0],
                    },
                },
                "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 2]}},
                "10": {
                    "class_type": "SaveImage",
                    "inputs": {"filename_prefix": f"vprovider/{uuid4().hex[:10]}", "images": ["9", 0]},
                },
            },
            "client_id": f"vprovider-{uuid4().hex[:8]}",
        }

    # ---------- kuyruk / yoklama ----------

    def build_video_workflow(
        self,
        checkpoint: str,
        prompt: str,
        negative_prompt: str,
        width: int = 512,
        height: int = 512,
        frames: int = VIDEO_DEFAULT_FRAMES,
        steps: int = VIDEO_DEFAULT_STEPS,
        cfg: float = VIDEO_DEFAULT_CFG,
        sampler: str = DEFAULT_SAMPLER,
        scheduler: str = DEFAULT_SCHEDULER,
        seed: int = -1,
    ) -> dict:
        """AnimateDiff (txt2video) workflow'u kurar.

        Düğümler: checkpoint -> pozitif/negatif kodlama -> AnimateDiff
        Loader (ADE) -> EmptyLatentImage(batch=kare sayısı) -> KSampler
        -> VAEDecode -> SaveImage (her kare). Kurulum ŞUNLARI gerektirir:
        ComfyUI Manager + "AnimateDiff Evolved" node paketi ve
        models/animate_diff/ içinde mm_sd_v15_v2.ckpt hareket modülü
        (bkz. deploy/comfyui-rehber.md). Kareler sunucuda GIF'e birleştirilir.
        """
        prefix = f"vprovider/video/{uuid4().hex[:10]}"
        return {
            "prompt": {
                "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 1]}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 1]}},
                "5": {
                    "class_type": "ADE_AnimateDiffLoaderWithContext",
                    "inputs": {
                        "model": ["2", 0],
                        "beta_schedule": "sqrt_linear (AnimateDiff)",
                        "model_name": VIDEO_MOTION_MODULE,
                    },
                },
                "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": frames}},
                "7": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": seed, "steps": steps, "cfg": cfg,
                        "sampler_name": sampler, "scheduler": scheduler,
                        "denoise": 1.0,
                        "model": ["5", 0], "positive": ["3", 0],
                        "negative": ["4", 0], "latent_image": ["6", 0],
                    },
                },
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 2]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
            },
            "client_id": f"vprovider-{uuid4().hex[:8]}",
        }

    def submit(self, payload: dict) -> str:
        """POST /prompt -> workflow'u kuyruğa alır, prompt_id döner."""
        with self._client() as c:
            r = c.post("/prompt", json=payload)
            r.raise_for_status()
            data = r.json()
        if "prompt_id" not in data:
            raise ComfyGenerationError(data.get("error", "Bilinmeyen ComfyUI hatası"))
        return data["prompt_id"]

    def result(self, prompt_id: str) -> dict:
        """GET /history/{id} -> {done, error?, images:[{filename,subfolder,type}]}"""
        with self._client() as c:
            r = c.get(f"/history/{prompt_id}")
            r.raise_for_status()
            history = r.json()
        entry = history.get(prompt_id)
        if not entry:
            return {"done": False}
        # ComfyUI hata işaretini status_str="error" ile bildirir
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            msgs = status.get("messages") or []
            error = msgs[0][1] if msgs else "Üretim sırasında bilinmeyen hata"
            return {"done": True, "error": str(error)}
        images = []
        for cls, out in (entry.get("outputs") or {}).items():
            if out.get("class_type") == "SaveImage":
                for img in out.get("images", []):
                    images.append({
                        "filename": img.get("filename", ""),
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
        return {"done": True, "images": images}

    def fetch_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
        """GET /view?filename=...&subfolder=...&type=... -> görsel baytları."""
        params = urlencode({"filename": filename, "subfolder": subfolder, "type": img_type})
        with self._client() as c:
            r = c.get(f"/view?{params}")
            r.raise_for_status()
            return r.content


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _gpu_vram_mb(device: dict) -> int:
    """ComfyUI cihaz kaydından VRAM'i MB olarak çözer."""
    vram = (device or {}).get("vram_total") or 0
    return int(vram // (1024 * 1024))


def _size_tuple(size: str) -> tuple[int, int]:
    """'512x768' gibi boyutların <width, height> biçimini döner."""
    try:
        return tuple(int(p) for p in size.lower().split("x")[:2])
    except (ValueError, TypeError):
        return (512, 512)


def build_default_negative() -> str:
    """ComfyUI ayarlarından varsayılan negatif prompt'u döner."""
    return settings.comfyui_default_negative or ""


def _build_gif(frames: list[bytes], fps: int = VIDEO_FPS) -> tuple[bytes, int, int, int]:
    """PNG kareleri animasyonlu GIF'e birleştirir.

    Dönüş: (gif_baytları, genişlik, yükseklik, kare_sayısı). Tek kare varsa
    giriş PNG'i olduğu gibi döner.
    """
    if not frames:
        raise ComfyGenerationError("Video üretimi kare üretmedi.")
    images = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    width, height = images[0].size
    if len(images) == 1:
        return frames[0], width, height, 1
    out = io.BytesIO()
    lap = max(1, int(1000 / max(fps, 1)))
    images[0].save(
        out, format="GIF", save_all=True,
        append_images=images[1:], duration=lap, loop=0,
    )
    return out.getvalue(), width, height, len(images)


async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    checkpoint: str = "",
    size: str = "512x512",
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    seed: int = -1,
    batch_size: int = DEFAULT_BATCH,
    timeout: int = GENERATE_TIMEOUT,
) -> dict:
    """Görsel üretir ve deposuna yazar.

    Adımlar: workflow kur -> kuyruğa al -> bitene kadar yokla (thread'lerde
    ağ çağrısı) -> görselleri indir -> depoda sakla.
    Dönüş: {"prompt_id":..., "images":[{"bytes","mime"}]}
    """
    if not settings.comfyui_enabled:
        raise ComfyUnavailableError("ComfyUI köprüsü kapalı (COMFYUI_ENABLED=false).")

    width, height = _size_tuple(size)
    client = ComfyClient()
    payload = client.build_txt2img_workflow(
        checkpoint=checkpoint or settings.comfyui_default_checkpoint,
        prompt=prompt,
        negative_prompt=negative_prompt or build_default_negative(),
        width=width, height=height, steps=steps, cfg=cfg, seed=seed, batch_size=batch_size,
    )

    prompt_id = await asyncio.to_thread(client.submit, payload)
    logger.info("Görsel üretim kuyruğa alındı: %s", prompt_id)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = await asyncio.to_thread(client.result, prompt_id)
        if res.get("done"):
            if res.get("error"):
                raise ComfyGenerationError(res["error"])
            images = []
            for img in res.get("images", []):
                data = await asyncio.to_thread(
                    client.fetch_image, img["filename"], img["subfolder"], img["type"]
                )
                images.append({"bytes": data, "mime": "image/png"})
            _store_images(prompt_id, images)
            logger.info("Görsel üretim tamam: %s (%d görsel)", prompt_id, len(images))
            return {"prompt_id": prompt_id, "images": images}
        await asyncio.sleep(0.5)

    raise ComfyGenerationError(f"Üretim {timeout}s içinde tamamlanmadı.")


async def generate_video(
    prompt: str,
    negative_prompt: str = "",
    checkpoint: str = "",
    size: str = "512x512",
    frames: int = VIDEO_DEFAULT_FRAMES,
    steps: int = VIDEO_DEFAULT_STEPS,
    cfg: float = VIDEO_DEFAULT_CFG,
    seed: int = -1,
    timeout: int = VIDEO_GENERATE_TIMEOUT,
) -> dict:
    """Video üretir (AnimateDiff) ve GIF olarak deposuna yazar.

    Adımlar: txt2video workflow kur -> kuyruğa al -> bitene kadar yokla ->
    kareleri indir -> Pillow ile GIF'e birleştir -> depoda sakla.
    Dönüş: {"prompt_id":..., "video":{"bytes","mime","width","height","frames"}}
    """
    if not settings.comfyui_enabled:
        raise ComfyUnavailableError("ComfyUI köprüsü kapalı (COMFYUI_ENABLED=false).")

    width, height = _size_tuple(size)
    client = ComfyClient()
    payload = client.build_video_workflow(
        checkpoint=checkpoint or settings.comfyui_default_checkpoint,
        prompt=prompt,
        negative_prompt=negative_prompt or build_default_negative(),
        width=width, height=height, frames=frames, steps=steps, cfg=cfg, seed=seed,
    )

    prompt_id = await asyncio.to_thread(client.submit, payload)
    logger.info("Video üretim kuyruğa alındı: %s", prompt_id)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        res = await asyncio.to_thread(client.result, prompt_id)
        if res.get("done"):
            if res.get("error"):
                raise ComfyGenerationError(res["error"])
            raw_frames = []
            for img in res.get("images", []):
                data = await asyncio.to_thread(
                    client.fetch_image, img["filename"], img["subfolder"], img["type"]
                )
                raw_frames.append(data)
            gif_bytes, w, h, n = await asyncio.to_thread(_build_gif, raw_frames)
            mime = "image/gif" if n > 1 else "image/png"
            video = {"bytes": gif_bytes, "mime": mime, "width": w, "height": h, "frames": n}
            _store_video(prompt_id, video)
            logger.info("Video üretim tamam: %s (%d kare, %dx%d)", prompt_id, n, w, h)
            return {"prompt_id": prompt_id, "video": video}
        await asyncio.sleep(0.5)

    raise ComfyGenerationError(f"Video üretimi {timeout}s içinde tamamlanmadı.")


# ------------------------------------------------------------------
# Singleton istemci (testlerde _test_client_override ile değiştirilebilir)
# ------------------------------------------------------------------

_test_client_override: Optional[ComfyClient] = None


def get_client() -> ComfyClient:
    """Uygulama genelinde kullanılacak ComfyUI istemcisini döner."""
    return _test_client_override or ComfyClient()