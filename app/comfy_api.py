# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 11-12 - OpenAI Uyumlu Görsel + Video Üretim API'si
#  Dosya:    app/comfy_api.py
#  Amaç:     ComfyUI motorunu OpenAI /v1/görseller ve /v1/videolar biçiminde
#            sunar. Aynı kimlik doğrulama (Bearer) ve hata yapısı (detail.error)
#            korunur; böylece OpenAI istemcileri tek değişiklikle üretebilir.
#  Mekanik:  - /v1/images/generations        -> görsel; url veya b64_json
#            - /v1/images/file/{id}/{i}      -> url modunda görsel
#            - /v1/images/comfy/checkpoints  -> checkpoint listesi
#            - /v1/videos/generations        -> video (GIF); url modu
#            - /v1/videos/file/{id}          -> url modunda video (Faz 12)
#            - model alanı checkpoint seçimidir; verilmezse varsayılan kullanılır.
#  Kullanım: app/main.py içinde app.include_router(comfy_router)
# ─────────────────────────────────────────────────────────────

import base64
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.routing import APIRouter as _Router
from pydantic import BaseModel, Field

from app import comfy_client
from app.auth import require_api_key
from app.comfy_client import ComfyGenerationError, ComfyUnavailableError, get_client
from app.config import settings
from app.openai_api import _openai_error

router: _Router = APIRouter(prefix="/v1/images", dependencies=[Depends(require_api_key)])
video_router: _Router = APIRouter(prefix="/v1/videos", dependencies=[Depends(require_api_key)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class ImagesGenerationRequest(BaseModel):
    """POST /v1/images/generations istek gövdesi."""

    prompt: str = Field(..., min_length=1)
    model: str = ""                 # checkpoint adı (isteğe bağlı)
    n: int = 1                      # üretilecek görsel sayısı
    size: str = "512x512"
    response_format: str = "url"    # url | b64_json
    negative_prompt: str = ""       # OpenAI'de yok; VProvider eklentisi
    steps: int = Field(20, ge=1, le=100)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    seed: int = -1


class VideoGenerationRequest(BaseModel):
    """POST /v1/videos/generations istek gövdesi."""

    prompt: str = Field(..., min_length=1)
    model: str = ""                 # checkpoint adı (AnimateDiff için)
    size: str = "512x512"
    frames: int = Field(16, ge=4, le=120)   # kare sayısı
    steps: int = Field(25, ge=1, le=100)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    seed: int = -1


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _require_comfy() -> None:
    """ComfyUI kapalıysa OpenAI uyumlu 400 hatası verir."""
    if not settings.comfyui_enabled:
        raise _openai_error(
            400,
            "Görsel üretim modülü kapalı: .env içinde COMFYUI_ENABLED=true yapın "
            "ve ComfyUI'yi çalıştırın (bkz. deploy/comfyui-rehber.md).",
            code="comfy_disabled",
        )


def _pick_checkpoint(model: str) -> str:
    """İstenen model checkpoint olarak kullanılır; yoksa varsayılana düşer."""
    if model:
        return model
    if settings.comfyui_default_checkpoint:
        return settings.comfyui_default_checkpoint
    raise _openai_error(
        400,
        "Checkpoint belirtilmedi. request.model ile veya COMFYUI_DEFAULT_CHECKPOINT "
        "ile seçin; mevcutları GET /v1/images/comfy/checkpoints döner.",
        param="model",
        code="checkpoint_required",
    )


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.get("/comfy/checkpoints")
def list_checkpoints():
    """ComfyUI'de kurulu checkpoint'leri listeler (ürün arayüzleri için)."""
    if not comfy_client.settings.comfyui_enabled:
        raise _openai_error(400, "Görsel üretim modülü kapalı.", code="comfy_disabled")
    try:
        checkpoints = get_client().checkpoints()
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI'ye ulaşılamadı: {exc}", code="comfy_unreachable")
    return {"object": "list", "data": [{"id": c, "object": "model"} for c in checkpoints]}


@router.post("/generations")
async def create_image(req: ImagesGenerationRequest):
    """Görsel üretir ve OpenAI biçiminde url veya b64_json döner."""
    _require_comfy()
    if not 1 <= req.n <= 8:
        raise _openai_error(400, "n değeri 1-8 arasında olmalıdır.", param="n")
    if req.response_format not in ("url", "b64_json"):
        raise _openai_error(400, "response_format yalnızca url veya b64_json olabilir.", param="response_format")

    checkpoint = _pick_checkpoint(req.model)
    try:
        result = await comfy_client.generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            checkpoint=checkpoint,
            size=req.size,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            batch_size=req.n,
        )
    except ComfyUnavailableError as exc:
        raise _openai_error(400, str(exc), code="comfy_disabled")
    except ComfyGenerationError as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI bağlantı hatası: {exc}", code="comfy_unreachable")

    prompt_id = result["prompt_id"]
    data = []
    for i, img in enumerate(result["images"]):
        if req.response_format == "b64_json":
            data.append({"b64_json": base64.b64encode(img["bytes"]).decode("ascii")})
        else:
            data.append({"url": f"/v1/images/file/{prompt_id}/{i}"})
    return {"created": int(time.time()), "data": data}


@router.get("/file/{prompt_id}/{index}")
def serve_image(prompt_id: str, index: int):
    """url modunda dönen /v1/images/file/... adresinin arkasındaki görsel."""
    img = comfy_client.get_stored_image(prompt_id, index)
    if img is None:
        raise HTTPException(status_code=404, detail="Görsel bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=img["bytes"], media_type=img.get("mime", "image/png"))


# ------------------------------------------------------------------
# Uç noktalar: /v1/videos/* (Faz 12)
# ------------------------------------------------------------------

@video_router.get("/comfy/checkpoints")
def list_video_checkpoints():
    """Video üretimi için de aynı checkpoint seti kullanılır."""
    if not comfy_client.settings.comfyui_enabled:
        raise _openai_error(400, "Görsel/video üretim modülü kapalı.", code="comfy_disabled")
    try:
        checkpoints = get_client().checkpoints()
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI'ye ulaşılamadı: {exc}", code="comfy_unreachable")
    return {"object": "list", "data": [{"id": c, "object": "model"} for c in checkpoints]}


@video_router.post("/generations")
async def create_video(req: VideoGenerationRequest):
    """Video üretir ve GIF dosyasını hazırlar (url modu)."""
    _require_comfy()

    checkpoint = _pick_checkpoint(req.model)
    try:
        result = await comfy_client.generate_video(
            prompt=req.prompt,
            negative_prompt="",
            checkpoint=checkpoint,
            size=req.size,
            frames=req.frames,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
        )
    except ComfyUnavailableError as exc:
        raise _openai_error(400, str(exc), code="comfy_disabled")
    except ComfyGenerationError as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI bağlantı hatası: {exc}", code="comfy_unreachable")

    video = result["video"]
    return {
        "created": int(time.time()),
        "data": [{
            "url": f"/v1/videos/file/{result['prompt_id']}",
            "mime_type": video["mime"],
            "width": video["width"],
            "height": video["height"],
            "frames": video["frames"],
        }],
    }


@video_router.get("/file/{prompt_id}")
def serve_video(prompt_id: str):
    """url modunda dönen videoyu (GIF) sunar."""
    video = comfy_client.get_stored_video(prompt_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=video["bytes"], media_type=video.get("mime", "image/gif"))