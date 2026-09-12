# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Paneli API'si
#  Dosya:    app/admin_api.py
#  Amaç:     Web panelinin arka ucu: ilk kurulum, giriş/çıkış, model
#            yönetimi, HuggingFace araması + indirme ve API anahtarları.
#  Mekanik:  - Halka açık   : POST /panel/setup, POST /panel/login
#            - Oturum korumalı: /panel/me, /panel/models/*, indirme,
#              /panel/apis (Depends(get_current_session))
#            - Oturum, httponly çereze (vprovider_session) yazılır.
#            - İndirme arka planda (asyncio task + thread) çalışır;
#              ilerleme hem JSON ile sorgulanır hem SSE ile akar.
#  Kullanım: app.main.py içinde app.include_router(admin_router)
# ─────────────────────────────────────────────────────────────

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import comfy_client, hf_downloader, tts_backend
from app.auth import (
    SESSION_COOKIE,
    authenticate_user,
    create_session_for_user,
    generate_token,
    get_current_session,
    hash_password,
)
from app.comfy_client import ComfyGenerationError, ComfyUnavailableError, get_client
from app.config import settings
from app.model_manager import get_manager
from app.user_store import SESSION_TTL_DAYS, get_store

logger = logging.getLogger("vprovider")

# Yönetim paneli router'ı
router = APIRouter(prefix="/panel", tags=["panel"])

# İndirme ilerlemesi için SSE aralığı (saniye)
_STREAM_INTERVAL = 0.5


# ------------------------------------------------------------------
# İstek modelleri
# ------------------------------------------------------------------

class SetupRequest(BaseModel):
    """İlk kurulum: yönetici hesabı.

    Proje kimliği (ad, site bağlantıları, logo) kod içine gömülüdür;
    bu nedenle kurulum yalnızca kullanıcı adı + şifre ister.
    """

    username: str
    password: str


class LoginRequest(BaseModel):
    """Pano girişi."""

    username: str
    password: str


class DownloadRequest(BaseModel):
    """HF modeli indirme isteği."""

    repo_id: str
    filename: str


class MemoryModeRequest(BaseModel):
    """Tek model için bellek modu."""

    memory_mode: str


class ComfyGenerateRequest(BaseModel):
    """Görsel üretim isteği (panel)."""

    prompt: str
    negative_prompt: str = ""
    checkpoint: str = ""
    size: str = "512x512"
    steps: int = 20
    cfg: float = 7.0
    seed: int = -1
    n: int = 1


class ComfyVideoGenerateRequest(BaseModel):
    """Video üretim isteği (panel)."""

    prompt: str
    checkpoint: str = ""
    size: str = "512x512"
    frames: int = 16
    steps: int = 25
    cfg: float = 7.0
    seed: int = -1


class TtsGenerateRequest(BaseModel):
    """Ses üretim isteği (panel)."""

    text: str
    voice: str = ""
    speed: float = 1.0


class PanelChatRequest(BaseModel):
    """Panel sohbet isteği (modeli arayüzden test etmek için)."""

    model: str
    messages: list[dict]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class CreateApiKeyRequest(BaseModel):
    """Yeni isimlendirilmiş API anahtarı isteği."""

    name: str = "API Anahtarı"


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

# Dosya adındaki ölçek etiketi (örn. "0.5b", "135m", "3B") -> standart biçim
_PARAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([bm])(?![a-z0-9])", re.IGNORECASE)


def _model_params(model_id: str) -> str:
    """Model dosya adındaki parametre etiketini standart biçime çevirir (bulunamazsa boş)."""
    m = _PARAM_RE.search(model_id)
    if not m:
        return ""
    return f"{m.group(1).replace(',', '.')}{m.group(2).upper()}"


# Kategori tahmini: dosya adındaki anahtar sözcükler; bulunamazsa metin modeli
_CATEGORY_KEYWORDS = {
    "görsel": ("flux", "sdxl", "stable-diffusion", "realvis", "juggernaut", "dreamshaper"),
    "ses": ("whisper", "tts", "vits", "piper", "kokoro"),
    "video": ("wav2lip", "sadtalker", "animatediff", "mochi"),
    "müzik": ("musicgen", "music", "audiocraft"),
}


def _model_category(model_id: str) -> str:
    """Model kategorisini dosya adından tahmin eder; bilinmiyorsa 'metin'."""
    low = model_id.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in low for k in keywords):
            return category
    return "metin"


def _set_session_cookie(response: Response, token: str) -> None:
    """Oturum token'ini httponly çerez olarak tarayıcıya yazar."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_DAYS * 86400,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    """Çerezi anında süresi dolmuş yaparak tarayıcıdan siler."""
    response.set_cookie(key=SESSION_COOKIE, value="", max_age=0, httponly=True, path="/")


def _require_setup_not_done() -> None:
    """Sistemde henüz hiç kullanıcı bulunmamasını garantiler (kayıt yalnızca ilk kez)."""
    if get_store().user_count() > 0:
        raise HTTPException(
            status_code=400,
            detail="Zaten kayıtlı bir kullanıcı var. Lütfen giriş yapın.",
        )


def _model_to_dict(model) -> dict:
    """ModelInfo nesnesini panel için sözlüğe çevirir."""
    return {
        "id": model.model_id,
        "path": str(model.path),
        "size_bytes": model.size_bytes,
        "size_mb": round(model.size_bytes / 1_048_576, 1),
        "loaded": model.loaded,
        "memory_mode": model.memory_mode or get_manager().memory_mode,
        "params": _model_params(model.model_id),
        "category": _model_category(model.model_id),
    }


# ------------------------------------------------------------------
# Halka açık: kurulum ve giriş
# ------------------------------------------------------------------

@router.get("/status")
def panel_status():
    """İlk kayıt (register) gerekip gerekmediğini döner (panel açılışında kullanılır).

    Kayıt sayfası yalnızca sistemde hiç kullanıcı yokken gösterilir;
    kullanıcı varsa her zaman giriş sayfası açılır.
    """
    store = get_store()
    return {
        "needs_setup": store.user_count() == 0,
        "setup_done": store.is_setup_done(),
        "app": "VProvider",
        "site": _site_brief(store.get_site_info()),
    }


@router.post("/setup")
def setup(req: SetupRequest, response: Response):
    """İlk kurulum: yönetici hesabını oluşturur ve panele giriş yapar.

    Yalnızca kurulum tamamlanmamışken çalışır. Varsayılan API anahtarı
    üretilmez; istenen anahtarlar API sayfasından elle oluşturulur.
    """
    _require_setup_not_done()
    username = req.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Kullanıcı adı en az 3 karakter olmalı")
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Şifre en az 8 karakter olmalı")

    store = get_store()
    password_hash = hash_password(req.password)
    if not store.create_user(username, password_hash):
        raise HTTPException(status_code=409, detail="Bu kullanıcı adı zaten kullanılıyor")

    store.mark_setup_done()

    user = store.get_user_by_username(username)
    token = create_session_for_user(user["id"])
    _set_session_cookie(response, token)
    logger.info("İlk kurulum tamamlandı: %s", username)
    return {"ok": True, "user": username}


@router.post("/login")
def login(req: LoginRequest, response: Response):
    """Kullanıcı adı ve şifre ile panele giriş yapar.

    Veritabanında kullanıcı yoksa giriş reddedilir (önce kayıt olunmalı).
    """
    store = get_store()
    if store.user_count() == 0:
        raise HTTPException(status_code=400, detail="Henüz kayıtlı kullanıcı yok. Önce kayıt olun (/panel/setup).")
    user = authenticate_user(req.username.strip(), req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı.")
    token = create_session_for_user(user["id"])
    _set_session_cookie(response, token)
    return {"ok": True, "user": user["username"]}


# ------------------------------------------------------------------
# Oturum korumalı: kimlik
# ------------------------------------------------------------------

@router.get("/me")
def me(session: dict = Depends(get_current_session)):
    """Geçerli oturumun kullanıcısını, kurulum durumunu ve site adını döner."""
    store = get_store()
    user = _user_of_session(session)
    info = store.get_site_info()
    return {
        "user": user["username"] if user else None,
        "setup_done": store.is_setup_done(),
        "site": _site_brief(info),
    }


def _user_of_session(session: dict) -> dict | None:
    """Oturum kaydının user_id'sinden kullanıcıyı bulur."""
    row = get_store()._run(
        "SELECT id, username FROM users WHERE id = ?", (session["user_id"],), fetch="one"
    )
    return dict(row) if row else None


def _site_brief(info: dict) -> dict:
    """Proje kimliğinden panelin ihtiyaç duyduğu kısa bilgiyi döner.

    Logo/favicon kod içine gömülü statik dosyalardır; SITE_IDENTITY
    sabitinden gelen bilgi üzerine dosya varlığına göre eklenir.
    """
    logo_file = Path(__file__).resolve().parent.parent / "static" / "logo.png"
    favicon_file = Path(__file__).resolve().parent.parent / "static" / "favicon.png"
    return {
        "project_name": info.get("project_name", "VProvider"),
        "has_logo": info.get("has_logo", True if logo_file.exists() else False),
        "has_favicon": info.get("has_favicon", True if favicon_file.exists() else False),
        "github_url": info.get("github_url", ""),
        "developer_domain": info.get("developer_domain", ""),
    }


@router.post("/logout")
def logout(response: Response, session: dict = Depends(get_current_session)):
    """Geçerli oturumu siler ve çerezi tarayıcıdan kaldırır."""
    get_store().delete_session(session["token"])
    _clear_session_cookie(response)
    return {"ok": True}


# ------------------------------------------------------------------
# Oturum korumalı: model yönetimi
# ------------------------------------------------------------------

@router.get("/models")
def list_panel_models(_: dict = Depends(get_current_session)):
    """Yerel GGUF modellerini durumlarıyla listeler."""
    return {"models": [_model_to_dict(m) for m in get_manager().list_models()]}


def _find_model(model_id: str):
    """Model diskte yoksa anlaşılır hata verir."""
    model = get_manager().get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"'{model_id}' modeli bulunamadı.")
    return model


@router.post("/models/{model_id}/load")
async def panel_load_model(model_id: str, _: dict = Depends(get_current_session)):
    """Modeli belleğe yükler."""
    _find_model(model_id)
    await get_manager().load(model_id)
    return {"ok": True, "loaded": True}


@router.post("/models/{model_id}/unload")
async def panel_unload_model(model_id: str, _: dict = Depends(get_current_session)):
    """Modeli bellekten boşaltır."""
    _find_model(model_id)
    await get_manager().unload(model_id)
    return {"ok": True, "loaded": False}


@router.post("/models/{model_id}/mode")
async def panel_set_mode(
    model_id: str, req: MemoryModeRequest, _: dict = Depends(get_current_session)
):
    """Tek modelin bellek modunu değiştirir (keep/dynamic)."""
    _find_model(model_id)
    get_manager().set_memory_mode(model_id, req.memory_mode)
    return {"ok": True, "memory_mode": req.memory_mode}


@router.post("/models/{model_id}/delete")
async def panel_delete_model(model_id: str, _: dict = Depends(get_current_session)):
    """Modeli önce boşaltır, sonra diskten siler."""
    _find_model(model_id)
    manager = get_manager()
    removed = await hf_downloader.delete_model(model_id, manager=manager, models_dir=manager.models_dir)
    logger.info("Model silindi: %s (%s)", model_id, ", ".join(removed) or "dosya yok")
    return {"ok": True, "removed": removed}


# ------------------------------------------------------------------
# Oturum korumalı: panel içi sohbet (modeli arayüzden test etmek için)
# ------------------------------------------------------------------

@router.post("/chat")
async def panel_chat(req: PanelChatRequest, _: dict = Depends(get_current_session)):
    """Seçili modelle arayüzden sohbet eder (API anahtarı gerekmez).

    Model gerekirse otomatik yüklenir; dynamic modda yanıt bitince boşaltılır.
    """
    for msg in req.messages:
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            raise HTTPException(status_code=400, detail="Geçersiz rol (system/user/assistant olmalı).")
        if not isinstance(msg.get("content"), str):
            raise HTTPException(status_code=400, detail="content alanı metin olmalı.")

    params: dict = {}
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.max_tokens is not None:
        params["max_tokens"] = req.max_tokens

    try:
        answer = await get_manager().chat(req.model, req.messages, **params)
        usage = await get_manager().usage(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Panel sohbet hatası (%s)", req.model)
        raise HTTPException(status_code=500, detail=f"Sohbet sırasında hata: {exc}")

    return {"ok": True, "model": req.model, "message": answer, "usage": usage or {}}


# ------------------------------------------------------------------
# Oturum korumalı: HuggingFace arama + indirme
# ------------------------------------------------------------------

@router.get("/models/search")
async def search_remote_models(q: str = "", category: str = "text-generation", limit: int = 20):
    """HuggingFace'te GGUF içeren modelleri arar (arka planda thread).

    q boşsa keşif listesi döner: kategorideki en çok indirilen GGUF
    modelleri (popup'ta "tüm modelleri" listeleme için kullanılır).
    """
    results = await asyncio.to_thread(
        hf_downloader.search_models, q.strip(), category, limit
    )
    return {
        "models": [
            {
                "repo_id": r.repo_id,
                "downloads": r.downloads,
                "likes": r.likes,
                "last_modified": r.last_modified,
                "gguf_count": r.gguf_count,
            }
            for r in results
        ]
    }


@router.get("/models/browse")
async def panel_browse_models(_: dict = Depends(get_current_session)):
    """GGUF içeren modellerin geniş kataloğunu döner (popup açılışında).

    İlk çağrı birkaç saniye sürebilir; hf_downloader kısa süreli önbellek
    tuttuğu için tekrar açılışlarda anında döner.
    """
    return {"models": await asyncio.to_thread(hf_downloader.browse_models)}


@router.get("/models/repo-files")
async def remote_repo_files(repo: str, _: dict = Depends(get_current_session)):
    """Seçilen repo'daki .gguf dosyalarını ad ve boyutla döner.

    repo id'si "/" barındırdığından URL yoluna değil query parametresine taşınır.
    """
    files = await asyncio.to_thread(hf_downloader.get_repo_files, repo)
    if not files:
        raise HTTPException(status_code=404, detail="Bu repo'da GGUFF dosyası bulunamadı.")
    return {
        "files": [
            {"filename": f.filename, "size_bytes": f.size_bytes,
             "size_mb": round(f.size_bytes / 1_048_576, 1)}
            for f in files
        ]
    }


@router.post("/models/download")
async def start_panel_download(req: DownloadRequest, _: dict = Depends(get_current_session)):
    """Modeli arka planda indirmeye başlar; durumu /status adresinden izlenir.

    İndirme bitince model varsayılan olarak 'dynamic' (istek ile aktif)
    bellek moduna atanır.
    """
    logger.info("İndirme başladı: %s / %s", req.repo_id, req.filename)

    async def _finish() -> None:
        try:
            result = await hf_downloader.start_download(req.repo_id, req.filename)
            if result.path:
                get_manager().set_memory_mode(Path(result.path).stem, "dynamic")
        except Exception:
            logger.exception("İndirme başarısız: %s", req.repo_id)

    task = asyncio.create_task(_finish())
    return {"ok": True, "repo_id": req.repo_id, "filename": req.filename, "task_id": id(task)}


@router.post("/models/download/cancel")
async def cancel_panel_download(req: DownloadRequest, _: dict = Depends(get_current_session)):
    """Devam eden indirmeyi iptal eder; yarım dosya temizlenir.

    İptal sinyali, indirme thread'inde parça parça doğrulanır;
    durum kısa süre içinde 'iptal' olur.
    """
    ok = hf_downloader.cancel_download(req.repo_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Bu repo için aktif bir indirme yok.")
    return {"ok": True, "repo_id": req.repo_id}


@router.get("/models/download/status")
def download_status(repo_id: str, _: dict = Depends(get_current_session)):
    """İndirme ilerlemesini JSON olarak döner (panel sorgusu için)."""
    status_ = hf_downloader.get_download_status(repo_id)
    if not status_:
        raise HTTPException(status_code=404, detail="Bu repo için indirme kaydı yok.")
    return status_


@router.get("/models/download/status/stream")
async def download_status_stream(repo_id: str, _: dict = Depends(get_current_session)):
    """İndirme ilerlemesini SSE ile akıtır (canlı çubuk için)."""
    status_ = hf_downloader.get_download_status(repo_id)
    if not status_:
        raise HTTPException(status_code=404, detail="Bu repo için indirme kaydı yok.")
    return StreamingResponse(
        _download_progress_stream(repo_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models/downloads")
def all_downloads(_: dict = Depends(get_current_session)):
    """Tüm (aktif + biten) indirme kayıtlarını döner; çoklu indirmede
    her repo'nun durumu ayrı görünür."""
    return hf_downloader.list_active_downloads()


async def _download_progress_stream(repo_id: str) -> AsyncIterator[str]:
    """İndirme bitene kadar ilerlemeyi düzenli aralıklarla SSE olarak yollar."""
    while True:
        status_ = hf_downloader.get_download_status(repo_id)
        if not status_:
            yield f"data: {json.dumps({'status': 'yok'})}\n\n"
            break
        yield f"data: {json.dumps(status_)}\n\n"
        if status_.get("status") in ("tamam", "hata", "iptal"):
            break
        await asyncio.sleep(_STREAM_INTERVAL)
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------
# Sistem durumu (oturum korumalı)
# ------------------------------------------------------------------

@router.get("/system/gpu")
def system_gpu(_: dict = Depends(get_current_session)):
    """Donanım + kurulu/aktif llama_cpp backend bilgisini döner.

    Örnek kullanım: panel "Görsel" veya "Sistem" sayfasında GPU modunu
    göstermek. GPU algılamasını tüketmez; yalnızca özet üretir.
    """
    from app.gpu_detect import compiled_backends, detect_hardware, resolve_runtime

    hw = detect_hardware()
    resolved = resolve_runtime(gpu_mode=settings.gpu_mode, requested_layers=settings.gpu_layers)
    return {
        "hardware": {
            "vendor": hw.vendor,
            "name": hw.name,
            "vram_total_mb": hw.vram_total_mb,
            "vram_free_mb": hw.vram_free_mb,
            "driver": hw.driver,
        },
        "compiled_backends": compiled_backends(),
        "backend": resolved.backend,
        "gpu_layers": resolved.gpu_layers,
        "note": resolved.note,
        "gpu_mode": settings.gpu_mode,
        "gpu_layers_setting": settings.gpu_layers,
    }


# ------------------------------------------------------------------
# ComfyUI görsel üretim (oturum korumalı)
# ------------------------------------------------------------------

@router.get("/comfy/status")
def comfy_status(_: dict = Depends(get_current_session)):
    """ComfyUI köprüsünün durumu: açık mı, motor çalışıyor mu, GPU bilgisi."""
    info = {"enabled": settings.comfyui_enabled}
    if not settings.comfyui_enabled:
        return {**info, "ok": False, "error": "COMFYUI_ENABLED=false (modül kapalı)"}
    try:
        return {**info, **get_client().status()}
    except Exception as exc:
        return {**info, "ok": False, "error": str(exc)}


@router.get("/comfy/checkpoints")
def comfy_checkpoints(_: dict = Depends(get_current_session)):
    """ComfyUI'de kurulu checkpoint'leri listeler."""
    try:
        return {"checkpoints": get_client().checkpoints()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI'ye ulaşılamadı: {exc}")


@router.post("/comfy/generate")
async def comfy_generate(req: ComfyGenerateRequest, _: dict = Depends(get_current_session)):
    """Görsel üretir; tamamlanınca görsel URL'lerini döner (senkron bekleme)."""
    try:
        result = await comfy_client.generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            checkpoint=req.checkpoint,
            size=req.size,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            batch_size=req.n,
        )
    except ComfyUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ComfyGenerationError as exc:
        raise HTTPException(status_code=500, detail=f"Üretim hatası: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI bağlantı hatası: {exc}")

    return {
        "ok": True,
        "prompt_id": result["prompt_id"],
        "images": [
            {"url": f"/panel/comfy/image/{result['prompt_id']}/{i}"}
            for i in range(len(result["images"]))
        ],
    }


@router.get("/comfy/image/{prompt_id}/{index}")
def comfy_image(prompt_id: str, index: int, _: dict = Depends(get_current_session)):
    """Üretilen görseli panel içinden sunar (depo in-memory'dir)."""
    img = comfy_client.get_stored_image(prompt_id, index)
    if img is None:
        raise HTTPException(status_code=404, detail="Görsel bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=img["bytes"], media_type=img.get("mime", "image/png"))


@router.post("/video/generate")
async def comfy_video_generate(req: ComfyVideoGenerateRequest, _: dict = Depends(get_current_session)):
    """Video üretir; tamamlanınca GIF URL'sini döner (senkron bekleme)."""
    try:
        result = await comfy_client.generate_video(
            prompt=req.prompt,
            checkpoint=req.checkpoint,
            size=req.size,
            frames=req.frames,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
        )
    except ComfyUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ComfyGenerationError as exc:
        raise HTTPException(status_code=500, detail=f"Üretim hatası: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ComfyUI bağlantı hatası: {exc}")

    video = result["video"]
    return {
        "ok": True,
        "prompt_id": result["prompt_id"],
        "url": f"/panel/comfy/video/{result['prompt_id']}",
        "mime_type": video["mime"],
        "width": video["width"],
        "height": video["height"],
        "frames": video["frames"],
    }


@router.get("/comfy/video/{prompt_id}")
def comfy_video(prompt_id: str, _: dict = Depends(get_current_session)):
    """Üretilen videoyu (GIF) panel içinden sunar."""
    video = comfy_client.get_stored_video(prompt_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=video["bytes"], media_type=video.get("mime", "image/gif"))


# ------------------------------------------------------------------
# Ses üretimi (TTS) — oturum korumalı
# ------------------------------------------------------------------

@router.get("/tts/status")
async def tts_status_endpoint(_: dict = Depends(get_current_session)):
    """TTS motoru durumu: açık mı, kurulu mu, hangi ses."""
    return await tts_backend.tts_status()


@router.post("/tts/generate")
async def tts_generate(req: TtsGenerateRequest, _: dict = Depends(get_current_session)):
    """Metni seslendirir; sonucu depoya yazar ve oynatma URL'si döner."""
    from uuid import uuid4

    key = uuid4().hex[:12]
    try:
        audio, mime = await tts_backend.synthesize(
            text=req.text,
            voice=req.voice,
            rate=tts_backend._edge_speed_rate(req.speed),
        )
    except tts_backend.TTSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ses motoru bağlantı hatası: {exc}")

    tts_backend.store_audio(key, audio, mime)
    return {"ok": True, "key": key, "url": f"/panel/tts/audio/{key}", "mime_type": mime}


@router.get("/tts/audio/{key}")
def tts_audio(key: str, _: dict = Depends(get_current_session)):
    """Üretilen sesi panel içinde oynatılmak üzere sunar."""
    audio = tts_backend.get_stored_audio(key)
    if audio is None:
        raise HTTPException(status_code=404, detail="Ses bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=audio["bytes"], media_type=audio.get("mime", "audio/mpeg"))


# ------------------------------------------------------------------
# Oturum korumalı: API anahtarları (isimlendirilmiş, birden çok)
# ------------------------------------------------------------------

@router.get("/apis")
def list_api_keys(_: dict = Depends(get_current_session)):
    """Tüm isimlendirilmiş API anahtarlarını adlarıyla döner."""
    return {"keys": get_store().list_api_keys()}


@router.post("/apis")
def create_api_key(req: CreateApiKeyRequest, _: dict = Depends(get_current_session)):
    """Yeni, isimlendirilmiş bir API anahtarı oluşturur."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Anahtar adı boş olamaz.")
    if len(name) > 50:
        raise HTTPException(status_code=400, detail="Anahtar adı en fazla 50 karakter olabilir.")
    key = generate_token()
    created = get_store().create_api_key(name, key)
    logger.info("API anahtarı oluşturuldu: %s", name)
    return {"ok": True, "key": created}


@router.delete("/apis/{key_id}")
def delete_api_key(key_id: int, _: dict = Depends(get_current_session)):
    """İsimlendirilmiş API anahtarını siler."""
    store = get_store()
    if store.get_api_key_by_id(key_id) is None:
        raise HTTPException(status_code=404, detail="API anahtarı bulunamadı.")
    store.delete_api_key(key_id)
    logger.info("API anahtarı silindi (id=%s)", key_id)
    return {"ok": True, "deleted": key_id}