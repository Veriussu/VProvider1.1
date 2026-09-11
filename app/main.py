# ─────────────────────────────────────────────────────────────
#  Bölüm:    Uygulama Girişi
#  Dosya:    app/main.py
#  Amaç:     FastAPI uygulamasını oluşturur, veri katmanını hazırlar,
#            logo/favicon uç noktalarını ve kullanılabilirlik
#            kontrolünü (health) sunar.
#  Mekanik:  - "app" FastAPI örneği uygulamanın kalbidir.
#            - Uygulama başlarken (lifespan) veritabanı şeması kurulur.
#            - /logo ve /favicon.ico, site_info tablosundaki BLOB'u
#              döner (admin panelden yüklenir).
#            - Sonraki fazlarda openai_api ve admin_api router'ları
#              buraya bağlanacaktır.
#  Kullanım: uvicorn app.main:app --host 0.0.0.0 --port 9055
# ─────────────────────────────────────────────────────────────

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.admin_api import router as admin_router
from app.comfy_api import router as comfy_router
from app.comfy_api import video_router as comfy_video_router
from app.config import settings
from app.openai_api import router as openai_router
from app.tts_api import router as tts_router
from app.user_store import get_store

# Yönetim panelinin tek dosyalık ön yüzü
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# Uygulama geneli loglayıcı (terminalde okunaklı çıktı verir)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("vprovider")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken çalışma klasörlerini ve veritabanını hazırlar."""
    settings.models_path.mkdir(parents=True, exist_ok=True)
    settings.database_file.parent.mkdir(parents=True, exist_ok=True)
    get_store().init()
    logger.info("%s sunucusu başlatıldı (bellek modu: %s)", settings.app_name, settings.memory_mode)
    yield
    logger.info("%s sunucusu kapatıldı", settings.app_name)


# FastAPI uygulaması (title, .env'den gelen proje adını kullanır)
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Hafif Yerel Yapay Zeka Model Sunucusu - OpenAI uyumlu API",
    lifespan=lifespan,
)

# OpenAI uyumlu /v1/* uç noktaları (API anahtarı korumalı)
app.include_router(openai_router)

# OpenAI uyumlu görsel üretim (ComfyUI köprüsü; API anahtarı korumalı)
app.include_router(comfy_router)

# OpenAI uyumlu video üretim (ComfyUI/AnimateDiff köprüsü; API anahtarı korumalı)
app.include_router(comfy_video_router)

# OpenAI uyumlu ses üretimi edge-tts (API anahtarı korumalı)
app.include_router(tts_router)

# Yönetim paneli API'si (kurulum, giriş, model/site yönetimi)
app.include_router(admin_router)

# Statik dosyalar (panel için js/css/varlıklar)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Yönetim paneli ön yüzünü sunar."""
    return FileResponse(INDEX_FILE)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP hatalarını OpenAI uyumlu biçimde döndürür.

    Eğer ayrıntı {"error": {...}} yapısındaysa olduğu gibi köke koyar;
    diğer hatalarda standart {"detail": ...} biçimini korur.
    """
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health")
def health():
    """Sunucunun ayakta olduğunu ve sürümünü raporlar."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "0.1.0",
        "memory_mode": settings.memory_mode,
    }


def serve_image(field: str, mime_default: str, name: str) -> Response:
    """Logo/favicon görsellerini döner.

    Görseller kod içine gömülü statik dosyalardan servis edilir; sistem
    kimliği sabit olduğundan DB'den okunmaz (hangi kurulum olursa olsun
    herkes aynı proje logosunu görür).
    """
    image_file = STATIC_DIR / f"{name}.png"
    if not image_file.exists():
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    mime = "image/x-icon" if name == "favicon" else mime_default
    return Response(content=image_file.read_bytes(), media_type=mime)


@app.get("/logo")
def logo():
    """Panelden yüklenen logo görselini döner."""
    return serve_image("logo", "image/png", "logo")


@app.get("/favicon.ico")
def favicon():
    """Panelden yüklenen favicon görselini döner."""
    return serve_image("favicon", "image/x-icon", "favicon")