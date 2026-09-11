# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 1 - Yapılandırma Katmanı
#  Dosya:    app/config.py
#  Amaç:     .env dosyasındaki tüm ayarları okur ve tek noktadan sunar
#  Mekanik:  pydantic-settings, .env'i okur ve Settings sınıfına eşler.
#            Tüm modüller ortadaki "settings" nesnesini kullanır.
#  Kullanım: from app.config import settings
# ─────────────────────────────────────────────────────────────

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Uygulama geneli ayarlar. Değerler .env dosyasından gelir."""

    # Uygulama kimliği
    app_name: str = "VProvider"

    # Sunucu dinleme adresi ve portu
    host: str = "0.0.0.0"
    port: int = 9055

    # Bellek stratejisi: keep (her daim hazır) | dynamic (boşta boşalt)
    # idle_timeout_minutes = 0 -> kullanım bitince anında GPU'dan boşalt
    # (tek GPU'da birden çok model arasında geçiş için idealdir).
    memory_mode: str = "dynamic"
    idle_timeout_minutes: int = 0

    # Çalışma klasörleri (proje köküne göre otomatik tam yol üretilir)
    models_dir: str = "models"
    data_dir: str = "data"
    database_path: str = "data/vprovider.db"

    # LLM çalıştırma parametreleri
    context_size: int = 4096
    gpu_layers: int = -1        # -1 = tümü GPU'da; az VRAM'de düşürülür
    threads: int = 0            # 0 = otomatik

    # ComfyUI (görsel üretim motoru) köprüsü — isteğe bağlı modül
    comfyui_enabled: bool = False
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    comfyui_dir: str = ""       # ComfyUI kurulum klasörü (scriptler için)
    comfyui_default_checkpoint: str = ""
    comfyui_default_negative: str = "blur, ugly, low quality, watermark"

    # Ses üretimi (TTS) köprüsü — isteğe bağlı modül (Faz 13)
    tts_enabled: bool = False
    tts_engine: str = "edge"            # edge-tts (Microsoft çevrimiçi motor)
    tts_voice: str = "tr-TR-EmelNeural" # varsayılan Türkçe kadın sesi
    tts_voice_rate: str = "+0%"         # edge-tts hız ayarı (+0%/-10%/+20%...)

    # HuggingFace (isteğe bağlı: gated repolar için)
    hf_token: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def models_path(self) -> Path:
        """Modeller klasörünün mutlak yolu."""
        return Path(self.models_dir)

    @property
    def database_file(self) -> Path:
        """Veritabanı dosyasının mutlak yolu."""
        return Path(self.database_path)


# Uygulama genelinde kullanılan tek settings örneği
settings = Settings()