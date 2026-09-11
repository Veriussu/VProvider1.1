# ─────────────────────────────────────────────────────────────
#  Bölüm:    LLM Motoru Testleri
#  Dosya:    tests/test_llama_backend.py
#  Amaç:     Gerçek llama.cpp motorunun (LlamaCppEngine) yükleme,
#            yanıt üretme, akışlı yanıt ve boşaltma davranışını doğrular.
#  Mekanik:  - Gerekli bağımlılık (llama-cpp-python) yoksa atlanır.
#            - İndirilmiş deneme modeli (models/smoke/...) yoksa atlanır.
#            - Normal koşulda küçük modelle hızlıca çalışır.
# ─────────────────────────────────────────────────────────────

from pathlib import Path

import pytest

from app.llama_backend import LlamaEngine, create_engine

# Deneme modeli: install.sh sonrası veya manuel indirme ile oluşur
SMOKE_MODEL = Path("models/smoke/qwen2-0.5b-q2.gguf")


def _has_llama_cpp() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


# Hem gerçek motora (llama-cpp-python) hem de deneme modeline ihtiyaç var
pytestmark = pytest.mark.skipif(
    not SMOKE_MODEL.exists() or not _has_llama_cpp(),
    reason="llama-cpp-python veya deneme modeli eksik (models/smoke/qwen2-0.5b-q2.gguf)",
)


@pytest.fixture()
def engine():
    """Küçük bir gerçek modelle motoru kurar; test sonunda boşaltır."""
    eng = create_engine(
        str(SMOKE_MODEL),
        context_size=512,
        gpu_layers=0,   # testlerde donanımdan bağımsız çalışsın (CPU)
        threads=2,
    )
    yield eng
    eng.unload()


def test_engine_is_llama_interface(engine):
    """Üretilen motor arayüz sözleşmesine uyar."""
    assert isinstance(engine, LlamaEngine)


def test_load_run_unload(engine):
    """Yükleme -> yanıt üretme -> boşaltma akışı çalışır."""
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True

    yanit = engine.run_chat([{"role": "user", "content": "Kısa bir yanıt ver."}])
    assert isinstance(yanit, str)
    assert len(yanit) > 0  # anlamlı bir metin üretmeli

    engine.unload()
    assert engine.is_loaded is False


def test_stream_chat_yields_text(engine):
    """Akışlı yanıt parça parça üretilir ve parçalar birleşince metin olur."""
    engine.load()
    parcalar = list(engine.stream_chat([{"role": "user", "content": "Merhaba!"}]))
    assert len(parcalar) > 0
    assert all(isinstance(p, str) for p in parcalar)
    assert "".join(parcalar).strip() != ""


def test_run_without_load_raises():
    """Yüklenmemiş motora yanıt istemek anlaşılır hata verir."""
    eng = create_engine(str(SMOKE_MODEL), context_size=256, gpu_layers=0)
    with pytest.raises(RuntimeError):
        eng.run_chat([{"role": "user", "content": "x"}])


def test_create_engine_status():
    """Model dosyası mevcut olduğunda motor üretilebilir."""
    eng = create_engine(str(SMOKE_MODEL), context_size=256, gpu_layers=0)
    assert eng is not None
    eng.unload()