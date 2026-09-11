# ─────────────────────────────────────────────────────────────
#  Bölüm:    Faz 3 - Model Yöneticisi Testleri
#  Dosya:    tests/test_model_manager.py
#  Amaç:     GGUF taraması, bellek modları (keep/dynamic), yükleme/
#            boşaltma ve idle süresiyle otomatik boşaltmayı sahte
#            motor ile doğrular. Gerçek llama.cpp motoruna ihtiyaç duymaz.
#  Mekanik:  FakeEngine, gerekli arayüzü uygular ve yükleme/boşaltma
#            sayılarını kaydeder; böylece yöneticinin davranışı ölçülür.
# ─────────────────────────────────────────────────────────────

import asyncio
import time

from app.model_manager import ModelManager


class FakeEngine:
    """Gerçek motoru taklit eden sahte motor (testler için)."""

    def __init__(self, path):
        self.path = path
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self._loaded = True
        self.load_count += 1

    def unload(self):
        self._loaded = False
        self.unload_count += 1

    @property
    def is_loaded(self):
        return self._loaded

    def run_chat(self, messages):
        return "Sahte yanıt: " + messages[-1]["content"]

    def stream_chat(self, messages):
        yield "Sahte "
        yield "akış "
        yield "yanıtı"


async def _test(fn):
    return asyncio.run(fn)


def _manager(tmp_path, memory_mode="keep", idle_timeout=0):
    """Ortak test yöneticisini kurar: geçici klasör + sahte motorlar."""
    manager = ModelManager(
        models_dir=tmp_path,
        engine_factory=lambda info: FakeEngine(str(info.path)),
        memory_mode=memory_mode,
        idle_timeout_minutes=idle_timeout,
    )
    # Sahte .gguf dosyaları oluştur
    (tmp_path / "model-a.gguf").write_bytes(b"data")
    (tmp_path / "model-b-q4.gguf").write_bytes(b"data")
    return manager


async def _run_chat(manager, model_id):
    return await manager.chat(model_id, [{"role": "user", "content": "merhaba"}])


async def _wait_unloaded(manager, model_id, tries=50):
    """Anında boşaltma görevinin çalışmasını bekler (0.01 s adımlarla)."""
    for _ in range(tries):
        if not manager._is_loaded(model_id):
            return True
        await asyncio.sleep(0.01)
    return not manager._is_loaded(model_id)


# ------------------------------------------------------------------
# Tarama
# ------------------------------------------------------------------

def test_scan_finds_gguf_files(tmp_path):
    """Diskteki tüm .gguf dosyaları model olarak listelenir."""
    manager = _manager(tmp_path)
    models = manager.scan_models()
    ids = {m.model_id for m in models}
    assert ids == {"model-a", "model-b-q4"}
    assert all(m.size_bytes > 0 for m in models)


# ------------------------------------------------------------------
# keep modu (her daim hazır)
# ------------------------------------------------------------------

def test_keep_mode_stays_loaded(tmp_path):
    """keep modunda model yüklenir ve kullanımdan sonra bellekte kalır."""
    manager = _manager(tmp_path, memory_mode="keep")
    ids = [m.model_id for m in manager.scan_models()]

    async def scenario():
        answer = await manager.chat(ids[0], [{"role": "user", "content": "merhaba"}])
        return answer, manager._is_loaded(ids[0])

    answer, still_loaded = asyncio.run(scenario())
    assert "Sahte yanıt" in answer
    assert still_loaded is True  # keep: kullanımdan sonra boşaltılmaz


def test_keep_mode_does_not_reload(tmp_path):
    """keep modunda ikinci istekte motor yeniden yüklenmemelidir."""
    manager = _manager(tmp_path, memory_mode="keep")
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        await manager.chat(model_id, [{"role": "user", "content": "x"}])
        await manager.chat(model_id, [{"role": "user", "content": "y"}])
        engine = manager._engines[model_id]
        return engine.load_count

    load_count = asyncio.run(scenario())
    assert load_count == 1


# ------------------------------------------------------------------
# dynamic modu (kullanırken yükle, boşta boşalt)
# ------------------------------------------------------------------

def test_dynamic_mode_unloads_after_idle(tmp_path):
    """dynamic modda idle süresi dolunca model bellekten boşaltılır."""
    # 6 milisaniye ≈ idle süresi 0.0001 dakika
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0.0001)
    model_id = manager.scan_models()[0].model_id
    engine = None

    async def scenario():
        nonlocal engine
        await manager.chat(model_id, [{"role": "user", "content": "x"}])
        engine = manager._engines[model_id]
        assert engine.is_loaded is True  # işlem sırasında yüklü
        await asyncio.sleep(0.1)  # idle süresinin geçmesini bekle

    asyncio.run(scenario())
    assert engine is not None
    assert engine.is_loaded is False
    assert engine.unload_count == 1


def test_dynamic_mode_reloads_on_next_request(tmp_path):
    """dynamic modda boşaltılmış model yeni istekte (yeni motorla) yeniden yüklenir."""
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0.0001)
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        answer1 = await manager.chat(model_id, [{"role": "user", "content": "x"}])
        engine1 = manager._engines[model_id]  # ilk yüklenen motor
        await asyncio.sleep(0.1)              # idle: model boşaltılsın
        assert manager._is_loaded(model_id) is False
        answer2 = await manager.chat(model_id, [{"role": "user", "content": "y"}])
        engine2 = manager._engines[model_id]  # yeniden yüklenen motor
        return answer1, answer2, engine1, engine2

    a1, a2, e1, e2 = asyncio.run(scenario())
    assert "Sahte yanıt" in a1
    assert "Sahte yanıt" in a2
    # Yeniden yükleme, yeni bir motor nesnesi üretir (her ikisi birer kez yüklendi)
    assert e1 is not e2
    assert e1.load_count == 1
    assert e2.load_count == 1


# ------------------------------------------------------------------
# Anında boşaltma (idle_timeout = 0): kullanım bitince GPU'dan ayrıl
# ------------------------------------------------------------------

def test_dynamic_zero_unloads_immediately_after_use(tmp_path):
    """idle_timeout=0 ise yanıt biter bitmez model GPU'dan boşaltılır."""
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0)
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        answer = await manager.chat(model_id, [{"role": "user", "content": "x"}])
        freed = await _wait_unloaded(manager, model_id)
        return answer, freed, manager._engines.get(model_id)

    answer, freed, engine = asyncio.run(scenario())
    assert "Sahte yanıt" in answer
    assert freed is True
    assert engine is None  # motor kaydı da kaldırıldı (VRAM/RAM boş)

def test_dynamic_zero_rotates_between_models(tmp_path):
    """Tek GPU senaryosu: model A kullanılır, kullanım biter bitmez boşalır,
    model B yüklenir; ikisi de bellekten çıkmış halde sonlanır."""
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0)
    ids = [m.model_id for m in manager.scan_models()]  # model-a, model-b-q4

    async def scenario():
        await manager.chat(ids[0], [{"role": "user", "content": "a"}])
        a_free = await _wait_unloaded(manager, ids[0])
        await manager.chat(ids[1], [{"role": "user", "content": "b"}])
        b_free = await _wait_unloaded(manager, ids[1])
        return a_free, b_free

    a_free, b_free = asyncio.run(scenario())
    assert a_free is True    # A kullanımı biter bitmez boşaldı
    assert b_free is True    # B de kullanımdan sonra boşaldı
    assert len(manager._engines) == 0  # hiçbir model GPU'da kalmadı

def test_switch_to_dynamic_zero_frees_kept_model(tmp_path):
    """keep'te yüklü model, dynamic+0'a geçirilince kısa sürede boşaltılır."""
    manager = _manager(tmp_path, memory_mode="keep")
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        await manager.load(model_id)
        assert manager._is_loaded(model_id) is True
        manager.set_memory_mode(model_id, "dynamic")  # 0 süreli dynamic varsayılanı
        return await _wait_unloaded(manager, model_id)

    assert asyncio.run(scenario()) is True


# ------------------------------------------------------------------
# Mod değiştirme ve manuel boşaltma
# ------------------------------------------------------------------

def test_memory_mode_change_keep_prevents_unload(tmp_path):
    """dynamic'den keep'e geçilirse idle görevi iptal edilir; model kalır."""
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0.0001)
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        await manager.load(model_id, memory_mode="keep")
        await asyncio.sleep(0.1)
        return manager._is_loaded(model_id)

    still_loaded = asyncio.run(scenario())
    assert still_loaded is True


def test_manual_unload(tmp_path):
    """Açık boşaltma çağrısı modeli bellekten çıkarır."""
    manager = _manager(tmp_path, memory_mode="keep")
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        await manager.load(model_id)
        assert manager._is_loaded(model_id) is True
        await manager.unload(model_id)
        return manager._is_loaded(model_id)

    assert asyncio.run(scenario()) is False


def test_unknown_model_raises(tmp_path):
    """Var olmayan bir model istenince anlaşılır hata fırlatılır."""
    manager = _manager(tmp_path)

    async def scenario():
        await manager.chat("yok-boyle-model", [{"role": "user", "content": "x"}])

    try:
        asyncio.run(scenario())
        assert False, "Hata fırlatılmadı"
    except ValueError as exc:
        assert "bulunamadı" in str(exc)


# ------------------------------------------------------------------
# Akışlı yanıt (streaming)
# ------------------------------------------------------------------

def test_stream_chat_yields_chunks(tmp_path):
    """Akışlı yanıt parça parça üretilir ve bittiğinde boşaltma zamanlanır."""
    manager = _manager(tmp_path, memory_mode="dynamic", idle_timeout=0.0001)
    model_id = manager.scan_models()[0].model_id

    async def scenario():
        parcalar = []
        async for chunk in manager.chat_stream(model_id, [{"role": "user", "content": "x"}]):
            parcalar.append(chunk)
        return parcalar

    parcalar = asyncio.run(scenario())
    assert "".join(parcalar) == "Sahte akış yanıtı"