# ─────────────────────────────────────────────────────────────
#  Bölüm:    GPU Algılama Testleri
#  Dosya:    tests/test_gpu_detect.py
#  Amaç:     Donanım algılama (NVIDIA CSV ayrıştırma), GGUF meta okuyucu,
#            otomatik gpu_layers hesabı ve resolve_runtime karar mantığını
#            donanımdan bağımsız (mock) doğrular.
#  Mekanik:  - Alt süreç ve llama_cpp çağrıları monkeypatch ile sabitlenir.
#            - Gerçek GPU'ya / model dosyasına bağımlılık yoktur.
# ─────────────────────────────────────────────────────────────

import struct

import pytest

from app import gpu_detect
from app.gpu_detect import (
    GPUInfo,
    auto_gpu_layers,
    compiled_backends,
    detect_hardware,
    gguf_meta,
    gguf_n_layers,
    resolve_runtime,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Algılama fonksiyonlarının lru_cache'ini her testte sıfırlar."""
    detect_hardware.cache_clear()
    compiled_backends.cache_clear()
    yield
    detect_hardware.cache_clear()
    compiled_backends.cache_clear()


def _hw(vendor="nvidia", name="RTX", total=8192, free=8192, driver="x"):
    return GPUInfo(vendor, name, total, free, driver)


# ─────────────────────────────────────────────
# NVIDIA CSV ayrıştırma
# ─────────────────────────────────────────────

def test_nvidia_csv_parsing(monkeypatch):
    """nvidia-smi csv çıktısı doğru ayrıştırılmalı (kart adında virgül olsa bile)."""
    fake = "NVIDIA GeForce RTX 4060, TU106 [GeForce], 550.54, 8192, 5120\n"
    monkeypatch.setattr(gpu_detect, "_run", lambda *a, **k: fake)
    info = gpu_detect._nvidia_info()
    assert info.vendor == "nvidia"
    assert "RTX 4060" in info.name
    assert info.driver == "550.54"
    assert info.vram_total_mb == 8192
    assert info.vram_free_mb == 5120


def test_nvidia_missing_tool(monkeypatch):
    """nvidia-smi yoksa/çalışmazsa None dönmeli (zararsız devam)."""
    monkeypatch.setattr(gpu_detect, "_run", lambda *a, **k: None)
    assert gpu_detect._nvidia_info() is None
    monkeypatch.setattr(gpu_detect, "_nvidia_info", lambda: None)
    monkeypatch.setattr(gpu_detect, "_amd_info", lambda: None)
    monkeypatch.setattr(gpu_detect, "_amd_sysfs_info", lambda: None)
    monkeypatch.setattr(gpu_detect, "_intel_info", lambda: None)
    assert detect_hardware().vendor == "none"


def test_detect_order_nvidia_first(monkeypatch):
    """NVIDIA algılanınca AMD/Intel aygıtlarına bakılmaz."""
    calls = {"nvidia": 0, "amd": 0, "intel": 0}

    def _nv():
        calls["nvidia"] += 1
        return _hw()

    def _amd():
        calls["amd"] += 1
        return _hw("amd", "RX 6700")

    def _intel():
        calls["intel"] += 1
        return _hw("intel", "UHD")

    monkeypatch.setattr(gpu_detect, "_nvidia_info", _nv)
    monkeypatch.setattr(gpu_detect, "_amd_sysfs_info", _amd)
    monkeypatch.setattr(gpu_detect, "_intel_info", _intel)
    info = detect_hardware()
    assert info.vendor == "nvidia"
    assert calls["amd"] == 0 and calls["intel"] == 0


# ─────────────────────────────────────────────
# GGUF meta okuyucu
# ─────────────────────────────────────────────

def _make_gguf(path, arch="llama", layers=42, with_array=True):
    """Başlık bloğunda metadata içeren küçük bir GGUF yazar."""
    def _str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    kv = _str("general.architecture") + struct.pack("<I", 8) + _str(arch)
    if with_array:
        # array düzeni: eleman türü (u32) + sayı (u64) + elemanlar
        kv += _str("general.tags") + struct.pack("<I", 9)
        kv += struct.pack("<I", 8) + struct.pack("<Q", 2) + _str("tag1") + _str("tag2")
    kv += _str(f"{arch}.block_count") + struct.pack("<I", 11) + struct.pack("<q", layers)
    n_kv = 3 if with_array else 2
    header = struct.pack("<II", 0x46554747, 3) + struct.pack("<Q", 0) + struct.pack("<Q", n_kv)
    path.write_bytes(header + kv)


def test_gguf_meta_reads_block_count(tmp_path):
    """Array (tags) ve scalar değerler güvenle okunmalı; katman sayısı bulunmalı."""
    p = tmp_path / "m.gguf"
    _make_gguf(p)
    meta = gguf_meta(str(p))
    assert meta.get("general.architecture") == "llama"
    assert meta.get("llama.block_count") == 42
    assert gguf_n_layers(str(p)) == 42


def test_gguf_meta_arch_specific(tmp_path):
    """gemma4 gibi özel mimarilerde mimariye özgü anahtar okunmalı."""
    p = tmp_path / "m.gguf"
    _make_gguf(p, arch="gemma4", layers=23, with_array=True)
    assert gguf_n_layers(str(p)) == 23


def test_gguf_meta_invalid_file(tmp_path):
    """GGUF olmayan / bozuk dosyada boş sonuç dönmeli (hata fırlatmamalı)."""
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"not a gguf file at all")
    assert gguf_meta(str(p)) == {}
    assert gguf_n_layers(str(p)) is None


def test_gguf_meta_missing_file(tmp_path):
    assert gguf_meta(str(tmp_path / "yok.gguf")) == {}


def test_gguf_meta_huge_array_aborts(tmp_path):
    """Saçma büyük array sayısı parsing'i durdurmalı; öncesi korunmalı."""
    p = tmp_path / "m.gguf"

    def _str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    header = struct.pack("<II", 0x46554747, 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
    body = _str("general.architecture") + struct.pack("<I", 8) + _str("llama")
    # elem türü u8, sayı ~4 miyar -> MAX_ARRAY_SKIP aşımı
    body += _str("general.tags") + struct.pack("<I", 9) + struct.pack("<I", 0) + struct.pack("<Q", 4_000_000_000)
    p.write_bytes(header + body)
    meta = gguf_meta(str(p))
    assert meta.get("general.architecture") == "llama"


# ─────────────────────────────────────────────
# Otomatik katman hesabı
# ─────────────────────────────────────────────

def test_auto_gpu_layers_full_offload():
    """Bol VRAM'de tümü GPU'ya (-1) karar verilmeli."""
    assert auto_gpu_layers(vram_free_mb=8192, model_size_mb=4000, n_layers=42) == -1


def test_auto_gpu_layers_unknown_vram():
    """VRAM bilinmiyorsa (intel/paylaşımlı) varsayılan -1 korunmalı."""
    assert auto_gpu_layers(vram_free_mb=0, model_size_mb=4000, n_layers=42) == -1


def test_auto_gpu_layers_partial():
    """Az boş VRAM'de kısmi katman sayısı üretilmeli (en az 1)."""
    count = auto_gpu_layers(vram_free_mb=2048, model_size_mb=4360, n_layers=42)
    assert 0 < count < 42


def test_auto_gpu_layers_tiny_vram():
    """Çok az VRAM olsa bile en az 1 katman önerilmeli."""
    assert auto_gpu_layers(vram_free_mb=64, model_size_mb=4360, n_layers=42) >= 1


# ─────────────────────────────────────────────
# resolve_runtime karar mantığı
# ─────────────────────────────────────────────

def test_auto_nvidia_cuda_compiled(monkeypatch):
    """auto modda NVIDIA + cuda derlemesi -> GPU (cuda)."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw(free=8192))
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu", "cuda"])
    monkeypatch.setattr(gpu_detect, "gguf_n_layers", lambda p: 42)
    rt = resolve_runtime("auto", -1, model_path="m", model_size_bytes=4_000_000_000)
    assert rt.backend == "cuda"
    assert rt.gpu_layers == -1  # 8 GB VRAM tüm 4 GB modeli taşır


def test_auto_cpu_fallback_when_backend_missing(monkeypatch):
    """auto modda NVIDIA var ama cuda derlenmemiş -> CPU'ya dön ve açıkla."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu"])
    rt = resolve_runtime("auto", -1, model_path="m", model_size_bytes=4_000_000_000)
    assert rt.backend == "cpu"
    assert rt.gpu_layers == 0
    assert "install.sh --rebuild" in rt.note


def test_auto_no_gpu_cpu(monkeypatch):
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw(vendor="none"))
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu"])
    rt = resolve_runtime("auto", -1)
    assert rt.backend == "cpu" and rt.gpu_layers == 0


def test_cpu_mode_forced(monkeypatch):
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cuda"])
    rt = resolve_runtime("cpu", -1)
    assert rt.backend == "cpu" and rt.gpu_layers == 0


def test_requested_zero_is_cpu(monkeypatch):
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cuda"])
    rt = resolve_runtime("auto", 0)
    assert rt.backend == "cpu" and rt.gpu_layers == 0


def test_requested_positive_kept(monkeypatch):
    """Pozitif gpu_layers değeri olduğu gibi korunmalı."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cuda"])
    monkeypatch.setattr(gpu_detect, "gguf_n_layers", lambda p: 42)
    rt = resolve_runtime("auto", 12, model_path="m", model_size_bytes=4_000_000_000)
    assert rt.backend == "cuda" and rt.gpu_layers == 12


def test_requested_negative_auto_calc(monkeypatch):
    """-1 ve az VRAM -> kısmi katman hesaplanmalı."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw(free=2048))
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cuda"])
    monkeypatch.setattr(gpu_detect, "gguf_n_layers", lambda p: 42)
    rt = resolve_runtime("auto", -1, model_path="m", model_size_bytes=4_360_000_000)
    assert rt.backend == "cuda"
    assert 0 < rt.gpu_layers < 42


def test_explicit_cuda_missing_raises(monkeypatch):
    """Zorlamalı cuda seçilmiş ama derlenmemişse anlaşılır RuntimeError."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu"])
    with pytest.raises(RuntimeError, match="install.sh --rebuild"):
        resolve_runtime("cuda", -1)


def test_explicit_vulkan_compiled(monkeypatch):
    """Vulkan zorlaması kurulu derlemeyle eşleşince GPU olarak kullanılır."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu", "vulkan"])
    monkeypatch.setattr(gpu_detect, "gguf_n_layers", lambda p: 42)
    rt = resolve_runtime("vulkan", -1, model_path="m", model_size_bytes=4_000_000_000)
    assert rt.backend == "vulkan"


def test_invalid_mode_valueerror(monkeypatch):
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw())
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu"])
    with pytest.raises(ValueError):
        resolve_runtime("tpU", -1)


def test_amd_falls_back_to_vulkan(monkeypatch):
    """AMD + rocm yok + vulkan derlenmiş -> vulkan kullanılmalı."""
    monkeypatch.setattr(gpu_detect, "detect_hardware", lambda: _hw(vendor="amd", name="RX 6700"))
    monkeypatch.setattr(gpu_detect, "compiled_backends", lambda: ["cpu", "vulkan"])
    rt = resolve_runtime("auto", -1)
    assert rt.backend == "vulkan"