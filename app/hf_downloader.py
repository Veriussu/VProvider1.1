# ─────────────────────────────────────────────────────────────
#  Bölüm:    Model İndirici
#  Dosya:    app/hf_downloader.py
#  Amaç:     HuggingFace'ten GGUF modellerini aramak, indirmek ve
#            silmek için yalın bir katman sunar.
#  Mekanik:  - Arama: HfApi.list_models (gguf etiketi + metin üretimi)
#            - Dosya seçimi: model_info(files_metadata) ile repo içindeki
#              .gguf dosyaları ve boyutları listelenir; varsayılan
#              nicelik Q4_K_M tercih edilir.
#            - İndirme: httpx ile parça parça, bölüm (Range) destekli ve
#              sürdürülebilir (resumable). ".gguf.part" geçici dosyasına
#              inilir, tamamlanınca atomik olarak ".gguf" yapılır; yarım
#              dosyalar model listesinde görünmez.
#            - İlerleme: in-memory sözlüğe yazılır (panel sonra SSE ile
#              okuyabilir). Eşzamanlı aynı repo indirmesi korunur.
#            - Silme: model yüklüyse önce boşaltılır, sonra dosya silinir.
#  Kullanım: hf_downloader.search_models("küçük dil modeli")
#            await hf_downloader.start_download("org/model", "model-Q4_K_M.gguf")
# ─────────────────────────────────────────────────────────────

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
from huggingface_hub import HfApi

from app.config import settings

# Varsayılan nicelik (quantization): tercihen bu desen bulunur
DEFAULT_QUANT = "Q4_K_M"
# İndirme parça boyutu (1 MB)
_CHUNK_SIZE = 1024 * 1024
# Kalıcı tek tek istek zaman aşımı (ilk bağlantı vs.)
_REQUEST_TIMEOUT = 60.0
# Katalog (browse) önbellek süresi: ilk sorgu ağır, tekrar anında
_BROWSE_TTL = 600  # saniye

# İlerleme kaydı ve aktif indirme kilidi
_downloads: dict[str, dict] = {}
_downloads_lock = threading.Lock()
_active: set[str] = set()
# Kullanıcı tarafından iptal edilen indirmeler (her parçada denetlenir)
_cancelled: set[str] = set()

# Katalog önbelleği (thread güvenli)
_browse_cache: dict | None = None
_browse_lock = threading.Lock()


class DownloadCancelledError(Exception):
    """İndirme kullanıcı tarafından iptal edildi."""


# ------------------------------------------------------------------
# Veri yapıları
# ------------------------------------------------------------------

@dataclass
class RemoteModel:
    """HuggingFace arama sonucundaki tek model kartı."""

    repo_id: str            # "org/ad-adi"
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    gguf_count: int = 0


@dataclass
class RepoFile:
    """Repo içindeki tek .gguf dosyası."""

    filename: str
    size_bytes: int


@dataclass
class DownloadResult:
    """Başarılı indirmenin özeti."""

    path: Path
    repo_id: str
    filename: str
    size_bytes: int


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _api(token: Optional[str] = None) -> HfApi:
    """HuggingFace API nesnesi (isimsiz istek için token isteğe bağlı)."""
    return HfApi(token=token or (settings.hf_token or None))


def _repo_size(repo_id: str, filename: str) -> int:
    """İndirilecek dosyanın repo bilgisinden boyutunu bulur."""
    try:
        for f in get_repo_files(repo_id):
            if f.filename == filename:
                return f.size_bytes
    except Exception:
        pass
    return 0


# ------------------------------------------------------------------
# Arama ve dosya listeleme
# ------------------------------------------------------------------

def search_models(
    query: str = "",
    category: str = "text-generation",
    limit: int = 24,
    scan_max: int = 40,
    token: Optional[str] = None,
) -> list[RemoteModel]:
    """HuggingFace'te GGUF içeren model araması yapar.

    category: pipeline etiketi (text-generation, text2text-generation vb.).
    HF, içerik kataloğunda 'gguf' kitaplık etiketi tutmadığı için her adayın
    dosya listesi denetlenir; yalnızca .gguf içeren repo'lar döner.
    """
    api = _api(token)
    candidates = list(
        api.list_models(
            search=query or None,
            pipeline_tag=category,
            sort="downloads",
            limit=scan_max,
        )
    )
    results: list[RemoteModel] = []
    for m in candidates:
        repo_id = m.modelId
        gguf_files = _repo_gguf_files(repo_id, token=token)
        if not gguf_files:
            continue
        results.append(
            RemoteModel(
                repo_id=repo_id,
                downloads=getattr(m, "downloads", 0) or 0,
                likes=getattr(m, "likes", 0) or 0,
                last_modified=getattr(m, "lastModified", "") or "",
                tags=list(getattr(m, "tags", []) or []),
                gguf_count=len(gguf_files),
            )
        )
        if len(results) >= limit:
            break
    return results


def _repo_gguf_files(repo_id: str, token: Optional[str] = None) -> list[str]:
    """Repo'daki .gguf dosya adlarını döner (yoksa / hata varsa boş liste)."""
    try:
        files = _api(token).list_repo_files(repo_id)
    except Exception:
        return []
    return [f for f in files if f.endswith(".gguf") and not f.startswith(".")]


def get_repo_files(repo_id: str, token: Optional[str] = None) -> list[RepoFile]:
    """Repo'daki .gguf dosyalarını (ad + boyut) boydan küçüğe döner."""
    info = _api(token).model_info(repo_id, files_metadata=True)
    files = [
        RepoFile(s.rfilename, s.size or 0)
        for s in info.siblings
        if s.rfilename.endswith(".gguf") and not s.rfilename.startswith(".")
    ]
    return sorted(files, key=lambda f: f.size_bytes, reverse=True)


def _repo_gguf_count(repo_id: str, token: Optional[str] = None) -> int:
    """Repodaki .gguf dosya sayısı (yoksa/hata varsa 0)."""
    return len(_repo_gguf_files(repo_id, token))


def browse_models(
    limit: int = 150,
    scan_max: int = 300,
    workers: int = 16,
    token: Optional[str] = None,
) -> list[dict]:
    """GGUF içeren modellerin geniş kataloğu (popup boş araması için).

    "gguf" adını taşıyan/etiketleyen repoları indirme sırasına göre tarar;
    repo başına dosya denetimi eşzamanlı (thread pool) yapılır. İlk sorgu
    birkaç saniye sürebilir; sonuç kısa süre önbelleğe alınır (yeniden
    açılışta anında gösterilir).
    """
    global _browse_cache
    with _browse_lock:
        if _browse_cache is not None and _browse_cache["ts"] + _BROWSE_TTL > time.time():
            return [dict(r) for r in _browse_cache["items"]]

    api = _api(token)
    candidates = list(
        api.list_models(search="gguf", sort="downloads", limit=scan_max)
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_repo_gguf_count, m.modelId, token): m for m in candidates
        }
        for fut in as_completed(future_map):
            m = future_map[fut]
            try:
                count = fut.result()
            except Exception:
                count = 0
            if count <= 0:
                continue
            results.append(
                {
                    "repo_id": m.modelId,
                    "downloads": getattr(m, "downloads", 0) or 0,
                    "likes": getattr(m, "likes", 0) or 0,
                    "last_modified": getattr(m, "lastModified", "") or "",
                    "gguf_count": count,
                    "pipeline": getattr(m, "pipeline_tag", "") or "",
                }
            )

    results.sort(key=lambda r: r["downloads"], reverse=True)
    results = results[:limit]

    with _browse_lock:
        _browse_cache = {"ts": time.time(), "items": [dict(r) for r in results]}
    return results


def pick_gguf(repo_id: str, quant: str = DEFAULT_QUANT, token: Optional[str] = None) -> RepoFile:
    """Repo'ya en uygun GGUF dosyasını seçer.

    Tercih: istenen nicelik (varsayılan Q4_K_M); yoksa en büyük dosya.
    """
    files = get_repo_files(repo_id, token=token)
    if not files:
        raise ValueError(f"'{repo_id}' repo'sunda .gguf dosyası bulunamadı")

    for f in files:
        if any(t in f.filename for t in (f"-{quant}.", f"_{quant}.", f".{quant}.")):
            return f
    return files[0]


# ------------------------------------------------------------------
# İlerleme kaydı
# ------------------------------------------------------------------

def _update_progress(repo_id: str, received: int, total: int, status: str, error: str = "") -> None:
    with _downloads_lock:
        _downloads[repo_id] = {
            "received": received,
            "total": total,
            "status": status,
            "error": error,
        }


def get_download_status(repo_id: str) -> dict:
    """Repo'nun güncel indirme durumunu döner (yoksa boş kayıt)."""
    with _downloads_lock:
        return dict(_downloads.get(repo_id, {}))


def list_active_downloads() -> dict:
    """Tüm indirme durumlarını kopyalar (panel listesi için)."""
    with _downloads_lock:
        return {k: dict(v) for k, v in _downloads.items()}


# ------------------------------------------------------------------
# Parça parça sürdürülebilir indirme çekirdeği
# ------------------------------------------------------------------

def _resumable_get(
    url: str,
    dest: Path,
    token: Optional[str],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Tek bir URL'yi Range başlığıyla indirir; kaldığı yerden devam eder.

    dest üzerinde var olan bayt sayısı kadar ileriden başlanır.
    Sunucu 200 dönerse en baştan; 206 dönerse devam; 416 verirse
    dosyanın zaten eksiksiz olduğunu kabul eder.
    cancel_check True dönerse DownloadCancelledError yükseltilir.
    """
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=_REQUEST_TIMEOUT
    ) as response:
        if response.status_code == 416:
            # Dosya zaten eksiksiz
            if progress_cb:
                progress_cb(existing, existing)
            return {"received": existing, "total": existing}

        response.raise_for_status()

        resume = response.status_code == 206
        received = existing if resume else 0
        total = received + int(response.headers.get("Content-Length") or 0)
        mode = "ab" if resume else "wb"

        with open(dest, mode) as f:
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                if cancel_check and cancel_check():
                    raise DownloadCancelledError()
                f.write(chunk)
                received += len(chunk)
                if progress_cb:
                    progress_cb(received, total)
    return {"received": received, "total": total}


# ------------------------------------------------------------------
# Kamuya açık indirme işlevleri
# ------------------------------------------------------------------

def _resolve_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """İndirilecek HF dosya URL'sini oluşturur (testlerde değiştirilebilir)."""
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def download_model(
    repo_id: str,
    filename: str,
    revision: str = "main",
    models_dir: Optional[Path | str] = None,
    token: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> DownloadResult:
    """GGUF dosyasını models/ klasörüne indirir.

    İki aşama: önce ".part" adıyla iner (yarım kalırsa model listesine
    girmez), tamamlanınca ".gguf" olarak yeniden adlandırılır. Kaldığı
    yerden devam etmeyi (resume) destekler. Aynı repo için eşzamanlı
    ikinci indirme isteği engellenir.
    """
    with _downloads_lock:
        if repo_id in _active:
            raise RuntimeError(f"'{repo_id}' için indirme zaten sürüyor")

    models_dir = Path(models_dir or settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dest = models_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    url = _resolve_url(repo_id, filename, revision)
    total_hint = _repo_size(repo_id, filename)

    _update_progress(repo_id, part.stat().st_size if part.exists() else 0, total_hint, "baslıyor")

    def _local_progress(received: int, total: int) -> None:
        _update_progress(repo_id, received, total, "indiriliyor")
        if progress_cb:
            progress_cb(received, total)

    def _cancelled_check() -> bool:
        with _downloads_lock:
            return repo_id in _cancelled

    try:
        with _downloads_lock:
            _active.add(repo_id)
        result = _resumable_get(url, part, token, _local_progress, _cancelled_check)
        # Atomik bitiş: yarım dosya model listesine girmesin
        part.replace(dest)
        _update_progress(repo_id, result["received"], result["total"], "tamam")
        return DownloadResult(
            path=dest,
            repo_id=repo_id,
            filename=filename,
            size_bytes=result["received"],
        )
    except DownloadCancelledError:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        _update_progress(repo_id, 0, 0, "iptal", "İndirme iptal edildi")
        raise
    except Exception as exc:  # ağ, disk veya sunucu hatası
        _update_progress(repo_id, 0, 0, "hata", str(exc))
        raise
    finally:
        with _downloads_lock:
            _active.discard(repo_id)
            _cancelled.discard(repo_id)


def cancel_download(repo_id: str) -> bool:
    """Devam eden indirmeyi iptal eder; yarım dosya temizlenir.

    İptal sinyali parça parça indirme döngüsünde denetlenir, böylece
    eşzamanlı çalışan indirme thread'i kısa sürede sonlanır (status: iptal).
    """
    with _downloads_lock:
        status_ = _downloads.get(repo_id, {}).get("status", "")
        if repo_id not in _active and status_ not in ("baslıyor", "indiriliyor"):
            return False
        _cancelled.add(repo_id)
    return True


async def start_download(
    repo_id: str,
    filename: str,
    revision: str = "main",
    models_dir: Optional[Path | str] = None,
    token: Optional[str] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> DownloadResult:
    """FastAPI/panel çağrısı için indirmeyi arka planda (thread) yürütür."""
    return await asyncio.to_thread(
        download_model, repo_id, filename, revision, models_dir, token, progress_cb
    )


# ------------------------------------------------------------------
# Model silme
# ------------------------------------------------------------------

def delete_model_files(model_id: str, models_dir: Optional[Path | str] = None) -> list[str]:
    """Diskteki model dosyalarını (ve varsa yarım indirmesini) siler.

    model_id, model yöneticisindeki dosya gövdesiyle aynıdır (dosya adı
    minüş .gguf). Sistemde ".gguf" eşleşmesine göre temizler.
    """
    models_dir = Path(models_dir or settings.models_dir)
    if not models_dir.exists():
        return []
    removed: list[str] = []
    for p in list(models_dir.rglob("*.gguf")) + list(models_dir.rglob("*.part")):
        if p.stem == model_id or p.name == f"{model_id}.gguf.part":
            try:
                p.unlink()
                removed.append(p.name)
            except FileNotFoundError:
                pass
    return removed


async def delete_model(
    model_id: str,
    manager=None,
    models_dir: Optional[Path | str] = None,
) -> list[str]:
    """Modeli yönetikten boşaltıp dosyalarını siler (panel işlemi).

    manager verilmezse uygulamanın tekil yöneticisi kullanılır; model
    bellekteyse önce boşaltılır, ardından diski temizlenir.
    """
    if manager is None:
        from app.model_manager import get_manager

        manager = get_manager()
    await manager.unload(model_id)  # yüklüyse VRAM/RAM bırakır
    return delete_model_files(model_id, models_dir=models_dir)