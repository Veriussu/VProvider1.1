# ─────────────────────────────────────────────────────────────
#  Bölüm:    Model Yöneticisi
#  Dosya:    app/model_manager.py
#  Amaç:     GGUF modellerini yönetir: disk taraması, belleğe yükleme/
#            boşaltma, bellek modları (keep/dynamic) ve model başına
#            eşzamanlılık kontrolü.
#  Mekanik:  - Her model, diskte bulunduğu .gguf dosyasından tanınır.
#            - Her model için bir asyncio.Lock tutulur; aynı modelin
#              istekleri seri işlenir (llama.cpp tek bağlamı paylaşır).
#            - keep modu: model ilk istekte yüklenir, kalıcı kalır.
#            - dynamic modu: model istek üzerine yüklenir; işlem bitince
#              idle_süresi başlar, süre dolunca arka plan görevi modeli
#              tamamen boşaltır (0 MB VRAM).
#            - Motor ayrıntısına bağımlı değildir: engine_factory
#              parametresiyle sahte motor verilerek test edilebilir.
#  Kullanım: manager = get_manager()  -> tek örnek
# ─────────────────────────────────────────────────────────────

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from app.config import settings
from app.llama_backend import LlamaEngine

# Motor üretim fonksiyonu: ModelInfo alır, LlamaEngine döner
EngineFactory = Callable[["ModelInfo"], LlamaEngine]


@dataclass
class ModelInfo:
    """Diskteki tek bir GGUF modelinin bilgisi."""

    model_id: str            # klasörden benzersiz tanımlayıcı (dosya adı)
    path: Path               # .gguf dosyasının tam yolu
    size_bytes: int          # dosya boyutu
    loaded: bool = False     # o anda bellekte mi
    memory_mode: str = ""    # aktif bellek modu (boşsa küresel mod geçerli)


class ModelManager:
    """Modellerin yükleme/boşaltma ve bellek stratejisini yürütür."""

    MEMORY_KEEP = "keep"          # her daim bellekte
    MEMORY_DYNAMIC = "dynamic"    # boşta boşalt

    def __init__(
        self,
        models_dir: str | Path,
        engine_factory: Optional[EngineFactory] = None,
        memory_mode: str = "",
        idle_timeout_minutes: int = 0,
    ) -> None:
        self.models_dir = Path(models_dir)
        # Varsayılan motor üretici: llama_backend.create_engine
        self._factory = engine_factory or self._default_factory
        self.memory_mode = memory_mode or settings.memory_mode
        self.idle_timeout_minutes = idle_timeout_minutes or settings.idle_timeout_minutes

        self._engines: dict[str, LlamaEngine] = {}   # model_id -> çalışan motor
        self._locks: dict[str, asyncio.Lock] = {}    # model_id -> kilit
        self._modes: dict[str, str] = {}             # model_id -> özel mod
        self._idle_tasks: dict[str, asyncio.Task] = {}  # model_id -> bekleme görevi

    # ------------------------------------------------------------------
    # Motor üretimi (varsayılan)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_factory(info: ModelInfo) -> LlamaEngine:
        """Gerçek llama.cpp motorunu üretir."""
        from app import llama_backend

        return llama_backend.create_engine(
            str(info.path),
            context_size=settings.context_size,
            gpu_layers=settings.gpu_layers,
            threads=settings.threads,
        )

    # ------------------------------------------------------------------
    # Disk taraması
    # ------------------------------------------------------------------

    def scan_models(self) -> list[ModelInfo]:
        """models/ klasöründeki tüm .gguf dosyalarını tarar."""
        infos: list[ModelInfo] = []
        for p in sorted(self.models_dir.rglob("*.gguf")):
            info = ModelInfo(
                model_id=p.stem,
                path=p,
                size_bytes=p.stat().st_size,
                loaded=self._is_loaded(p.stem),
                memory_mode=self._modes.get(p.stem, self.memory_mode),
            )
            infos.append(info)
        return infos

    def list_models(self) -> list[ModelInfo]:
        """Panel için kullanılan model listesi (tarama sonucu)."""
        return self.scan_models()

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Tek bir modeli bulur; yoksa None döner."""
        for info in self.scan_models():
            if info.model_id == model_id:
                return info
        return None

    # ------------------------------------------------------------------
    # Kilitler ve motor kaydı
    # ------------------------------------------------------------------

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        """Model başına bir asyncio.Lock döner (ilk çağrıda üretir)."""
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    def _is_loaded(self, model_id: str) -> bool:
        engine = self._engines.get(model_id)
        return engine is not None and engine.is_loaded

    def _mode_for(self, model_id: str) -> str:
        return self._modes.get(model_id, self.memory_mode)

    # ------------------------------------------------------------------
    # Yükleme / boşaltma
    # ------------------------------------------------------------------

    async def _load_engine(self, model_id: str) -> LlamaEngine:
        """Model motorunu oluşturur ve belleğe yükler (kilit zaten tutulmalı)."""
        info = self.get_model(model_id)
        if info is None:
            raise ValueError(f"'{model_id}' adında bir model bulunamadı")

        engine = self._engines.get(model_id)
        if engine is not None and engine.is_loaded:
            return engine

        # Ağır işlem (model kurulumu) olay döngüsünü bloklamasın
        engine = self._factory(info)
        await asyncio.to_thread(engine.load)
        self._engines[model_id] = engine

        # Yalnızca pozitif idle süresi olan dynamic modda yüklemede bekleyen
        # boşaltma zamanlanır; 0 (anında boşalt) ise işlem bitince _release_after_use
        # eliyle boşaltılır (yükleme sırasında planlamak yarışa yol açardı).
        if (
            self._mode_for(model_id) == self.MEMORY_DYNAMIC
            and self.idle_timeout_minutes > 0
        ):
            self._reschedule_idle(model_id)
        return engine

    async def load(self, model_id: str, memory_mode: str = "") -> bool:
        """Modeli açıkça belleğe yükler (panelden çağrılır).

        memory_mode verilirse o model için geçerli mod olur.
        """
        if memory_mode:
            self._modes[model_id] = memory_mode
        async with self._lock_for(model_id):
            await self._load_engine(model_id)
        return True

    async def unload(self, model_id: str) -> bool:
        """Modeli bellekten boşaltır (varsa idle görevini iptal eder)."""
        self._cancel_idle(model_id)
        async with self._lock_for(model_id):
            engine = self._engines.pop(model_id, None)
            if engine is None:
                return False
            # Ağır işlem (VRAM/RAM bırakma) olay döngüsünü bloklamasın
            await asyncio.to_thread(engine.unload)
        return True

    def set_memory_mode(self, model_id: str, memory_mode: str) -> None:
        """Tek bir modelin bellek modunu değiştirir (keep/dynamic)."""
        if memory_mode not in (self.MEMORY_KEEP, self.MEMORY_DYNAMIC):
            raise ValueError("memory_mode yalnızca 'keep' veya 'dynamic' olabilir")
        self._modes[model_id] = memory_mode

        if memory_mode == self.MEMORY_KEEP:
            # Kalıcı mod: bekleyen boşaltıcı görevi iptal edilir
            self._cancel_idle(model_id)
        elif self._is_loaded(model_id):
            # Dinamik moda geçildi -> boşaltıcı tekrar zamanlanır
            self._release_after_use(model_id)

    def set_global_memory_mode(self, memory_mode: str) -> None:
        """Varsayılan (küresel) bellek modunu değiştirir."""
        if memory_mode not in (self.MEMORY_KEEP, self.MEMORY_DYNAMIC):
            raise ValueError("memory_mode yalnızca 'keep' veya 'dynamic' olabilir")
        self.memory_mode = memory_mode

    # ------------------------------------------------------------------
    # Çağrı (chat)
    # ------------------------------------------------------------------

    async def chat(self, model_id: str, messages: list, **params) -> str:
        """Akışsız yanıt üretir; model gerektiğinde yüklenir.

        **params: OpenAI parametreleri (temperature, max_tokens, top_p, stop...)
        """
        async with self._lock_for(model_id):
            engine = await self._ensure_loaded_locked(model_id)
            try:
                answer = await asyncio.to_thread(engine.run_chat, messages, **params)
            finally:
                self._release_after_use(model_id)
        return answer

    async def chat_stream(self, model_id: str, messages: list, **params) -> AsyncIterator[str]:
        """Akışlı (stream) yanıt üretir; aynı modelde başka istek varsa bekler."""
        async with self._lock_for(model_id):
            engine = await self._ensure_loaded_locked(model_id)
            queue: asyncio.Queue = asyncio.Queue()

            def produce() -> None:
                """Senkron üreticiyi kuyruğa taşır (ayrı iş parçacığında çalışır)."""
                try:
                    for chunk in engine.stream_chat(messages, **params):
                        queue.put_nowait(chunk)
                finally:
                    queue.put_nowait(None)  # bitiş işareti

            producer = asyncio.get_running_loop().run_in_executor(None, produce)
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                await producer
                self._release_after_use(model_id)

    async def completion(self, model_id: str, prompt: str, **params) -> str:
        """Akışsız metin tamamlama (completion) üretir."""
        async with self._lock_for(model_id):
            engine = await self._ensure_loaded_locked(model_id)
            try:
                answer = await asyncio.to_thread(engine.run_completion, prompt, **params)
            finally:
                self._release_after_use(model_id)
        return answer

    async def completion_stream(self, model_id: str, prompt: str, **params) -> AsyncIterator[str]:
        """Akışlı metin tamamlama (completion) üretir."""
        async with self._lock_for(model_id):
            engine = await self._ensure_loaded_locked(model_id)
            queue: asyncio.Queue = asyncio.Queue()

            def produce() -> None:
                try:
                    for chunk in engine.stream_completion(prompt, **params):
                        queue.put_nowait(chunk)
                finally:
                    queue.put_nowait(None)

            producer = asyncio.get_running_loop().run_in_executor(None, produce)
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                await producer
                self._release_after_use(model_id)

    async def usage(self, model_id: str) -> dict:
        """Son çağrının token kullanım bilgisini döner (üretilmemişse boş)."""
        engine = self._engines.get(model_id)
        return engine.usage_info() if engine is not None else {}

    async def tool_call_info(self, model_id: str):
        """Son chat çağrısının (araç çağrıları, bitiş nedeni) bilgisini döner.

        Motorun arayüzü tool_call_info sağlamıyorsa varsayılan (None, 'stop')
        döner; böylece eski/sahte motorlar da güvenle çalışır.
        """
        engine = self._engines.get(model_id)
        if engine is None:
            return None, "stop"
        getter = getattr(engine, "tool_call_info", None)
        if getter is None:
            return None, "stop"
        return getter()

    async def chat_stream_events(
        self, model_id: str, messages: list, **params
    ) -> AsyncIterator[dict]:
        """Akışlı sohbeti olay (event) biçiminde üretir.

        Motorun stream_chat'i _yield_events=True ile çağrılır; her parça
        sözlük olarak döner (content/tool_args/finish/usage). Eski motorlar
        string parçalar dönerse olduğu gibi aktarılır.
        """
        async with self._lock_for(model_id):
            engine = await self._ensure_loaded_locked(model_id)
            queue: asyncio.Queue = asyncio.Queue()

            def produce() -> None:
                try:
                    for ev in engine.stream_chat(messages, _yield_events=True, **params):
                        queue.put_nowait(ev)
                finally:
                    queue.put_nowait(None)  # bitiş işareti

            producer = asyncio.get_running_loop().run_in_executor(None, produce)
            try:
                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    yield ev
            finally:
                await producer
                self._release_after_use(model_id)

    async def _ensure_loaded_locked(self, model_id: str) -> LlamaEngine:
        """Modelin yüklü olduğundan emin olur (kilit zaten tutulmalı)."""
        engine = self._engines.get(model_id)
        if engine is not None and engine.is_loaded:
            return engine
        return await self._load_engine(model_id)

    def _release_after_use(self, model_id: str) -> None:
        """Dynamic modda işlem bittikten sonra boşaltmayı düzenler.

        idle_timeout_minutes > 0  -> zamanlayıcı sıfırlanır (boşta kalınca boşalt).
        idle_timeout_minutes = 0  -> kullanım biter bitmez GPU'dan anında boşaltılır;
        böylece tek GPU'da modelden modele geçişte VRAM hemen açılır.
        """
        if self._mode_for(model_id) != self.MEMORY_DYNAMIC:
            return
        self._cancel_idle(model_id)
        if self.idle_timeout_minutes > 0:
            self._reschedule_idle(model_id)
        else:
            asyncio.get_running_loop().create_task(self._deferred_unload(model_id))

    async def _deferred_unload(self, model_id: str) -> None:
        """Kilit bırakıldıktan sonra modeli boşaltır (anında boşaltma yolu)."""
        await asyncio.sleep(0)  # çağıranın kilidini bırakması için bir adım bekle
        # Hâlâ yüklü ve dinamik moddaysa boşalt (arada keep'e geçilmiş olabilir)
        if self._is_loaded(model_id) and self._mode_for(model_id) == self.MEMORY_DYNAMIC:
            await self.unload(model_id)

    # ------------------------------------------------------------------
    # Dinamik boşaltma (idle) zamanlayıcısı
    # ------------------------------------------------------------------

    def _reschedule_idle(self, model_id: str) -> None:
        """Boşaltma süresini sıfırlar (yeni bir bekleme görevi başlatır)."""
        self._cancel_idle(model_id)
        task = asyncio.create_task(self._idle_unload(model_id))
        self._idle_tasks[model_id] = task

    def _cancel_idle(self, model_id: str) -> None:
        """Varsa bekleyen boşaltma görevini iptal eder."""
        task = self._idle_tasks.pop(model_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _idle_unload(self, model_id: str) -> None:
        """Idle süresi dolunca modeli bellekten boşaltır."""
        try:
            await asyncio.sleep(self.idle_timeout_minutes * 60)
        except asyncio.CancelledError:
            return  # yeni istek geldi, iptal edildi
        # Bu görev tetiklendi: önce defterden kendini çıkar, sonra boşalt.
        # (Kendi görevi hâlâ defterdeyse unload onu da iptal eder -> yarım kalır.)
        self._idle_tasks.pop(model_id, None)
        # Hâlâ yüklü ve dinamik moddaysa boşalt
        if self._is_loaded(model_id) and self._mode_for(model_id) == self.MEMORY_DYNAMIC:
            await self.unload(model_id)

    # ------------------------------------------------------------------
    # Kapanış
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Tüm idle görevlerini iptal eder ve modelleri boşaltır."""
        for task in list(self._idle_tasks.values()):
            if not task.done():
                task.cancel()
        self._idle_tasks.clear()
        # Çoğaltmadan güvenli bir şekilde tüm modelleri boşalt
        for model_id in list(self._engines.keys()):
            await self.unload(model_id)


# ------------------------------------------------------------------
# Singleton erişimi: uygulama genelinde tek model yöneticisi.
# ------------------------------------------------------------------

_default_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Uygulama genelinde tek ModelManager örneğini döner."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager(
            models_dir=settings.models_dir,
            memory_mode=settings.memory_mode,
            idle_timeout_minutes=settings.idle_timeout_minutes,
        )
    return _default_manager