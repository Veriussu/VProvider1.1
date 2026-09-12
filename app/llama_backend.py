# ─────────────────────────────────────────────────────────────
#  Bölüm:    LLM Motoru
#  Dosya:    app/llama_backend.py
#  Amaç:     GGUF modelini çalıştıran motorun arayüzünü (sözleşme) ve
#            gerçek llama.cpp uygulamasını sunar.
#  Mekanik:  - LlamaEngine: tüm motorların uyması gereken soyut arayüz.
#            - LlamaCppEngine: llama-cpp-python üzerinden gerçek modeli
#              yükler; chat ve completion çağrılarını (normal + akışlı)
#              çalıştırır, OpenAI parametrelerini (temperature, max_tokens,
#              top_p, stop) geçirir.
#            - last_usage: son çağrının token kullanım bilgisini tutar
#              (OpenAI yanıtındaki "usage" alanı için).
#            - unload: model nesnesini siler, GC çağırır -> VRAM/RAM bırakılır.
#            - create_engine: llama-cpp-python kurulu değilse anlaşılır hata verir.
#  Kullanım: engine = create_engine(model_yolu, context_size=..., gpu_layers=...)
# ─────────────────────────────────────────────────────────────

import gc
import json
import logging
import os
from typing import Iterable, Optional

from app.gpu_detect import resolve_runtime

logger = logging.getLogger("vprovider")


class LlamaEngine:
    """Model motorları için arayüz (sözleşme).

    Gerçek llama.cpp motoru bu yöntemleri uygular; model_manager
    motorun iç ayrıntısını bilmeden yalnızca bu arayüzü kullanır.
    """

    def load(self) -> None:
        """Modeli belleğe yükler (VRAM/RAM kullanımına hazırlar)."""
        raise NotImplementedError

    def unload(self) -> None:
        """Modeli bellekten tamamen boşaltır (0 MB VRAM hedefi)."""
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        """Modelin o anda bellekte olup olmadığını döner."""
        raise NotImplementedError

    def run_chat(self, messages: list, **params) -> str:
        """Sohbet (chat) yanıtını akışsız üretir. **params: OpenAI parametreleri."""
        raise NotImplementedError

    def stream_chat(self, messages: list, **params) -> Iterable[str]:
        """Sohbet (chat) yanıtını parça parça (chunk) üretir."""
        raise NotImplementedError

    def tool_call_info(self):
        """Son chat çağrısının (araç, bitiş nedeni) bilgisini döner.

        Dönüş: ((tool_calls listesi | None), finish_reason). Tool çağrısı
        yapılmadıysa ilk değer None olur. Motor bilmiyorsa varsayılan
        (None, 'stop') döner; böylece eski motorlar da çalışır.
        """
        return None, "stop"

    def run_completion(self, prompt: str, **params) -> str:
        """Metin tamamlama (completion) yanıtını akışsız üretir."""
        raise NotImplementedError

    def stream_completion(self, prompt: str, **params) -> Iterable[str]:
        """Metin tamamlama (completion) yanıtını parça parça üretir."""
        raise NotImplementedError

    def usage_info(self) -> dict:
        """Son çağrının token kullanım bilgisini döner (boş olabilir)."""
        return {}


class LlamaCppEngine(LlamaEngine):
    """llama-cpp-python tabanlı gerçek GGUF motoru."""

    def __init__(
        self,
        model_path: str,
        context_size: int = 4096,
        gpu_layers: int = -1,
        threads: int = 0,
        gpu_mode: str = "auto",
        runtime_note: str = "",
    ) -> None:
        self.model_path = model_path
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.gpu_mode = gpu_mode
        self.runtime_note = runtime_note
        self.fallback_note = ""
        self.threads = threads
        self._llama = None        # llama_cpp.Llama örneği; yüklenene kadar None
        self.last_usage: dict = {}
        self.last_tool_calls = None              # son chat'te üretilen araç çağrıları
        self.last_finish_reason: str = "stop"    # son chat'in bitiş nedeni

    def load(self) -> None:
        """Modeli belleğe yükler. Aynı motor ikinci kez yüklenirse hızlıca döner.

        GPU offload denemesi başarısız olursa (ör. VRAM yetersiz) ve mod
        'auto' ise otomatik olarak CPU'ya (gpu_layers=0) düşülür ve uyarı
        fallback_note'a yazılır.
        """
        if self._llama is not None:
            return
        from llama_cpp import Llama  # yalnızca bu anda içe aktarılır

        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.context_size,
            "n_gpu_layers": self.gpu_layers,
            "n_threads": self.threads if self.threads > 0 else None,
            "verbose": False,
        }
        try:
            self._llama = Llama(**kwargs)
        except Exception as exc:  # noqa: BLE001
            auto_can_fallback = self.gpu_mode == "auto" and self.gpu_layers not in (0, -1)
            if not auto_can_fallback:
                raise
            logger.warning(
                "GPU offload başarısız (%s). CPU (gpu_layers=0) ile yeniden deneniyor: %s",
                exc, self.runtime_note,
            )
            kwargs["n_gpu_layers"] = 0
            self.gpu_layers = 0
            self.fallback_note = (
                f"GPU offload başarısızdı ({type(exc).__name__}); CPU'ya geçildi."
            )
            self._llama = Llama(**kwargs)

    def unload(self) -> None:
        """Modeli bellekten boşaltır; VRAM/RAM içeriği serbest bırakılır."""
        if self._llama is None:
            return
        del self._llama
        self._llama = None
        self.last_usage = {}
        self.last_tool_calls = None
        self.last_finish_reason = "stop"
        # Bekleyen nesneleri topla: bellek (VRAM dahil) anında boşalır
        gc.collect()

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    def usage_info(self) -> dict:
        """Son çağrının token kullanımını döner."""
        return self.last_usage

    def tool_call_info(self):
        """Son chat çağrısının araç çağrılarını ve bitiş nedenini döner."""
        return self.last_tool_calls, self.last_finish_reason

    @staticmethod
    def _normalize_tool_calls(tool_calls):
        """llama.cpp tool_calls çıktısını OpenAI biçiminde normalleştirir.

        arguments yalnızca string olmalıdır; dick/lista gelirse JSON'a çevrilir.
        """
        if not tool_calls:
            return None
        normalized = []
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments") or ""
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            normalized.append(
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {"name": fn.get("name"), "arguments": str(args)},
                }
            )
        return normalized

    # ------------------------------------------------------------------
    # Chat (sohbet)
    # ------------------------------------------------------------------

    def run_chat(self, messages: list, **params) -> str:
        """Akışsız sohbet: mesaj dizisini işler, tam yanıt metnini döner.

        Model araç (tool) çağrısı yaptıysa içerik boş olabilir; araç çağrısı
        bilgisini tool_call_info() ile alınır.
        """
        if self._llama is None:
            raise RuntimeError("Model yüklenmemiş; önce load() çağrılmalı.")
        result = self._llama.create_chat_completion(messages=messages, stream=False, **params)
        self.last_usage = result.get("usage", {})
        choice = result["choices"][0]
        message = choice.get("message", {})
        self.last_finish_reason = choice.get("finish_reason", "stop")
        self.last_tool_calls = self._normalize_tool_calls(message.get("tool_calls"))
        content = message.get("content") or ""
        return content if isinstance(content, str) else str(content)

    def stream_chat(self, messages: list, _yield_events: bool = False, **params) -> Iterable:
        """Akışlı sohbet: yanıt parçalarını sırayla üretir.

        Varsayılan mod: yalnızca içerik parçaları (str). _yield_events=True
        ise her parça sözlük olayı olarak verilir:
            {"type":"content","text":...}
            {"type":"tool_args","index":i,"id":...,"name":...,"delta":...}
            {"type":"finish","reason":"stop|tool_calls|length"}
            {"type":"usage","usage":{...}}
        Bu mod, OpenAI /v1/responses akışı gibi araç çağrısı farkında akışlar
        kurmak için kullanılır. Her iki modda da son çağrının araç bilgisi
        tool_call_info() üzerinden alınabilir.
        """
        if self._llama is None:
            raise RuntimeError("Model yüklenmemiş; önce load() çağrılmalı.")
        stream = self._llama.create_chat_completion(messages=messages, stream=True, **params)

        acc: dict = {}          # index -> {id, name, args} araç çağrısı toplama
        finish_reason: str = "stop"
        usage: dict = {}
        for chunk in stream:
            if chunk.get("usage"):
                self.last_usage = chunk["usage"]
                usage = chunk["usage"]
            choice = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
            if not choice:
                continue
            delta = choice.get("delta") or {}
            part = delta.get("content")
            for td in delta.get("tool_calls") or []:
                idx = td.get("index", 0)
                rec = acc.setdefault(idx, {"id": None, "name": None, "args": ""})
                fn = td.get("function") or {}
                if td.get("id"):
                    rec["id"] = td["id"]
                if fn.get("name"):
                    rec["name"] = fn["name"]
                if fn.get("arguments"):
                    rec["args"] += str(fn["arguments"])
                if _yield_events:
                    yield {
                        "type": "tool_args",
                        "index": idx,
                        "id": rec.get("id"),
                        "name": rec.get("name"),
                        "delta": str(fn.get("arguments") or ""),
                    }
            if part:
                text = part if isinstance(part, str) else str(part)
                if _yield_events:
                    yield {"type": "content", "text": text}
                else:
                    yield text
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        self.last_finish_reason = finish_reason
        if acc:
            self.last_tool_calls = [
                {
                    "id": rec.get("id"),
                    "type": "function",
                    "function": {"name": rec.get("name"), "arguments": rec.get("args")},
                }
                for idx, rec in sorted(acc.items())
            ]
        else:
            self.last_tool_calls = None

        if _yield_events:
            yield {"type": "finish", "reason": finish_reason}
            if usage:
                yield {"type": "usage", "usage": usage}

    # ------------------------------------------------------------------
    # Completion (metin tamamlama)
    # ------------------------------------------------------------------

    def run_completion(self, prompt: str, **params) -> str:
        """Akışsız completion: düz metni işler, tam yanıt metnini döner."""
        if self._llama is None:
            raise RuntimeError("Model yüklenmemiş; önce load() çağrılmalı.")
        result = self._llama.create_completion(prompt=prompt, stream=False, **params)
        self.last_usage = result.get("usage", {})
        return result["choices"][0].get("text") or ""

    def stream_completion(self, prompt: str, **params) -> Iterable[str]:
        """Akışlı completion: yanıt parçalarını sırayla üretir."""
        if self._llama is None:
            raise RuntimeError("Model yüklenmemiş; önce load() çağrılmalı.")
        stream = self._llama.create_completion(prompt=prompt, stream=True, **params)
        for chunk in stream:
            if chunk.get("usage"):
                self.last_usage = chunk["usage"]
            part = chunk.get("choices", [{}])[0].get("text")
            if part:
                yield part


def create_engine(model_path: str, **options) -> LlamaEngine:
    """Gerçek llama.cpp motorunu kurar.

    llama-cpp-python kurulu değilse (henüz install.sh çalışmamışsa)
    anlaşılır bir hata verir; kuruluysa LLM GPU algılamasına göre
    nihai gpu_layers değerini hesaplayıp LlamaCppEngine döner.

    Özel seçenekler (LlamaCppEngine'e geçmez):
      gpu_mode          -> auto|cuda|rocm|sycl|vulkan|cpu (.env GPU_MODE)
      model_size_bytes  -> VRAM hesapları için model dosya boyutu
    """
    try:
        import llama_cpp  # noqa: F401  (kurulu mu?)
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python kurulu değil. Donanıma özel derleme için "
            "install.sh betiğini çalıştırın (veya CPU için: "
            "pip install llama-cpp-python)."
        ) from exc

    gpu_mode = options.pop("gpu_mode", "auto")
    model_size_bytes = options.pop("model_size_bytes", 0)
    if not model_size_bytes:
        try:
            model_size_bytes = os.path.getsize(model_path)
        except OSError:
            model_size_bytes = 0
    requested_layers = options.get("gpu_layers", -1)

    runtime = resolve_runtime(
        gpu_mode=gpu_mode,
        requested_layers=requested_layers,
        model_path=model_path,
        model_size_bytes=model_size_bytes,
        n_ctx=options.get("context_size", 4096),
    )
    options["gpu_layers"] = runtime.gpu_layers
    logger.info("GPU kararı: %s", runtime.note)
    logger.info(
        "Donanım: %s | Derlenmiş backend: %s | Kullanılacak backend: %s",
        runtime.hardware.name or runtime.hardware.vendor,
        ", ".join(runtime.compiled_backends),
        runtime.backend,
    )
    engine = LlamaCppEngine(
        model_path,
        context_size=options.get("context_size", 4096),
        gpu_layers=runtime.gpu_layers,
        threads=options.get("threads", 0),
        gpu_mode=gpu_mode,
        runtime_note=runtime.note,
    )
    return engine