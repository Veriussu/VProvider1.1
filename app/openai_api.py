# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu API
#  Dosya:    app/openai_api.py
#  Amaç:     OpenAI standartlarında /v1/* uç noktalarını sunar.
#  Mekanik:  - Tüm uç noktalar API anahtarı (Bearer) zorunludur.
#            - GET /v1/models      -> yüklü modellerin listesi
#            - POST /v1/chat/completions -> sohbet (akışsız / SSE)
#            - POST /v1/completions      -> metin tamamlama (akışsız / SSE)
#            - Yanıtlar OpenAI nesne biçiminde döner; hatalar da
#              OpenAI hata yapısıyla (detail.error) verilir.
#            - Akışta parçalar "data: {...}" satırları + "[DONE]" olur.
#  Kullanım: app/main.py içinde app.include_router(openai_router)
# ─────────────────────────────────────────────────────────────

import json
import time
from typing import AsyncIterator, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.model_manager import get_manager

# OpenAI uyumlu router; tüm /v1/* istekleri API anahtarı ister
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

# Akış (SSE) yanıt başlıkları
STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ------------------------------------------------------------------
# İstek modelleri (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Chat mesajı: rol + içerik (+ araç çağrısı alanları).

    Araç çağrısı senaryolarında:
      - assistant mesajları tool_calls (dize önceki çağrı) taşıyabilir.
      - tool mesajları tool_call_id ile hangi çağrıya cevap olduğunu söyler.
    """

    role: str
    content: str | list | None = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list] = None


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions istek gövdesi."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: str | list[str] | None = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[list] = None
    tool_choice: str | dict | None = None
    response_format: Optional[dict] = None
    parallel_tool_calls: Optional[bool] = None
    stream_options: Optional[dict] = None
    model_config = {"extra": "ignore"}


class CompletionRequest(BaseModel):
    """POST /v1/completions istek gövdesi."""

    model: str
    prompt: str
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: str | list[str] | None = None


class ResponsesRequest(BaseModel):
    """POST /v1/responses istek gövdesi (OpenAI Responses API).

    OpenAI'ın responses şemasındaki gerekli alanları kapsar; Codex CLI gibi
    araçların standart gövdesi ile çalışır. Bilinmeyen alanlar yoksayılır
    (extra="ignore") böylece yeni şema alanları 422 hata üretmez.
    """

    model_config = {"extra": "ignore"}

    model: str
    input: str | list | dict | None = None
    instructions: Optional[str] = None
    tools: Optional[list] = None
    tool_choice: str | dict | None = None
    stream: bool = False
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[dict] = None
    stream_options: Optional[dict] = None
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None
    include: Optional[list] = None
    text: Optional[dict] = None
    service_tier: Optional[str] = None
    metadata: Optional[dict] = None
    user: Optional[str] = None


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _openai_error(status: int, message: str, param: str | None = None, code: str | None = None):
    """OpenAI hata yapısında HTTPException üretir."""
    return HTTPException(
        status_code=status,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


def _require_model(model_id: str) -> None:
    """Model diskte yoksa OpenAI uyumlu 404 hatası verir."""
    model = get_manager().get_model(model_id)
    if model is None:
        raise _openai_error(
            404,
            f"Aradığınız '{model_id}' model bulunamadı. Panelden indirin veya doğru adı girin.",
            param="model",
            code="model_not_found",
        )


def _filter_params(req: BaseModel) -> dict:
    """İstekte dolu olan OpenAI parametrelerini sözlüğe ayıklar.

    max_completion_tokens (yeni ad) max_tokens'a çevrilir; üzerine yazarsa
    aynı istekte ikisi birden verilmiş demektir, yeni ad üstün tutulur.
    """
    params = {}
    for name in (
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "tools",
        "tool_choice",
        "response_format",
    ):
        value = getattr(req, name, None)
        if value is not None:
            params[name] = value
    if getattr(req, "max_completion_tokens", None) is not None:
        params["max_tokens"] = req.max_completion_tokens
    return params


def _chat_response(
    model_id: str,
    content: str,
    usage: dict,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> dict:
    """Akışsız chat yanıtını OpenAI biçiminde kurar.

    Araç çağrısı (tool_calls) verilirse mesaja eklenir ve bitiş nedeni
    'tool_calls' olur; içerik OpenAI'da olduğu gibi boş (None) kalabilir.
    """
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _completion_response(model_id: str, content: str, usage: dict) -> dict:
    """Akışsız completion yanıtını OpenAI biçiminde kurar."""
    return {
        "id": f"cmpl-{uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _sse_chat_chunk(stream_id: str, model_id: str, content: str, finish: str | None) -> str:
    """Akışlı chat yanıtının tek parçasını SSE biçiminde üretir.

    Türkçe karakterler için ensure_ascii=False kullanılır; akış boyunca aynı
    akış kimliği (stream_id) korunur.
    """
    payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_chat_chunk_full(stream_id: str, model_id: str, delta: dict, finish: str | None) -> str:
    """Akışlı chat yanıtında serbest delta (örn. tool_calls) içeren parça üretir."""
    payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_completion_chunk(stream_id: str, model_id: str, content: str, finish: str | None) -> str:
    """Akışlı completion yanıtının tek parçasını SSE biçiminde üretir."""
    payload = {
        "id": stream_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": finish,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage_payload(usage: dict) -> dict:
    """Kullanım bilgisini olabildiğince OpenAI biçimine dönüştürür."""
    if not usage:
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# ------------------------------------------------------------------
# Model listesi
# ------------------------------------------------------------------

@router.get("/models")
def list_models():
    """Diskteki tüm GGUF modellerini OpenAI /v1/models biçiminde listeler."""
    data = []
    for model in get_manager().list_models():
        data.append(
            {
                "id": model.model_id,
                "object": "model",
                "created": int(model.path.stat().st_mtime),
                "owned_by": "vprovider",
            }
        )
    return {"object": "list", "data": data}


# ------------------------------------------------------------------
# Chat completions
# ------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Sohbet yanıtı üretir; stream=True ise SSE akışı döner."""
    _require_model(req.model)
    params = _filter_params(req)
    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _chat_stream_generator(
                req.model,
                messages,
                params,
                include_usage=bool((req.stream_options or {}).get("include_usage")),
            ),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    content = await get_manager().chat(req.model, messages, **params)
    usage = _usage_payload(await get_manager().usage(req.model))
    tool_calls, finish = await get_manager().tool_call_info(req.model)
    return _chat_response(req.model, content, usage, tool_calls=tool_calls, finish_reason=finish)


async def _chat_stream_generator(
    model_id: str, messages: list, params: dict, include_usage: bool = False
) -> AsyncIterator[str]:
    """Chat akışını OpenAI SSE biçiminde parça parça üretir.

    İçerik parçalarının yanı sıra araç çağrısı (tool_calls) delta'larını da
    iletir; bitiş parçasında finish_reason ve (istenirse) kullanım bilgisi
    verilir.
    """
    stream_id = f"chatcmpl-{uuid4().hex[:12]}"
    finish_reason = "stop"
    usage_payload: dict = {}
    async for ev in get_manager().chat_stream_events(model_id, messages, **params):
        if isinstance(ev, str):
            # Geriye dönük uyum: eski/sahte motorlar doğrudan metin verir.
            yield _sse_chat_chunk(stream_id, model_id, ev, None)
            continue
        kind = ev.get("type")
        if kind == "content":
            yield _sse_chat_chunk(stream_id, model_id, ev.get("text", ""), None)
        elif kind == "tool_args":
            delta = {
                "tool_calls": [
                    {
                        "index": ev["index"],
                        "id": ev.get("id"),
                        "type": "function",
                        "function": {
                            "name": ev.get("name"),
                            "arguments": ev.get("delta", ""),
                        },
                    }
                ]
            }
            yield _sse_chat_chunk_full(stream_id, model_id, delta, None)
        elif kind == "finish":
            finish_reason = ev.get("reason", "stop")
        elif kind == "usage":
            if include_usage:
                usage_payload = _usage_payload(ev.get("usage", {}))

    if finish_reason == "tool_calls":
        yield _sse_chat_chunk_full(stream_id, model_id, {}, "tool_calls")
    else:
        yield _sse_chat_chunk(stream_id, model_id, "", "stop")
    if include_usage and usage_payload:
        payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "usage": usage_payload,
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------
# Completions (metin tamamlama)
# ------------------------------------------------------------------

@router.post("/completions")
async def completions(req: CompletionRequest):
    """Metin tamamlama yanıtı üretir; stream=True ise SSE akışı döner."""
    _require_model(req.model)
    params = _filter_params(req)

    if req.stream:
        return StreamingResponse(
            _completion_stream_generator(req.model, req.prompt, params),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    content = await get_manager().completion(req.model, req.prompt, **params)
    usage = _usage_payload(await get_manager().usage(req.model))
    return _completion_response(req.model, content, usage)


async def _completion_stream_generator(model_id: str, prompt: str, params: dict) -> AsyncIterator[str]:
    """Completion akışını OpenAI SSE biçiminde parça parça üretir."""
    stream_id = f"cmpl-{uuid4().hex[:12]}"
    async for chunk in get_manager().completion_stream(model_id, prompt, **params):
        yield _sse_completion_chunk(stream_id, model_id, chunk, None)
    yield _sse_completion_chunk(stream_id, model_id, "", "stop")
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------
# Responses (OpenAI Responses API) — Codex CLI uyumu
# ------------------------------------------------------------------
# Kod:  - POST /v1/responses, Codex CLI'nın konuştuğu şema ile çalışır.
#       - Giriş: input (metin veya kalem dizisi) + instructions + tools.
#       - Çıkış: mesaj (output_text) ve function_call kalemleri; stream=1 ise
#         response.* SSE olayları yayınlanır.
#       - Altta aynı chat çekirdeğini kullanır; yalnızca protokol dönüşümü yapar.

def _content_to_text(content):
    """Responses içerik bloklarını (input_text/output_text) düz metne çevirir."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                body = block.get("text")
                if body:
                    texts.append(body)
        return "\n".join(t for t in texts if t) or None
    return None


def _responses_input_to_messages(input_val, instructions: str | None) -> list:
    """Responses isteğinin input/instructions alanlarını chat mesajlarına çevirir."""
    messages: list = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if input_val is None:
        return messages
    if isinstance(input_val, str):
        if input_val:
            messages.append({"role": "user", "content": input_val})
        return messages
    if isinstance(input_val, dict):
        input_val = [input_val]
    for item in input_val:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            role = item.get("role") or "user"
            text = _content_to_text(item.get("content"))
            messages.append({"role": role, "content": text if text is not None else ""})
        elif itype == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": str(item.get("output") if item.get("output") is not None else ""),
                }
            )
        elif itype == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        else:
            role = item.get("role") or "user"
            content = item.get("content")
            if content is not None:
                text = content if isinstance(content, str) else _content_to_text(content)
                messages.append({"role": role, "content": text or ""})
    return messages


def _responses_tools_to_chat(tools) -> list:
    """Responses tool şemasını (type/name/desc/param) chat.completions şemasına çevirir."""
    out: list = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function":
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name") or "function",
                        "description": t.get("description"),
                        "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        # type="custom" ve yerleşik araçlar (web_search vb.) yerel motorla
        # çalışmaz; bu sürümde sessizce yoksayılır.
    return out


def _responses_tool_choice_to_chat(tool_choice):
    """Responses tool_choice (str veya {type, name}) chat biçimine çevirir."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice  # auto | none | required
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return {"type": "function", "function": {"name": tool_choice.get("name")}}
        if tool_choice.get("type") in ("auto", "none", "required"):
            return tool_choice["type"]
    return None


def _responses_params(req: ResponsesRequest) -> dict:
    """Responses istek alanlarını motor parametrelerine ayıklar."""
    params: dict = {}
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.top_p is not None:
        params["top_p"] = req.top_p
    if req.max_output_tokens is not None:
        params["max_tokens"] = req.max_output_tokens
    tools = _responses_tools_to_chat(req.tools)
    if tools:
        params["tools"] = tools
    tc = _responses_tool_choice_to_chat(req.tool_choice)
    if tc:
        params["tool_choice"] = tc
    # parallel_tool_calls motor tarafından desteklenmez; yanıtta req'ten yansıtılır.
    return params


def _responses_usage(usage: dict) -> dict:
    """Motor kullanımını Responses usage şemasına çevirir."""
    if not usage:
        return {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.get("completion_tokens", 0),
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.get("total_tokens", 0),
    }


def _msg_item(item_id: str, status: str, text: str) -> dict:
    """Responses mesaj kalemini (message item) üretir."""
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "annotations": [],
    }


def _responses_output_items(content: str, tool_calls) -> list:
    """Akışsız yanıtın output kalemlerini (mesaj + fonksiyon çağrıları) üretir."""
    items: list = []
    if content:
        items.append(
            {
                "id": f"msg_{uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
                "annotations": [],
            }
        )
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        items.append(
            {
                "id": f"fc_{uuid4().hex[:12]}",
                "type": "function_call",
                "status": "completed",
                "call_id": tc.get("id") or f"call_{uuid4().hex[:12]}",
                "name": fn.get("name"),
                "arguments": fn.get("arguments") or "",
                "parallel_tool_calls": True,
                "annotations": [],
            }
        )
    return items


def _resp_obj(
    model_id: str,
    req: ResponsesRequest,
    output: list,
    usage: dict,
    status: str = "completed",
    created_at: float | None = None,
    resp_id: str | None = None,
) -> dict:
    """Responses yanıt nesnesini OpenAI şemasında kurar."""
    now = created_at or time.time()
    return {
        "id": resp_id or f"resp_{uuid4().hex[:12]}",
        "object": "response",
        "created_at": now,
        "status": status,
        "model": model_id,
        "instructions": req.instructions,
        "output": output,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_output_tokens": req.max_output_tokens,
        "parallel_tool_calls": req.parallel_tool_calls if req.parallel_tool_calls is not None else True,
        "reasoning": req.reasoning or {"effort": None, "summary": None},
        "tools": req.tools or [],
        "tool_choice": req.tool_choice or "auto",
        "usage": _responses_usage(usage),
        "error": None,
        "incomplete_details": None,
        "metadata": req.metadata or {},
        "service_tier": req.service_tier or "default",
    }


def _sse_resp_event(payload: dict) -> str:
    """Responses SSE olayını (response.*) tek satır olarak üretir."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _collect_calls(call_state: dict) -> list:
    """Akışta biriken araç çağrılarını normalleştirilmiş listeye çevirir."""
    calls = []
    for idx, st in sorted(call_state.items()):
        calls.append(
            {
                "id": st.get("call_id"),
                "type": "function",
                "function": {"name": st.get("name"), "arguments": st.get("args", "")},
            }
        )
    return calls


@router.post("/responses")
async def responses(req: ResponsesRequest):
    """OpenAI Responses API uyumlu yanıt üretir; stream=True ise SSE akışı.

    Codex CLI gibi araçların konuştuğu şema: input + instructions ile sor,
    output içinde message/function_call kalemleriyle cevap al.
    """
    _require_model(req.model)
    messages = _responses_input_to_messages(req.input, req.instructions)
    params = _responses_params(req)

    if req.stream:
        return StreamingResponse(
            _responses_stream_generator(req.model, messages, params, req),
            media_type="text/event-stream",
            headers=STREAM_HEADERS,
        )

    content = await get_manager().chat(req.model, messages, **params)
    usage = await get_manager().usage(req.model)
    tool_calls, _finish = await get_manager().tool_call_info(req.model)
    output = _responses_output_items(content, tool_calls)
    return _resp_obj(req.model, req, output, usage, status="completed")


async def _responses_stream_generator(
    model_id: str, messages: list, params: dict, req: ResponsesRequest
) -> AsyncIterator[str]:
    """Responses akışını OpenAI SSE olaylarıyla üretir.

    Kod, içerik (output_text) ve araç çağrısı (function_call) kalemlerini
    response.output_item.added -> response.output_text.delta /
    function_call_arguments.delta -> response.output_item.done olaylarıyla
    yayınlar; tamamlanınca response.completed + response.done ve [DONE] gönderir.
    """
    resp_id = f"resp_{uuid4().hex[:12]}"
    created_at = time.time()
    pending = _resp_obj(
        model_id, req, [], _responses_usage({}), status="in_progress",
        created_at=created_at, resp_id=resp_id,
    )
    yield _sse_resp_event({"type": "response.created", "response": pending})
    yield _sse_resp_event({"type": "response.in_progress", "response": pending})

    msg_item_id = f"msg_{uuid4().hex[:12]}"
    content_buf = ""
    call_state: dict = {}
    usage: dict = {}

    async for ev in get_manager().chat_stream_events(model_id, messages, **params):
        if isinstance(ev, str):
            # Geriye dönük uyum: eski/sahte motorlar doğrudan metin üretir.
            ev = {"type": "content", "text": ev}
        kind = ev.get("type")
        if kind == "content":
            text = ev.get("text", "")
            if not content_buf:
                yield _sse_resp_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": _msg_item(msg_item_id, status="in_progress", text=""),
                    }
                )
                yield _sse_resp_event(
                    {
                        "type": "response.content_part.added",
                        "item_id": msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
            content_buf += text
            yield _sse_resp_event(
                {
                    "type": "response.output_text.delta",
                    "item_id": msg_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                    "logprobs": [],
                }
            )
        elif kind == "tool_args":
            idx = ev["index"]
            st = call_state.setdefault(
                idx,
                {
                    "call_id": ev.get("id") or f"call_{uuid4().hex[:12]}",
                    "name": ev.get("name") or "function",
                    "args": "",
                    "item_id": f"fc_{uuid4().hex[:12]}",
                    "opened": False,
                },
            )
            if not st["opened"]:
                st["opened"] = True
                yield _sse_resp_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": idx + 1,
                        "item": {
                            "type": "function_call",
                            "id": st["item_id"],
                            "call_id": st["call_id"],
                            "name": st["name"],
                            "arguments": "",
                            "status": "in_progress",
                            "parallel_tool_calls": True,
                        },
                    }
                )
            delta = ev.get("delta", "") or ""
            st["args"] += delta
            yield _sse_resp_event(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": st["item_id"],
                    "output_index": idx + 1,
                    "delta": delta,
                }
            )
        elif kind == "usage":
            usage = ev.get("usage", {})
        elif kind == "finish":
            pass  # tamamlama olayları akışın sonunda üretilir

    # İçerik kalemini bitir (varsa)
    if content_buf:
        yield _sse_resp_event(
            {
                "type": "response.output_text.done",
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": content_buf,
                "logprobs": [],
            }
        )
        yield _sse_resp_event(
            {
                "type": "response.content_part.done",
                "item_id": msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": content_buf, "annotations": []},
            }
        )
        yield _sse_resp_event(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": _msg_item(msg_item_id, status="completed", text=content_buf),
            }
        )

    # Araç çağrısı kalemlerini bitir (varsa)
    for idx, st in sorted(call_state.items()):
        yield _sse_resp_event(
            {
                "type": "response.function_call_arguments.done",
                "item_id": st["item_id"],
                "output_index": idx + 1,
                "arguments": st.get("args", ""),
            }
        )
        yield _sse_resp_event(
            {
                "type": "response.output_item.done",
                "output_index": idx + 1,
                "item": {
                    "type": "function_call",
                    "id": st["item_id"],
                    "call_id": st.get("call_id"),
                    "name": st.get("name"),
                    "arguments": st.get("args", ""),
                    "status": "completed",
                    "parallel_tool_calls": True,
                },
            }
        )

    # Tam yanıt nesnesini kur ve bitir
    if not usage:
        # Usage akış içinde gelmediyse motor defterinden al (varsa)
        usage = await get_manager().usage(model_id)
    output = _responses_output_items(content_buf, _collect_calls(call_state))
    final = _resp_obj(
        model_id, req, output, usage, status="completed",
        created_at=created_at, resp_id=resp_id,
    )
    yield _sse_resp_event({"type": "response.completed", "response": final})
    yield _sse_resp_event({"type": "response.done", "response": final})
    yield "data: [DONE]\n\n"