# ─────────────────────────────────────────────────────────────
#  Bölüm:    Motor Araç (Tool) Davranışı Testleri
#  Dosya:    tests/test_llama_backend_tools.py
#  Amaç:     LlamaCppEngine'in araç çağrısı mantığını (tool_call_info,
#            event'li akış) gerçek model gerektirmeden stub ile doğrular.
#  Mekanik:  Gerçek llama.cpp nesnesi yerine create_chat_completion'i taklit
#            eden bir stub (_llama) kullanılır; parse/state mantığı CPU
#            modeli olmadan test edilir.
# ─────────────────────────────────────────────────────────────

from app.llama_backend import LlamaCppEngine


class _StubLlama:
    """LlamaCppEngine._llama yerine geçen sahte model arayüzü."""

    def __init__(self, stream_events=None, nonstream=None):
        self.stream_events = stream_events or []
        self.nonstream_result = nonstream

    def create_chat_completion(self, messages=None, stream=False, **kwargs):
        if stream:
            return iter(self.stream_events)
        return self.nonstream_result


def _motor(stream_events=None, nonstream=None) -> LlamaCppEngine:
    """İçinde stub model olan bir motor kurar (gerçek model gerekmez)."""
    eng = LlamaCppEngine("yok.gguf", context_size=512, gpu_layers=0, threads=2)
    eng._llama = _StubLlama(stream_events=stream_events, nonstream=nonstream)
    return eng


TOOL_CALLS_RESULT = {
    "choices": [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "ls"}},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
}


def test_run_chat_surfaces_tool_calls():
    """Akışsız çalıştırma araç çağrısını ve bitiş nedenini tool_call_info'ya yazar."""
    eng = _motor(nonstream=TOOL_CALLS_RESULT)
    assert eng.is_loaded is True
    cevap = eng.run_chat([{"role": "user", "content": "ls çalıştır"}])
    assert cevap == ""  # araç çağrısında içerik tipik olarak boştur
    calls, finish = eng.tool_call_info()
    assert finish == "tool_calls"
    assert calls[0]["function"]["name"] == "bash"
    # arguments sözlük ise JSON dizgesine çevrilmeli
    assert calls[0]["function"]["arguments"] == '{"command": "ls"}'


def test_run_chat_no_tools_gives_none():
    """Araç çağrısı yoksa tool_call_info boş döner."""
    eng = _motor(
        nonstream={
            "choices": [{"message": {"content": "Selam"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 3},
        }
    )
    eng.run_chat([{"role": "user", "content": "x"}])
    calls, finish = eng.tool_call_info()
    assert calls is None
    assert finish == "stop"


def test_stream_chat_events_tool_args():
    """Event'li akış araç çağrısı delta'larını yayınlar ve biriktirir."""
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"c'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'ommand":"ls"}'}}
                        ]
                    }
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
    ]
    eng = _motor(stream_events=events)
    got = list(eng.stream_chat([{"role": "user", "content": "ls"}], _yield_events=True))
    types = [g["type"] for g in got]
    assert types == ["tool_args", "tool_args", "finish", "usage"]
    args_delta = "".join(g.get("delta", "") for g in got if g["type"] == "tool_args")
    assert args_delta == '{"command":"ls"}'
    calls, finish = eng.tool_call_info()
    assert finish == "tool_calls"
    assert calls[0]["function"]["name"] == "bash"
    assert calls[0]["function"]["arguments"] == '{"command":"ls"}'


def test_stream_chat_events_content():
    """Event'li akış içerik delta'larını ve eski modda metni doğru üretir."""
    events = [
        {"choices": [{"delta": {"content": "Mer"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "haba"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 4}},
    ]
    eng = _motor(stream_events=events)

    g1 = list(eng.stream_chat([{"role": "user", "content": "x"}], _yield_events=True))
    assert [g["type"] for g in g1] == ["content", "content", "finish", "usage"]

    g2 = list(eng.stream_chat([{"role": "user", "content": "x"}], _yield_events=False))
    assert g2 == ["Mer", "haba"]