from __future__ import annotations

from agent_core.tracing import EventType, TraceEvent
from agent_core.tracing.redaction import REDACTED, RedactingTracer, redact_text, redact_value


class CapturingTracer:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None


def test_masks_openai_style_keys():
    text = redact_text("using key sk-proj-abcdefghijklmnop1234 for the call")
    assert "sk-proj" not in text
    assert REDACTED in text


def test_masks_bearer_tokens_and_kv_pairs():
    assert REDACTED in redact_text("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6")
    assert REDACTED in redact_text("api_key=super-secret-value-123")
    assert REDACTED in redact_text("password: hunter2hunter2")


def test_plain_text_is_untouched():
    text = "calculate 17*19 and write to out.txt"
    assert redact_text(text) == text


def test_redacts_nested_payload_structures():
    payload = {
        "arguments": {"headers": ["Bearer abcdefghijklmnopqrstuv"], "url": "https://x.dev"},
    }
    cleaned = redact_value(payload)
    assert cleaned["arguments"]["headers"][0] == REDACTED
    assert cleaned["arguments"]["url"] == "https://x.dev"


async def test_tracer_wrapper_scrubs_events():
    inner = CapturingTracer()
    tracer = RedactingTracer(inner)
    await tracer.emit(
        TraceEvent(
            run_id="r",
            type=EventType.TOOL_REQUESTED,
            payload={"arguments": {"key": "sk-proj-abcdefghijklmnop1234"}},
        )
    )
    assert inner.events[0].payload["arguments"]["key"] == REDACTED
