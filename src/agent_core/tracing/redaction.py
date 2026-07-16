"""Redaction middleware: masks secrets in trace payloads before they hit disk."""

from __future__ import annotations

import re
from typing import Any

from agent_core.tracing.base import Tracer
from agent_core.tracing.schema import TraceEvent

REDACTED = "[REDACTED]"

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style API keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),  # Authorization headers
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),  # KEY=value pairs
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
]


def redact_text(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


class RedactingTracer:
    """Wraps another tracer and scrubs secrets from every event payload."""

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner

    async def emit(self, event: TraceEvent) -> None:
        await self._inner.emit(event.model_copy(update={"payload": redact_value(event.payload)}))

    async def close(self) -> None:
        await self._inner.close()
