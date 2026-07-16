"""Tracer protocol and the no-op implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_core.tracing.schema import TraceEvent


@runtime_checkable
class Tracer(Protocol):
    async def emit(self, event: TraceEvent) -> None: ...

    async def close(self) -> None: ...


class NullTracer:
    """Discards all events. Used when tracing is disabled."""

    async def emit(self, event: TraceEvent) -> None:
        return None

    async def close(self) -> None:
        return None
