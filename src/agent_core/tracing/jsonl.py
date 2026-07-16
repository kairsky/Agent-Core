"""JSONL tracer: appends one JSON line per event to traces/{run_id}.jsonl."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TextIO

from agent_core.tracing.schema import TraceEvent


class JsonlTracer:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: TextIO | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def emit(self, event: TraceEvent) -> None:
        async with self._lock:
            if self._file is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._path.open("a", encoding="utf-8")
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()

    async def close(self) -> None:
        async with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
