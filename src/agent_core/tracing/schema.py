"""Trace event models. One run = one JSONL file, one line = one TraceEvent."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

# How many characters of tool/LLM output land in `content_preview` fields.
CONTENT_PREVIEW_CHARS = 200


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    LLM_COMPLETED = "llm.completed"
    TOOL_REQUESTED = "tool.requested"
    TOOL_EXECUTED = "tool.executed"
    POLICY_DENIED = "policy.denied"
    POLICY_CONFIRMATION = "policy.confirmation"
    MEMORY_COMPACTED = "memory.compacted"
    RUN_FINISHED = "run.finished"
    ERROR = "error"


class TraceEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: EventType
    step: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


def preview(text: str | None, limit: int = CONTENT_PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "..."
