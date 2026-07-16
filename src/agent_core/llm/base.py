"""LLM provider contract. Providers know nothing about the agent loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_core.tools.base import ToolSpec
from agent_core.types import Message, Usage


@dataclass
class LLMResult:
    message: Message  # assistant message, possibly with tool_calls
    usage: Usage
    finish_reason: str | None = None
    raw: Any | None = None  # provider response for debugging


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float,
    ) -> LLMResult: ...


class LLMError(Exception):
    """Raised by providers on unrecoverable completion failures (after their own retries)."""
