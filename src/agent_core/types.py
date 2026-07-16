"""Shared core types: messages, tool calls, usage accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model (arguments already parsed)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None  # tool name when role == "tool"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    """Token/cost accounting for a single LLM completion."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def tool_response(tool_call_id: str, name: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id, name=name)
