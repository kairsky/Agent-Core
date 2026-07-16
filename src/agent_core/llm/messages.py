"""Conversion between core Message objects and the OpenAI chat wire format."""

from __future__ import annotations

import json
from typing import Any

from agent_core.tools.base import ToolSpec
from agent_core.types import Message, ToolCall


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role}
        if message.role == "tool":
            item["tool_call_id"] = message.tool_call_id
            item["content"] = message.content or ""
        else:
            item["content"] = message.content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in message.tool_calls
            ]
        result.append(item)
    return result


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
    """Parse provider tool calls; malformed JSON arguments raise ValueError."""
    if not raw_tool_calls:
        return None
    calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        try:
            arguments = json.loads(raw.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool call '{raw.function.name}' has invalid JSON arguments") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"tool call '{raw.function.name}' arguments must be an object")
        calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=arguments))
    return calls
