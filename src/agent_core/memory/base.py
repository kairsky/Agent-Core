"""Memory contract and shared helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_core.state import AgentState
from agent_core.types import Message


@runtime_checkable
class MemoryStore(Protocol):
    async def load(self, state: AgentState) -> list[Message]:
        """Messages to send to the LLM this step."""
        ...

    async def commit(self, state: AgentState, new_messages: list[Message]) -> None:
        """Called at the end of each step, after new messages were appended to state."""
        ...


def approx_tokens(message: Message) -> int:
    """Cheap token estimate (~4 chars/token) — good enough for trim thresholds."""
    total = len(message.content or "")
    for call in message.tool_calls or []:
        total += len(call.name) + len(str(call.arguments))
    return max(1, total // 4)


def total_tokens(messages: list[Message]) -> int:
    return sum(approx_tokens(m) for m in messages)


def split_protected_head(messages: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split into (system prefix + initial user goal, the rest). The head is never dropped."""
    head: list[Message] = []
    index = 0
    while index < len(messages) and messages[index].role == "system":
        head.append(messages[index])
        index += 1
    if index < len(messages) and messages[index].role == "user":
        head.append(messages[index])
        index += 1
    return head, messages[index:]


def drop_oldest_turn(tail: list[Message]) -> list[Message]:
    """Drop the oldest message; if it carried tool_calls, drop its tool results too."""
    if not tail:
        return tail
    dropped = tail[0]
    rest = tail[1:]
    if dropped.tool_calls:
        call_ids = {call.id for call in dropped.tool_calls}
        rest = [m for m in rest if not (m.role == "tool" and m.tool_call_id in call_ids)]
    # A tool message must never lead the window without its assistant parent.
    while rest and rest[0].role == "tool":
        rest = rest[1:]
    return rest
