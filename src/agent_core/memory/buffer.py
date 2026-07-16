"""Buffer memory: full history, trimmed to a token budget for the LLM call."""

from __future__ import annotations

from agent_core.memory.base import drop_oldest_turn, split_protected_head, total_tokens
from agent_core.state import AgentState
from agent_core.types import Message


class BufferMemory:
    def __init__(self, max_context_tokens: int | None = None) -> None:
        self._max_tokens = max_context_tokens

    async def load(self, state: AgentState) -> list[Message]:
        if self._max_tokens is None:
            return list(state.messages)
        head, tail = split_protected_head(state.messages)
        while tail and total_tokens(head + tail) > self._max_tokens:
            tail = drop_oldest_turn(tail)
        return head + tail

    async def commit(self, state: AgentState, new_messages: list[Message]) -> None:
        return None  # state.messages already holds the full history
