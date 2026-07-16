"""Agent loop contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_core.state import AgentState


@runtime_checkable
class AgentLoop(Protocol):
    async def run(self, state: AgentState) -> AgentState: ...
