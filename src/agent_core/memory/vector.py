"""Vector memory — planned for v2. Kept as an explicit stub so the seam is visible."""

from __future__ import annotations

from agent_core.state import AgentState
from agent_core.types import Message


class VectorMemory:
    def __init__(self) -> None:
        raise NotImplementedError("VectorMemory is planned for v2")

    async def load(self, state: AgentState) -> list[Message]:  # pragma: no cover
        raise NotImplementedError

    async def commit(  # pragma: no cover
        self, state: AgentState, new_messages: list[Message]
    ) -> None:
        raise NotImplementedError
