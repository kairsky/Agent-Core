"""Plan-Execute loop — scheduled for v1.1. Explicit stub to keep the seam visible."""

from __future__ import annotations

from agent_core.state import AgentState


class PlanExecuteLoop:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("PlanExecuteLoop is planned for v1.1")

    async def run(self, state: AgentState) -> AgentState:  # pragma: no cover
        raise NotImplementedError
