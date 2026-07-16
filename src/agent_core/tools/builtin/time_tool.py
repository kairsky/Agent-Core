"""Deterministic-shape utility tool: current UTC time."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_core.tools.base import ToolContext, ToolResult, ToolSpec


class GetCurrentTimeTool:
    spec = ToolSpec(
        name="get_current_time",
        description="Get the current date and time in UTC (ISO 8601).",
        parameters={"type": "object", "properties": {}},
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return ToolResult(ok=True, content=now, data={"iso": now})
