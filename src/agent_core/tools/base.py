"""Tool contracts: spec, handler protocol, execution context and result."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

ToolSource = Literal["builtin", "mcp", "http"]

# Async callback asked to approve a dangerous tool call: (tool_name, arguments) -> approved.
ConfirmCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    source: ToolSource = "builtin"
    mcp_server: str | None = None
    dangerous: bool = False


@dataclass
class ToolContext:
    run_id: str
    step: int
    cwd: Path
    confirm: ConfirmCallback | None = None


@dataclass
class ToolResult:
    ok: bool
    content: str  # what the model sees
    data: Any | None = None  # structured payload for traces
    error_type: str | None = None

    @classmethod
    def failure(cls, error_type: str, content: str) -> ToolResult:
        return cls(ok=False, content=content, error_type=error_type)


@runtime_checkable
class ToolHandler(Protocol):
    spec: ToolSpec

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class FunctionTool:
    """Wraps an async function into a ToolHandler."""

    def __init__(
        self,
        spec: ToolSpec,
        fn: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]],
    ) -> None:
        self.spec = spec
        self._fn = fn

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._fn(arguments, ctx)
