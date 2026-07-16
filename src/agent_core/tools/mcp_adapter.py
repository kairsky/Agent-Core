"""MCP integration: tools from MCP servers look like regular tools to the loop.

Naming convention: registered tool name is "{server}__{original_name}".
Requires the `mcp` extra for real connections; result/error mapping is SDK-agnostic
and unit-testable without it.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from agent_core.config import McpServerConfig
from agent_core.tools.base import ToolContext, ToolResult, ToolSpec
from agent_core.tools.registry import ToolRegistry

MCP_NAME_SEPARATOR = "__"


def map_call_result(result: Any) -> ToolResult:
    """Map an MCP CallToolResult to a ToolResult (text parts are joined)."""
    texts = [part.text for part in result.content if getattr(part, "text", None)]
    content = "\n".join(texts)
    if getattr(result, "isError", False):
        return ToolResult.failure("execution_error", content or "MCP tool reported an error.")
    return ToolResult(ok=True, content=content)


def map_exception(server: str, tool: str, exc: Exception) -> ToolResult:
    if isinstance(exc, TimeoutError):
        return ToolResult.failure("timeout", f"MCP tool '{tool}' timed out.")
    message = str(exc).lower()
    if "not found" in message or "unknown tool" in message:
        return ToolResult.failure("not_found", f"MCP tool '{tool}' not found on '{server}'.")
    if "invalid" in message and ("argument" in message or "param" in message):
        return ToolResult.failure("validation_error", f"Invalid arguments for '{tool}': {exc}")
    if isinstance(exc, (ConnectionError, BrokenPipeError, EOFError)):
        return ToolResult.failure("unavailable", f"MCP server '{server}' is unavailable.")
    return ToolResult.failure("execution_error", f"MCP tool '{tool}' failed: {type(exc).__name__}")


class McpToolAdapter:
    """Presents one remote MCP tool as a local ToolHandler."""

    def __init__(self, session: Any, server_name: str, spec: ToolSpec, original_name: str) -> None:
        self.spec = spec
        self._session = session
        self._server_name = server_name
        self._original_name = original_name

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await self._session.call_tool(self._original_name, arguments)
        except Exception as exc:  # noqa: BLE001 - normalized for the model
            return map_exception(self._server_name, self._original_name, exc)
        return map_call_result(result)


class McpConnection:
    """Owns one MCP server session: connect, discover tools, close."""

    def __init__(self, config: McpServerConfig) -> None:
        if config.transport != "stdio":
            raise NotImplementedError("v1 supports only stdio MCP transport")
        if not config.command:
            raise ValueError(f"MCP server '{config.name}' needs a command for stdio transport")
        self._config = config
        self._stack = AsyncExitStack()
        self._session: Any = None

    @property
    def name(self) -> str:
        return self._config.name

    async def connect(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MCP support requires the mcp SDK: pip install 'agent-core[mcp]'"
            ) from exc

        params = StdioServerParameters(
            command=self._config.command[0],
            args=self._config.command[1:],
            env=self._config.env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def register_tools(
        self, registry: ToolRegistry, dangerous_names: frozenset[str] = frozenset()
    ) -> int:
        listing = await self._session.list_tools()
        for tool in listing.tools:
            spec = ToolSpec(
                name=f"{self.name}{MCP_NAME_SEPARATOR}{tool.name}",
                description=tool.description or "",
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
                source="mcp",
                mcp_server=self.name,
                dangerous=tool.name in dangerous_names,
            )
            registry.register(McpToolAdapter(self._session, self.name, spec, tool.name))
        return len(listing.tools)

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None
