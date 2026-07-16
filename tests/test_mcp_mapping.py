"""MCP result/error mapping — tested with fakes, no MCP SDK required."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_core.tools.base import ToolContext, ToolSpec
from agent_core.tools.mcp_adapter import McpToolAdapter, map_call_result, map_exception


@dataclass
class FakeTextPart:
    text: str


@dataclass
class FakeCallToolResult:
    content: list[Any] = field(default_factory=list)
    isError: bool = False  # noqa: N815 - mirrors the MCP SDK field name


class FakeSession:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        return self._result


def test_text_parts_are_joined():
    result = map_call_result(
        FakeCallToolResult(content=[FakeTextPart("line1"), FakeTextPart("line2")])
    )
    assert result.ok
    assert result.content == "line1\nline2"


def test_error_result_maps_to_execution_error():
    result = map_call_result(FakeCallToolResult(content=[FakeTextPart("boom")], isError=True))
    assert not result.ok
    assert result.error_type == "execution_error"
    assert result.content == "boom"


def test_timeout_maps_to_timeout():
    assert map_exception("fs", "read", TimeoutError()).error_type == "timeout"


def test_connection_error_maps_to_unavailable():
    assert map_exception("fs", "read", ConnectionError("gone")).error_type == "unavailable"


def test_unknown_tool_maps_to_not_found():
    assert map_exception("fs", "read", RuntimeError("Unknown tool: read")).error_type == "not_found"


async def test_adapter_calls_original_name():
    session = FakeSession(result=FakeCallToolResult(content=[FakeTextPart("ok")]))
    spec = ToolSpec(name="fs__read_file", description="", source="mcp", mcp_server="fs")
    adapter = McpToolAdapter(session, "fs", spec, original_name="read_file")

    result = await adapter.run({"path": "a.txt"}, ToolContext(run_id="r", step=1, cwd=None))
    assert result.ok
    assert session.calls == [("read_file", {"path": "a.txt"})]


async def test_adapter_normalizes_session_errors():
    session = FakeSession(error=ConnectionError("server died"))
    spec = ToolSpec(name="fs__read_file", description="", source="mcp", mcp_server="fs")
    adapter = McpToolAdapter(session, "fs", spec, original_name="read_file")

    result = await adapter.run({}, ToolContext(run_id="r", step=1, cwd=None))
    assert not result.ok
    assert result.error_type == "unavailable"
