from __future__ import annotations

import asyncio
from typing import Any

from agent_core.tools.base import FunctionTool, ToolContext, ToolResult, ToolSpec
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.runtime import ToolRuntime
from agent_core.tracing import NullTracer
from agent_core.types import ToolCall

ECHO_SPEC = ToolSpec(
    name="echo",
    description="echo text",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


async def echo(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(ok=True, content=arguments["text"])


async def sleepy(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    await asyncio.sleep(5)
    return ToolResult(ok=True, content="done")


async def crashy(arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raise RuntimeError("secret traceback detail")


def make_runtime(config, handlers) -> ToolRuntime:
    return ToolRuntime(ToolRegistry(handlers), config, NullTracer())


def make_ctx(config) -> ToolContext:
    return ToolContext(run_id="r", step=1, cwd=config.workspace)


async def test_validates_arguments_against_schema(config):
    runtime = make_runtime(config, [FunctionTool(ECHO_SPEC, echo)])
    result = await runtime.execute(ToolCall(id="1", name="echo", arguments={}), make_ctx(config))
    assert not result.ok
    assert result.error_type == "validation_error"
    assert "required" in result.content  # schema hint returned to the model


async def test_unknown_tool_is_not_found(config):
    runtime = make_runtime(config, [])
    result = await runtime.execute(ToolCall(id="1", name="nope", arguments={}), make_ctx(config))
    assert result.error_type == "not_found"


async def test_timeout_is_normalized(config):
    spec = ToolSpec(name="sleepy", description="", parameters={"type": "object"})
    runtime = make_runtime(config, [FunctionTool(spec, sleepy)])
    result = await runtime.execute(ToolCall(id="1", name="sleepy", arguments={}), make_ctx(config))
    assert not result.ok
    assert result.error_type == "timeout"


async def test_exception_hides_traceback_from_model(config):
    spec = ToolSpec(name="crashy", description="", parameters={"type": "object"})
    runtime = make_runtime(config, [FunctionTool(spec, crashy)])
    result = await runtime.execute(ToolCall(id="1", name="crashy", arguments={}), make_ctx(config))
    assert not result.ok
    assert result.error_type == "execution_error"
    assert "secret traceback detail" not in result.content
    assert "RuntimeError" in result.content
