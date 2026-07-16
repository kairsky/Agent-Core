"""Strict tool execution: schema validation, timeouts, error normalization, tracing.

The model never sees raw tracebacks — only short, typed error messages.
Full details go to the trace.
"""

from __future__ import annotations

import asyncio
import time

import jsonschema

from agent_core.config import AgentConfig
from agent_core.tools.base import ToolContext, ToolResult
from agent_core.tools.registry import ToolRegistry
from agent_core.tracing import EventType, TraceEvent, Tracer, preview
from agent_core.types import ToolCall


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, config: AgentConfig, tracer: Tracer) -> None:
        self._registry = registry
        self._config = config
        self._tracer = tracer

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        await self._tracer.emit(
            TraceEvent(
                run_id=ctx.run_id,
                type=EventType.TOOL_REQUESTED,
                step=ctx.step,
                payload={"tool_call_id": call.id, "name": call.name, "arguments": call.arguments},
            )
        )
        started = time.monotonic()
        result = await self._execute_inner(call, ctx)
        latency_ms = int((time.monotonic() - started) * 1000)
        await self._tracer.emit(
            TraceEvent(
                run_id=ctx.run_id,
                type=EventType.TOOL_EXECUTED,
                step=ctx.step,
                payload={
                    "tool_call_id": call.id,
                    "name": call.name,
                    "ok": result.ok,
                    "error_type": result.error_type,
                    "latency_ms": latency_ms,
                    "content_preview": preview(result.content),
                },
            )
        )
        return result

    async def _execute_inner(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        handler = self._registry.get(call.name)
        if handler is None:
            return ToolResult.failure("not_found", f"Tool '{call.name}' does not exist.")

        validation_error = self._validate_arguments(call, handler.spec.parameters)
        if validation_error is not None:
            return validation_error

        try:
            return await asyncio.wait_for(
                handler.run(call.arguments, ctx), timeout=self._config.tool_timeout_s
            )
        except TimeoutError:
            return ToolResult.failure(
                "timeout", f"Tool '{call.name}' timed out after {self._config.tool_timeout_s}s."
            )
        except Exception as exc:  # noqa: BLE001 - normalize any handler failure
            await self._tracer.emit(
                TraceEvent(
                    run_id=ctx.run_id,
                    type=EventType.ERROR,
                    step=ctx.step,
                    payload={"where": f"tool:{call.name}", "message": repr(exc)},
                )
            )
            return ToolResult.failure(
                "execution_error", f"Tool '{call.name}' failed: {type(exc).__name__}."
            )

    @staticmethod
    def _validate_arguments(call: ToolCall, schema: dict) -> ToolResult | None:
        try:
            jsonschema.validate(call.arguments, schema)
        except jsonschema.ValidationError as exc:
            # Return the schema hint so the model can self-correct on the next attempt.
            return ToolResult.failure(
                "validation_error",
                f"Invalid arguments for '{call.name}': {exc.message}. Expected schema: {schema}",
            )
        except jsonschema.SchemaError:
            return ToolResult.failure(
                "validation_error", f"Tool '{call.name}' has a broken parameter schema."
            )
        return None
