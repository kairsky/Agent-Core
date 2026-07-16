"""Shared loop machinery: LLM turns with retries, tool-call handling, budget guards.

Concrete loops (ReAct, Plan-Execute) differ only in how they drive steps.
"""

from __future__ import annotations

import asyncio

from agent_core.config import AgentConfig
from agent_core.control.budgets import Budgets
from agent_core.control.policies import ToolPolicy
from agent_core.control.stop import StopReason
from agent_core.llm.base import LLMError, LLMProvider, LLMResult
from agent_core.memory.base import MemoryStore
from agent_core.state import AgentState
from agent_core.tools.base import ConfirmCallback, ToolContext, ToolResult
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.runtime import ToolRuntime
from agent_core.tracing import EventType, TraceEvent, Tracer, preview
from agent_core.types import Message, ToolCall, tool_response


class LoopBase:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMProvider,
        registry: ToolRegistry,
        tool_runtime: ToolRuntime,
        memory: MemoryStore,
        policy: ToolPolicy,
        tracer: Tracer,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._registry = registry
        self._tool_runtime = tool_runtime
        self._memory = memory
        self._policy = policy
        self._budgets = Budgets(config)
        self._tracer = tracer
        self._confirm = confirm

    def _guard(self, state: AgentState) -> StopReason | None:
        """Check step/budget limits; mark the run failed if any is exceeded."""
        if state.step >= self._config.max_steps:
            state.status = "failed"
            return StopReason.MAX_STEPS
        violation = self._budgets.exceeded(state)
        if violation is not None:
            state.status = "failed"
            return violation
        return None

    async def _emit_run_started(self, state: AgentState) -> None:
        await self._emit(
            state,
            EventType.RUN_STARTED,
            goal=state.goal,
            config_digest=self._config.digest(),
            model=self._config.model,
            loop=self._config.loop,
        )

    async def _emit_run_finished(self, state: AgentState, stop_reason: StopReason | None) -> None:
        if state.status == "failed" and state.last_error is None and stop_reason is not None:
            state.last_error = str(stop_reason)
        await self._emit(
            state,
            EventType.RUN_FINISHED,
            status=state.status,
            stop_reason=str(stop_reason) if stop_reason else None,
            final_answer=state.final_answer,
            usage_total={
                "tokens_in": state.tokens_in,
                "tokens_out": state.tokens_out,
                "cost_usd": round(state.cost_usd, 6),
            },
            steps=state.step,
        )

    async def _llm_turn(
        self, state: AgentState, messages: list[Message], tools: list
    ) -> Message | None:
        """One LLM completion: applies usage, appends and returns the assistant message.

        Returns None (with state marked failed) on an unrecoverable LLM error.
        """
        try:
            llm_result = await self._complete_with_retries(messages, tools, state)
        except LLMError as exc:
            state.status = "failed"
            state.last_error = str(exc)
            await self._emit(state, EventType.ERROR, where="llm", message=str(exc))
            return None

        state.apply_usage(llm_result.usage)
        assistant = llm_result.message
        state.messages.append(assistant)
        await self._emit(
            state,
            EventType.LLM_COMPLETED,
            model=self._config.model,
            finish_reason=llm_result.finish_reason,
            usage={"in": llm_result.usage.input_tokens, "out": llm_result.usage.output_tokens},
            has_tool_calls=bool(assistant.tool_calls),
            content_preview=preview(assistant.content),
        )
        return assistant

    async def _handle_tool_calls(
        self, state: AgentState, calls: list[ToolCall], new_messages: list
    ) -> StopReason | None:
        calls = calls[: self._config.max_tool_calls_per_step]
        ctx = ToolContext(
            run_id=state.run_id,
            step=state.step,
            cwd=self._config.workspace,
            confirm=self._confirm,
        )

        # Policy and HITL decisions are sequential; approved calls may run in parallel.
        executable: list[ToolCall] = []
        results: dict[str, ToolResult] = {}
        for call in calls:
            if not self._policy.allow(call.name):
                await self._emit(
                    state, EventType.POLICY_DENIED, name=call.name, reason="denied by policy"
                )
                results[call.id] = ToolResult.failure(
                    "policy_denied", f"Tool '{call.name}' is not allowed by policy."
                )
                continue
            if self._policy.needs_confirmation(call.name, call.arguments):
                if self._confirm is None:
                    state.status = "needs_input"
                    state.scratchpad["pending_confirmation"] = {
                        "tool_call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    return StopReason.NEEDS_INPUT
                approved = await self._confirm(call.name, call.arguments)
                await self._emit(
                    state, EventType.POLICY_CONFIRMATION, name=call.name, approved=approved
                )
                if not approved:
                    results[call.id] = ToolResult.failure(
                        "policy_denied", f"User declined to run '{call.name}'."
                    )
                    continue
            executable.append(call)

        if self._config.parallel_tools and len(executable) > 1:
            executed = await asyncio.gather(
                *(self._tool_runtime.execute(call, ctx) for call in executable)
            )
            for call, result in zip(executable, executed, strict=True):
                results[call.id] = result
                state.tool_calls_total += 1
        else:
            for call in executable:
                results[call.id] = await self._tool_runtime.execute(call, ctx)
                state.tool_calls_total += 1

        for call in calls:
            result = results.get(call.id)
            if result is None:
                continue
            message = tool_response(call.id, call.name, result.content)
            state.messages.append(message)
            new_messages.append(message)

        violation = self._budgets.exceeded(state)
        if violation is not None:
            state.status = "failed"
            return violation
        return None

    async def _complete_with_retries(
        self, messages: list[Message], tools: list, state: AgentState
    ) -> LLMResult:
        last_error: LLMError | None = None
        for attempt in range(self._config.llm_max_retries + 1):
            try:
                return await self._llm.complete(
                    messages,
                    tools,
                    model=self._config.model,
                    temperature=self._config.temperature,
                )
            except LLMError as exc:
                last_error = exc
                if attempt < self._config.llm_max_retries:
                    await self._emit(
                        state,
                        EventType.ERROR,
                        where="llm",
                        message=f"attempt {attempt + 1} failed, retrying: {exc}",
                    )
        raise last_error  # type: ignore[misc]

    async def _emit(self, state: AgentState, event_type: EventType, **payload) -> None:
        await self._tracer.emit(
            TraceEvent(run_id=state.run_id, type=event_type, step=state.step, payload=payload)
        )
