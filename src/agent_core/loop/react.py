"""ReAct loop — the reference implementation of the runtime contract.

Per step: load memory -> LLM completion -> execute requested tools -> commit memory.
The run ends on a final answer (no tool calls), max_steps, a budget violation,
an unrecoverable LLM error, or a dangerous tool awaiting human input.
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
from agent_core.types import ToolCall, tool_response


class ReActLoop:
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

    async def run(self, state: AgentState) -> AgentState:
        state.status = "running"
        stop_reason: StopReason | None = None
        await self._emit(
            state,
            EventType.RUN_STARTED,
            goal=state.goal,
            config_digest=self._config.digest(),
            model=self._config.model,
        )

        while state.status == "running":
            if state.step >= self._config.max_steps:
                stop_reason = StopReason.MAX_STEPS
                state.status = "failed"
                break
            budget_violation = self._budgets.exceeded(state)
            if budget_violation is not None:
                stop_reason = budget_violation
                state.status = "failed"
                break

            state.step += 1
            await self._emit(state, EventType.STEP_STARTED, step=state.step)
            stop_reason = await self._run_step(state)
            if state.status == "running":
                await self._emit(state, EventType.STEP_FINISHED, step=state.step)

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
        return state

    async def _run_step(self, state: AgentState) -> StopReason | None:
        messages_for_llm = await self._memory.load(state)
        tools = self._registry.list_specs(self._policy.allow)

        try:
            llm_result = await self._complete_with_retries(messages_for_llm, tools, state)
        except LLMError as exc:
            state.status = "failed"
            state.last_error = str(exc)
            await self._emit(state, EventType.ERROR, where="llm", message=str(exc))
            return StopReason.LLM_ERROR

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

        if not assistant.tool_calls:
            state.status = "succeeded"
            return StopReason.FINAL_ANSWER

        new_messages = [assistant]
        stop_reason = await self._handle_tool_calls(state, assistant.tool_calls, new_messages)
        await self._memory.commit(state, new_messages)
        return stop_reason

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
            if result is None:  # skipped after needs_input — unreachable, guarded above
                continue
            message = tool_response(call.id, call.name, result.content)
            state.messages.append(message)
            new_messages.append(message)

        violation = self._budgets.exceeded(state)
        if violation is not None:
            state.status = "failed"
            return violation
        return None

    async def _complete_with_retries(self, messages, tools, state: AgentState) -> LLMResult:
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
