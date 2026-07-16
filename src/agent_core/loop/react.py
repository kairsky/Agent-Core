"""ReAct loop — the reference implementation of the runtime contract.

Per step: load memory -> LLM completion -> execute requested tools -> commit memory.
The run ends on a final answer (no tool calls), max_steps, a budget violation,
an unrecoverable LLM error, or a dangerous tool awaiting human input.
"""

from __future__ import annotations

from agent_core.control.stop import StopReason
from agent_core.loop.common import LoopBase
from agent_core.state import AgentState
from agent_core.tracing import EventType


class ReActLoop(LoopBase):
    async def run(self, state: AgentState) -> AgentState:
        state.status = "running"
        stop_reason: StopReason | None = None
        await self._emit_run_started(state)

        while state.status == "running":
            stop_reason = self._guard(state)
            if stop_reason is not None:
                break

            state.step += 1
            await self._emit(state, EventType.STEP_STARTED, step=state.step)
            stop_reason = await self._run_step(state)
            if state.status == "running":
                await self._emit(state, EventType.STEP_FINISHED, step=state.step)

        await self._emit_run_finished(state, stop_reason)
        return state

    async def _run_step(self, state: AgentState) -> StopReason | None:
        messages_for_llm = await self._memory.load(state)
        tools = self._registry.list_specs(self._policy.allow)

        assistant = await self._llm_turn(state, messages_for_llm, tools)
        if assistant is None:
            return StopReason.LLM_ERROR

        if not assistant.tool_calls:
            state.status = "succeeded"
            return StopReason.FINAL_ANSWER

        new_messages = [assistant]
        stop_reason = await self._handle_tool_calls(state, assistant.tool_calls, new_messages)
        await self._memory.commit(state, new_messages)
        return stop_reason
