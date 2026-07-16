"""Plan-Execute loop: plan first, then work through tasks with focused ReAct steps.

Phases:
1. Plan   — one LLM turn (no tools) producing a JSON array of task strings;
            stored in state.scratchpad["plan"], emitted as plan.created.
2. Execute — each task runs ReAct-style turns until the model answers without
            tool calls; that answer is recorded as the task result.
3. Finalize — one last turn asking for the final answer to the original goal.

Shares max_steps and budgets with ReAct: every LLM turn is one step.
"""

from __future__ import annotations

import json
import re

from agent_core.control.stop import StopReason
from agent_core.loop.common import LoopBase
from agent_core.state import AgentState
from agent_core.tracing import EventType
from agent_core.types import user

PLAN_PROMPT = (
    "Break the goal above into a short ordered plan of concrete tasks. "
    "Respond with ONLY a JSON array of task strings, e.g. "
    '["compute the sum", "write it to a file"]. Use 1-6 tasks.'
)

TASK_PROMPT = (
    "Now execute task {index} of {total}: {task}\n"
    "Use tools if needed. When this task is complete, reply with a short plain-text "
    "result and no tool calls."
)

FINAL_PROMPT = (
    "All plan tasks are complete. Provide the final answer to the original goal "
    "in plain text, without tool calls."
)


def parse_plan(content: str | None) -> list[str] | None:
    """Extract a JSON array of task strings; tolerate surrounding prose/code fences."""
    if not content:
        return None
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tasks = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
    return tasks or None


class PlanExecuteLoop(LoopBase):
    async def run(self, state: AgentState) -> AgentState:
        state.status = "running"
        await self._emit_run_started(state)

        stop_reason = await self._plan_phase(state)
        if stop_reason is None:
            stop_reason = await self._execute_phase(state)
        if stop_reason is None:
            stop_reason = await self._finalize_phase(state)

        await self._emit_run_finished(state, stop_reason)
        return state

    async def _plan_phase(self, state: AgentState) -> StopReason | None:
        stop_reason = self._guard(state)
        if stop_reason is not None:
            return stop_reason
        state.step += 1
        await self._emit(state, EventType.STEP_STARTED, step=state.step, phase="plan")

        state.messages.append(user(PLAN_PROMPT))
        assistant = await self._llm_turn(state, await self._memory.load(state), tools=[])
        if assistant is None:
            return StopReason.LLM_ERROR

        tasks = parse_plan(assistant.content)
        if tasks is None:
            tasks = [state.goal]  # degenerate plan: treat the whole goal as one task
        state.scratchpad["plan"] = [
            {"id": index + 1, "task": task, "status": "pending", "result": None}
            for index, task in enumerate(tasks)
        ]
        await self._emit(state, EventType.PLAN_CREATED, tasks=tasks)
        await self._emit(state, EventType.STEP_FINISHED, step=state.step, phase="plan")
        return None

    async def _execute_phase(self, state: AgentState) -> StopReason | None:
        plan: list[dict] = state.scratchpad["plan"]
        for entry in plan:
            await self._emit(
                state, EventType.PLAN_TASK_STARTED, task_id=entry["id"], task=entry["task"]
            )
            entry["status"] = "running"
            state.messages.append(
                user(TASK_PROMPT.format(index=entry["id"], total=len(plan), task=entry["task"]))
            )

            stop_reason = await self._run_task(state, entry)
            await self._emit(
                state,
                EventType.PLAN_TASK_FINISHED,
                task_id=entry["id"],
                status=entry["status"],
                result_preview=(entry["result"] or "")[:200],
            )
            if stop_reason is not None:
                return stop_reason
        return None

    async def _run_task(self, state: AgentState, entry: dict) -> StopReason | None:
        """ReAct-style turns scoped to one task; ends when the model replies without tools."""
        while True:
            stop_reason = self._guard(state)
            if stop_reason is not None:
                entry["status"] = "failed"
                return stop_reason

            state.step += 1
            await self._emit(state, EventType.STEP_STARTED, step=state.step, task_id=entry["id"])
            tools = self._registry.list_specs(self._policy.allow)
            assistant = await self._llm_turn(state, await self._memory.load(state), tools)
            if assistant is None:
                entry["status"] = "failed"
                return StopReason.LLM_ERROR

            if not assistant.tool_calls:
                entry["status"] = "done"
                entry["result"] = assistant.content
                await self._emit(state, EventType.STEP_FINISHED, step=state.step)
                return None

            new_messages = [assistant]
            stop_reason = await self._handle_tool_calls(state, assistant.tool_calls, new_messages)
            await self._memory.commit(state, new_messages)
            if stop_reason is not None:
                entry["status"] = "failed"
                return stop_reason
            await self._emit(state, EventType.STEP_FINISHED, step=state.step)

    async def _finalize_phase(self, state: AgentState) -> StopReason | None:
        stop_reason = self._guard(state)
        if stop_reason is not None:
            return stop_reason
        state.step += 1
        await self._emit(state, EventType.STEP_STARTED, step=state.step, phase="final")

        state.messages.append(user(FINAL_PROMPT))
        assistant = await self._llm_turn(state, await self._memory.load(state), tools=[])
        if assistant is None:
            return StopReason.LLM_ERROR

        state.status = "succeeded"
        return StopReason.FINAL_ANSWER
