"""Summary memory: compacts old turns into a summary message when over budget."""

from __future__ import annotations

from agent_core.llm.base import LLMProvider
from agent_core.memory.base import split_protected_head, total_tokens
from agent_core.state import AgentState
from agent_core.tracing import EventType, NullTracer, TraceEvent, Tracer
from agent_core.types import Message, system

SUMMARY_PROMPT = (
    "Summarize the following agent conversation turns concisely. Preserve facts, "
    "tool results, decisions and open questions. Output only the summary."
)

KEEP_RECENT_MESSAGES = 6


class SummaryMemory:
    def __init__(
        self,
        llm: LLMProvider,
        model: str,
        max_context_tokens: int,
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_tokens = max_context_tokens
        self._tracer = tracer or NullTracer()

    async def load(self, state: AgentState) -> list[Message]:
        return list(state.messages)

    async def commit(self, state: AgentState, new_messages: list[Message]) -> None:
        if total_tokens(state.messages) <= self._max_tokens:
            return
        head, tail = split_protected_head(state.messages)
        if len(tail) <= KEEP_RECENT_MESSAGES:
            return
        old, recent = tail[:-KEEP_RECENT_MESSAGES], tail[-KEEP_RECENT_MESSAGES:]
        # Do not split an assistant/tool-results group across the boundary.
        while recent and recent[0].role == "tool":
            old.append(recent.pop(0))
        if not old:
            return

        summary = await self._summarize(old)
        before = len(state.messages)
        state.messages[:] = [*head, system(f"Summary of earlier steps:\n{summary}"), *recent]
        await self._tracer.emit(
            TraceEvent(
                run_id=state.run_id,
                type=EventType.MEMORY_COMPACTED,
                step=state.step,
                payload={"before_msgs": before, "after_msgs": len(state.messages)},
            )
        )

    async def _summarize(self, messages: list[Message]) -> str:
        transcript = "\n".join(
            f"[{m.role}{f':{m.name}' if m.name else ''}] {m.content or m.tool_calls}"
            for m in messages
        )
        result = await self._llm.complete(
            [system(SUMMARY_PROMPT), Message(role="user", content=transcript)],
            tools=[],
            model=self._model,
            temperature=0.0,
        )
        return result.message.content or ""
