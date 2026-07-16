"""Agent facade: wires config, LLM, tools, memory, policy, tracing and the loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from agent_core.config import AgentConfig
from agent_core.control.policies import ToolPolicy
from agent_core.llm.base import LLMProvider
from agent_core.loop.plan_execute import PlanExecuteLoop
from agent_core.loop.react import ReActLoop
from agent_core.memory.base import MemoryStore
from agent_core.memory.buffer import BufferMemory
from agent_core.memory.summary import SummaryMemory
from agent_core.state import AgentState
from agent_core.tools.base import ConfirmCallback, ToolHandler
from agent_core.tools.builtin import builtin_tools
from agent_core.tools.mcp_adapter import McpConnection
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.runtime import ToolRuntime
from agent_core.tracing import JsonlTracer, NullTracer, RedactingTracer, Tracer
from agent_core.types import Message, system, user


@dataclass
class AgentResult:
    run_id: str
    status: str
    final_answer: str | None
    state: AgentState
    trace_path: Path | None


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMProvider,
        *,
        tools: list[ToolHandler] | None = None,
        include_builtin_tools: bool = True,
        memory: MemoryStore | None = None,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._confirm = confirm
        self._connections: list[McpConnection] = []

        self.registry = ToolRegistry()
        if include_builtin_tools:
            for handler in builtin_tools(config.workspace):
                self.registry.register(handler)
        for handler in tools or []:
            self.registry.register(handler)

        self._tracer: Tracer = NullTracer()
        self._trace_dir = config.tracing_path
        self._memory = memory

    async def __aenter__(self) -> Self:
        for server_config in self._config.mcp_servers:
            connection = McpConnection(server_config)
            await connection.connect()
            await connection.register_tools(self.registry)
            self._connections.append(connection)
        return self

    async def __aexit__(self, *exc_info) -> None:
        for connection in reversed(self._connections):
            await connection.close()
        self._connections.clear()

    def prepare_state(
        self, goal: str, *, session_messages: list[Message] | None = None
    ) -> AgentState:
        """Create the initial run state. Exposed so callers (API) can observe it live."""
        messages = [system(self._config.system_prompt), *(session_messages or []), user(goal)]
        return AgentState(run_id=str(uuid.uuid4()), goal=goal, messages=messages)

    async def run(self, goal: str, *, session_messages: list[Message] | None = None) -> AgentResult:
        state = self.prepare_state(goal, session_messages=session_messages)
        return await self.run_state(state)

    async def run_state(self, state: AgentState) -> AgentResult:
        tracer = self._make_tracer(state.run_id)
        loop = self._build_loop(tracer)
        try:
            state = await loop.run(state)
        finally:
            await tracer.close()
        return self._result(state)

    async def resume(self, state: AgentState, approved: bool) -> AgentResult:
        """Continue a run paused on needs_input; the trace file is appended to."""
        if self._config.loop != "react":
            raise NotImplementedError("resume is supported for the react loop only")
        tracer = self._make_tracer(state.run_id)
        loop = self._build_loop(tracer)
        try:
            state = await loop.resume(state, approved)
        finally:
            await tracer.close()
        return self._result(state)

    def trace_path_for(self, run_id: str) -> Path | None:
        return self._trace_dir / f"{run_id}.jsonl" if self._trace_dir is not None else None

    def _make_tracer(self, run_id: str) -> Tracer:
        trace_path = self.trace_path_for(run_id)
        if trace_path is None:
            return NullTracer()
        return RedactingTracer(JsonlTracer(trace_path))

    def _build_loop(self, tracer: Tracer) -> ReActLoop | PlanExecuteLoop:
        memory = self._memory or self._build_memory(tracer)
        policy = ToolPolicy(self._config, {spec.name: spec for spec in self.registry.list_specs()})
        loop_cls = PlanExecuteLoop if self._config.loop == "plan_execute" else ReActLoop
        return loop_cls(
            config=self._config,
            llm=self._llm,
            registry=self.registry,
            tool_runtime=ToolRuntime(self.registry, self._config, tracer),
            memory=memory,
            policy=policy,
            tracer=tracer,
            confirm=self._confirm,
        )

    def _result(self, state: AgentState) -> AgentResult:
        return AgentResult(
            run_id=state.run_id,
            status=state.status,
            final_answer=state.final_answer if state.status == "succeeded" else None,
            state=state,
            trace_path=self.trace_path_for(state.run_id),
        )

    def _build_memory(self, tracer: Tracer) -> MemoryStore:
        if self._config.memory == "summary":
            return SummaryMemory(
                llm=self._llm,
                model=self._config.model,
                max_context_tokens=self._config.max_context_tokens or 8000,
                tracer=tracer,
            )
        return BufferMemory(max_context_tokens=self._config.max_context_tokens)
