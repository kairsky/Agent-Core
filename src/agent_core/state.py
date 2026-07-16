"""Run state: the single source of truth during one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.types import Message, Usage

RunStatus = Literal["running", "succeeded", "failed", "cancelled", "needs_input"]


@dataclass
class AgentState:
    run_id: str
    goal: str
    messages: list[Message]
    step: int = 0
    tool_calls_total: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    status: RunStatus = "running"
    last_error: str | None = None
    scratchpad: dict[str, Any] = field(default_factory=dict)

    def apply_usage(self, usage: Usage) -> None:
        self.tokens_in += usage.input_tokens
        self.tokens_out += usage.output_tokens
        self.cost_usd += usage.cost_usd

    @property
    def final_answer(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "assistant" and not message.tool_calls:
                return message.content
        return None
