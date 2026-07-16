"""Stop reasons recorded when a run terminates without a final answer."""

from __future__ import annotations

from enum import StrEnum


class StopReason(StrEnum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    BUDGET_TOOL_CALLS = "budget_tool_calls"
    BUDGET_TOKENS = "budget_tokens"
    BUDGET_COST = "budget_cost"
    LLM_ERROR = "llm_error"
    NEEDS_INPUT = "needs_input"
    CANCELLED = "cancelled"
