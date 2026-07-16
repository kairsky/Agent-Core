"""Budget checks over the run state: tool calls, tokens, cost."""

from __future__ import annotations

from agent_core.config import AgentConfig
from agent_core.control.stop import StopReason
from agent_core.state import AgentState


class Budgets:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def exceeded(self, state: AgentState) -> StopReason | None:
        """Return the violated budget, or None if the run may continue."""
        if state.tool_calls_total >= self._config.max_total_tool_calls:
            return StopReason.BUDGET_TOOL_CALLS
        if (
            self._config.max_tokens_budget is not None
            and state.tokens_in + state.tokens_out >= self._config.max_tokens_budget
        ):
            return StopReason.BUDGET_TOKENS
        if self._config.max_cost_usd is not None and state.cost_usd >= self._config.max_cost_usd:
            return StopReason.BUDGET_COST
        return None
