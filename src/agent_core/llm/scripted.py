"""Deterministic fake LLM for tests and offline examples.

Returns pre-scripted assistant messages in order, ignoring the input.
"""

from __future__ import annotations

from agent_core.llm.base import LLMError, LLMResult
from agent_core.tools.base import ToolSpec
from agent_core.types import Message, Usage


class ScriptedLLM:
    def __init__(self, script: list[Message], usage_per_call: Usage | None = None) -> None:
        self._script = list(script)
        self._usage = usage_per_call or Usage(input_tokens=10, output_tokens=5)
        self.calls: list[list[Message]] = []  # inputs seen, for assertions

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if not self._script:
            raise LLMError("ScriptedLLM ran out of scripted messages")
        message = self._script.pop(0)
        return LLMResult(message=message, usage=self._usage, finish_reason="stop")
