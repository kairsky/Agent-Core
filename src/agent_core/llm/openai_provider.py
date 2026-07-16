"""OpenAI chat completions adapter. Requires the `openai` extra."""

from __future__ import annotations

from agent_core.llm.base import LLMError, LLMResult
from agent_core.llm.messages import parse_tool_calls, to_openai_messages, to_openai_tools
from agent_core.tools.base import ToolSpec
from agent_core.types import Message, Usage

# USD per 1M tokens: model -> (input, output). Extend or override via constructor.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIProvider requires the openai SDK: pip install 'agent-core[openai]'"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._pricing = DEFAULT_PRICING | (pricing or {})

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float,
    ) -> LLMResult:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=to_openai_messages(messages),
                tools=to_openai_tools(tools) or None,  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise LLMError(f"OpenAI completion failed: {exc}") from exc

        choice = response.choices[0]
        try:
            tool_calls = parse_tool_calls(choice.message.tool_calls)
        except ValueError as exc:
            raise LLMError(str(exc)) from exc

        usage = Usage()
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost_usd=self._estimate_cost(
                    model, response.usage.prompt_tokens, response.usage.completion_tokens
                ),
            )
        return LLMResult(
            message=Message(
                role="assistant", content=choice.message.content, tool_calls=tool_calls
            ),
            usage=usage,
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def _estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        if model not in self._pricing:
            return 0.0
        price_in, price_out = self._pricing[model]
        return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
