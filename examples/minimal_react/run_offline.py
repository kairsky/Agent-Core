"""Offline demo: full agent loop with a scripted LLM — no API key required.

Run from the repo root:
    python examples/minimal_react/run_offline.py
Then inspect the trace:
    agent-core replay examples/minimal_react/traces/<run_id>.jsonl
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_core import Agent, AgentConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import ToolCall, assistant

HERE = Path(__file__).parent

SCRIPT = [
    assistant(
        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "17 * 19"})]
    ),
    assistant(
        tool_calls=[
            ToolCall(
                id="w1",
                name="write_file",
                arguments={"path": "result.txt", "content": "17 * 19 = 323"},
            )
        ]
    ),
    assistant(content="17 * 19 = 323. I saved the result to result.txt."),
]


async def approve_everything(tool_name: str, arguments: dict) -> bool:
    print(f"[HITL] auto-approving {tool_name}({arguments})")
    return True


async def main() -> None:
    config = AgentConfig(
        model="scripted-demo",
        workspace=HERE / "workspace",
        tracing_path=HERE / "traces",
    )
    agent = Agent(config, ScriptedLLM(SCRIPT), confirm=approve_everything)
    result = await agent.run("Calculate 17*19 and write the result to result.txt")

    print(f"status: {result.status}")
    print(f"answer: {result.final_answer}")
    print(f"trace:  {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
