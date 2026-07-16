"""Resuming a run paused on needs_input (HITL over API-style flow)."""

from __future__ import annotations

import pytest

from agent_core.agent import Agent
from agent_core.config import AgentConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import Message, ToolCall, assistant


def write_then_answer_script() -> list[Message]:
    return [
        assistant(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "out.txt", "content": "4"})
            ]
        ),
        assistant(content="Wrote 4 to out.txt"),
    ]


async def test_resume_approved_executes_and_finishes(config: AgentConfig):
    llm = ScriptedLLM(write_then_answer_script())
    agent = Agent(config, llm)  # no confirm callback -> needs_input

    paused = await agent.run("Write 4 to out.txt")
    assert paused.status == "needs_input"

    resumed = await agent.resume(paused.state, approved=True)
    assert resumed.status == "succeeded"
    assert resumed.final_answer == "Wrote 4 to out.txt"
    assert (config.workspace / "out.txt").read_text() == "4"
    # The model saw the real tool result after resume.
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert "Wrote 1 chars" in tool_messages[0].content


async def test_resume_denied_reports_to_model(config: AgentConfig):
    llm = ScriptedLLM(
        [
            write_then_answer_script()[0],
            assistant(content="Understood, not writing the file."),
        ]
    )
    agent = Agent(config, llm)

    paused = await agent.run("Write 4 to out.txt")
    resumed = await agent.resume(paused.state, approved=False)

    assert resumed.status == "succeeded"
    assert not (config.workspace / "out.txt").exists()
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert "declined" in tool_messages[0].content


async def test_resume_requires_needs_input(config: AgentConfig):
    llm = ScriptedLLM([assistant(content="done")])
    agent = Agent(config, llm)
    finished = await agent.run("Say done")

    with pytest.raises(ValueError, match="not awaiting confirmation"):
        await agent.resume(finished.state, approved=True)


async def test_resume_appends_to_same_trace(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    llm = ScriptedLLM(write_then_answer_script())
    agent = Agent(config, llm)

    paused = await agent.run("Write 4 to out.txt")
    resumed = await agent.resume(paused.state, approved=True)

    assert resumed.trace_path == paused.trace_path
    lines = resumed.trace_path.read_text(encoding="utf-8").splitlines()
    types = [__import__("json").loads(line)["type"] for line in lines]
    assert types.count("run.finished") == 2  # pause + final
    assert "run.resumed" in types
