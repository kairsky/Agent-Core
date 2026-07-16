"""Plan-Execute loop scenarios with a scripted LLM."""

from __future__ import annotations

from agent_core.agent import Agent
from agent_core.config import AgentConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.loop.plan_execute import parse_plan
from agent_core.types import Message, ToolCall, assistant


def test_parse_plan_plain_json():
    assert parse_plan('["a", "b"]') == ["a", "b"]


def test_parse_plan_with_prose_and_fences():
    content = 'Here is my plan:\n```json\n["compute", "write file"]\n```\nDone.'
    assert parse_plan(content) == ["compute", "write file"]


def test_parse_plan_rejects_garbage():
    assert parse_plan("no plan here") is None
    assert parse_plan("[1, 2, 3]") is None
    assert parse_plan(None) is None


def plan_execute_script() -> list[Message]:
    return [
        # Phase 1: the plan.
        assistant(content='["calculate 2+2", "report the result"]'),
        # Task 1: uses the calculator, then reports task completion.
        assistant(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="Calculated: 4"),
        # Task 2: no tools needed.
        assistant(content="Result reported: 4"),
        # Phase 3: final answer.
        assistant(content="2+2 equals 4."),
    ]


async def test_plan_execute_end_to_end(config: AgentConfig):
    config.loop = "plan_execute"
    llm = ScriptedLLM(plan_execute_script())
    result = await Agent(config, llm).run("What is 2+2? Report the result.")

    assert result.status == "succeeded"
    assert result.final_answer == "2+2 equals 4."
    plan = result.state.scratchpad["plan"]
    assert [entry["status"] for entry in plan] == ["done", "done"]
    assert plan[0]["result"] == "Calculated: 4"
    # 5 LLM turns = 5 steps (plan, 2 for task 1, 1 for task 2, final).
    assert result.state.step == 5


async def test_unparseable_plan_degrades_to_single_task(config: AgentConfig):
    config.loop = "plan_execute"
    llm = ScriptedLLM(
        [
            assistant(content="I will just do it."),  # not a JSON plan
            assistant(content="Task done: 4"),
            assistant(content="The answer is 4."),
        ]
    )
    result = await Agent(config, llm).run("What is 2+2?")

    assert result.status == "succeeded"
    plan = result.state.scratchpad["plan"]
    assert len(plan) == 1
    assert plan[0]["task"] == "What is 2+2?"


async def test_max_steps_fails_mid_plan(config: AgentConfig):
    config.loop = "plan_execute"
    config.max_steps = 2
    endless = assistant(
        tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "1+1"})]
    )
    llm = ScriptedLLM([assistant(content='["loop forever"]'), *[endless] * 10])
    result = await Agent(config, llm).run("Loop")

    assert result.status == "failed"
    assert result.state.last_error == "max_steps"
    assert result.state.scratchpad["plan"][0]["status"] == "failed"


async def test_dangerous_tool_still_needs_input(config: AgentConfig):
    config.loop = "plan_execute"
    llm = ScriptedLLM(
        [
            assistant(content='["write the file"]'),
            assistant(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "4"},
                    )
                ]
            ),
        ]
    )
    result = await Agent(config, llm).run("Write 4 to out.txt")

    assert result.status == "needs_input"
    assert result.state.scratchpad["pending_confirmation"]["name"] == "write_file"
