"""Integration scenarios: ScriptedLLM driving the full Agent facade."""

from __future__ import annotations

from agent_core.agent import Agent
from agent_core.config import AgentConfig, ToolAccessConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import Message, ToolCall, assistant


def calculator_script() -> list[Message]:
    return [
        assistant(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="The answer is 4."),
    ]


async def test_calculator_scenario_succeeds(config: AgentConfig):
    llm = ScriptedLLM(calculator_script())
    result = await Agent(config, llm).run("What is 2+2?")

    assert result.status == "succeeded"
    assert result.final_answer == "The answer is 4."
    # The model saw the tool result before answering.
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert tool_messages[0].content == "4"
    assert tool_messages[0].tool_call_id == "c1"


async def test_tool_error_is_reported_to_model(config: AgentConfig):
    llm = ScriptedLLM(
        [
            assistant(
                tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "x+1"})]
            ),
            assistant(content="I could not compute that."),
        ]
    )
    result = await Agent(config, llm).run("Compute x+1")

    assert result.status == "succeeded"
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert "Cannot evaluate" in tool_messages[0].content


async def test_stops_on_max_steps(config: AgentConfig):
    config.max_steps = 3
    endless_call = assistant(tool_calls=[ToolCall(id="c", name="get_current_time", arguments={})])
    llm = ScriptedLLM([endless_call] * 10)
    result = await Agent(config, llm).run("Loop forever")

    assert result.status == "failed"
    assert result.state.step == 3
    assert result.state.last_error == "max_steps"


async def test_deny_policy_blocks_tool(config: AgentConfig):
    config.tools = ToolAccessConfig(allow=["*"], deny=["calculator"])
    llm = ScriptedLLM(calculator_script())
    result = await Agent(config, llm).run("What is 2+2?")

    assert result.status == "succeeded"
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert "not allowed by policy" in tool_messages[0].content
    assert result.state.tool_calls_total == 0  # nothing actually executed


async def test_denied_tool_is_hidden_from_llm(config: AgentConfig):
    config.tools = ToolAccessConfig(allow=["*"], deny=["calculator"])
    llm = ScriptedLLM([assistant(content="done")])
    await Agent(config, llm).run("anything")
    # The denied tool must not be advertised in specs either.
    agent = Agent(config, llm)
    from agent_core.control.policies import ToolPolicy

    policy = ToolPolicy(config)
    visible = agent.registry.list_specs(policy.allow)
    assert all(spec.name != "calculator" for spec in visible)


async def test_hitl_approved_write_executes(config: AgentConfig, tmp_path):
    approvals: list[str] = []

    async def confirm(name: str, args: dict) -> bool:
        approvals.append(name)
        return True

    llm = ScriptedLLM(
        [
            assistant(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "4"},
                    )
                ]
            ),
            assistant(content="Wrote 4 to out.txt"),
        ]
    )
    result = await Agent(config, llm, confirm=confirm).run("Write 4 to out.txt")

    assert result.status == "succeeded"
    assert approvals == ["write_file"]
    assert (config.workspace / "out.txt").read_text() == "4"


async def test_hitl_rejection_reaches_model(config: AgentConfig):
    async def confirm(name: str, args: dict) -> bool:
        return False

    llm = ScriptedLLM(
        [
            assistant(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "4"},
                    )
                ]
            ),
            assistant(content="User declined the write."),
        ]
    )
    result = await Agent(config, llm, confirm=confirm).run("Write 4 to out.txt")

    assert result.status == "succeeded"
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert "declined" in tool_messages[0].content
    assert not (config.workspace / "out.txt").exists()


async def test_dangerous_tool_without_confirm_needs_input(config: AgentConfig):
    llm = ScriptedLLM(
        [
            assistant(
                tool_calls=[
                    ToolCall(
                        id="w1",
                        name="write_file",
                        arguments={"path": "out.txt", "content": "4"},
                    )
                ]
            )
        ]
    )
    result = await Agent(config, llm).run("Write 4 to out.txt")

    assert result.status == "needs_input"
    assert result.state.scratchpad["pending_confirmation"]["name"] == "write_file"


async def test_tool_call_budget_fails_run(config: AgentConfig):
    config.max_total_tool_calls = 2
    call = assistant(tool_calls=[ToolCall(id="c", name="get_current_time", arguments={})])
    llm = ScriptedLLM([call] * 10)
    result = await Agent(config, llm).run("Loop")

    assert result.status == "failed"
    assert result.state.tool_calls_total == 2
    assert result.state.last_error == "budget_tool_calls"


async def test_max_tool_calls_per_step_is_enforced(config: AgentConfig):
    config.max_tool_calls_per_step = 2
    calls = [ToolCall(id=f"c{i}", name="get_current_time", arguments={}) for i in range(5)]
    llm = ScriptedLLM([assistant(tool_calls=calls), assistant(content="done")])
    result = await Agent(config, llm).run("Many calls")

    assert result.status == "succeeded"
    assert result.state.tool_calls_total == 2
