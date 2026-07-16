from __future__ import annotations

from agent_core.memory.buffer import BufferMemory
from agent_core.state import AgentState
from agent_core.types import ToolCall, assistant, system, tool_response, user


def make_state(messages) -> AgentState:
    return AgentState(run_id="r", goal="g", messages=messages)


async def test_no_limit_returns_everything():
    state = make_state([system("s"), user("goal"), assistant("a")])
    memory = BufferMemory()
    assert len(await memory.load(state)) == 3


async def test_trim_keeps_system_and_goal():
    long_text = "x" * 400  # ~100 tokens each
    messages = [system("sys"), user("goal")] + [assistant(long_text) for _ in range(10)]
    state = make_state(messages)
    memory = BufferMemory(max_context_tokens=350)

    loaded = await memory.load(state)
    assert loaded[0].content == "sys"
    assert loaded[1].content == "goal"
    assert len(loaded) < len(messages)
    assert len(state.messages) == len(messages)  # state itself is untouched


async def test_trim_never_orphans_tool_messages():
    call = ToolCall(id="t1", name="calculator", arguments={"expression": "1"})
    messages = [
        system("sys"),
        user("goal"),
        assistant("x" * 400, tool_calls=[call]),
        tool_response("t1", "calculator", "y" * 400),
        assistant("z" * 40),
    ]
    state = make_state(messages)
    memory = BufferMemory(max_context_tokens=120)

    loaded = await memory.load(state)
    # If the assistant tool-call message is dropped, its tool result must go too.
    tool_ids_present = {m.tool_call_id for m in loaded if m.role == "tool"}
    assistant_call_ids = {
        c.id for m in loaded if m.role == "assistant" for c in (m.tool_calls or [])
    }
    assert tool_ids_present <= assistant_call_ids
