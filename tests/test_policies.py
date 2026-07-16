from __future__ import annotations

from agent_core.config import AgentConfig, ToolAccessConfig
from agent_core.control.policies import ToolPolicy
from agent_core.tools.base import ToolSpec


def make_policy(allow: list[str], deny: list[str], confirm: list[str] | None = None) -> ToolPolicy:
    config = AgentConfig(
        model="m",
        tools=ToolAccessConfig(allow=allow, deny=deny),
        require_confirmation_for=confirm or [],
    )
    specs = {
        "write_file": ToolSpec(name="write_file", description="", dangerous=True),
        "calculator": ToolSpec(name="calculator", description=""),
    }
    return ToolPolicy(config, specs)


def test_glob_allow_and_deny():
    policy = make_policy(allow=["fs__*", "calculator"], deny=["fs__delete*"])
    assert policy.allow("fs__read_file")
    assert policy.allow("calculator")
    assert not policy.allow("fs__delete_file")  # deny wins
    assert not policy.allow("http_get")  # not in allow list


def test_dangerous_spec_needs_confirmation():
    policy = make_policy(allow=["*"], deny=[])
    assert policy.needs_confirmation("write_file", {})
    assert not policy.needs_confirmation("calculator", {})


def test_explicit_confirmation_list():
    policy = make_policy(allow=["*"], deny=[], confirm=["calculator"])
    assert policy.needs_confirmation("calculator", {})
