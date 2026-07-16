"""Golden traces: the normalized event stream is a behavioral contract of the loop.

Regenerate with: UPDATE_GOLDEN=1 pytest tests/test_golden_traces.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_core.agent import Agent
from agent_core.config import AgentConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import Message, ToolCall, assistant

GOLDEN_DIR = Path(__file__).parent / "golden"

VOLATILE_KEYS = {"ts", "run_id"}
VOLATILE_PAYLOAD_KEYS = {"latency_ms", "config_digest"}


def normalize(raw_lines: list[str]) -> list[dict]:
    events = []
    for line in raw_lines:
        event = json.loads(line)
        for key in VOLATILE_KEYS:
            event.pop(key, None)
        for key in VOLATILE_PAYLOAD_KEYS:
            if key in event.get("payload", {}):
                event["payload"][key] = "<volatile>"
        events.append(event)
    return events


async def run_and_normalize(config: AgentConfig, script: list[Message], goal: str) -> list[dict]:
    result = await Agent(config, ScriptedLLM(script)).run(goal)
    assert result.trace_path is not None
    lines = result.trace_path.read_text(encoding="utf-8").splitlines()
    return normalize([line for line in lines if line.strip()])


def check_golden(name: str, events: list[dict]) -> None:
    golden_path = GOLDEN_DIR / f"{name}.jsonl"
    rendered = "\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in events) + "\n"
    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(rendered, encoding="utf-8")
        return
    if not golden_path.is_file():
        pytest.fail(f"golden file missing: {golden_path} (run with UPDATE_GOLDEN=1)")
    assert rendered == golden_path.read_text(encoding="utf-8")


async def test_golden_calculator_success(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    script = [
        assistant(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="The answer is 4."),
    ]
    events = await run_and_normalize(config, script, "What is 2+2?")
    check_golden("calculator_success", events)


async def test_golden_deny_policy(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    config.tools.deny = ["calculator"]
    script = [
        assistant(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="Calculator is not available."),
    ]
    events = await run_and_normalize(config, script, "What is 2+2?")
    check_golden("deny_policy", events)


async def test_golden_max_steps(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    config.max_steps = 2
    call = assistant(
        tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "1+1"})]
    )
    events = await run_and_normalize(config, [call] * 5, "Loop forever")
    check_golden("max_steps", events)
