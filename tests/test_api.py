"""HTTP API scenarios over the ASGI transport, driven by scripted LLMs."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_core.agent import Agent
from agent_core.api import create_app
from agent_core.config import AgentConfig
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import Message, ToolCall, assistant


def make_client(config: AgentConfig, script: list[Message]) -> httpx.AsyncClient:
    agent = Agent(config, ScriptedLLM(script))
    app = create_app(agent)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def wait_for_status(client: httpx.AsyncClient, run_id: str, expected: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/runs/{run_id}")
        data = response.json()
        if data["status"] == expected:
            return data
        await asyncio.sleep(0.01)
    pytest.fail(f"run never reached status {expected!r}, last: {data['status']}")


async def test_simple_run_lifecycle(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    script = [
        assistant(
            tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="The answer is 4."),
    ]
    async with make_client(config, script) as client:
        created = (await client.post("/runs", json={"goal": "What is 2+2?"})).json()
        data = await wait_for_status(client, created["run_id"], "succeeded")

        assert data["final_answer"] == "The answer is 4."
        assert data["usage"]["tokens_in"] > 0

        trace = (await client.get(f"/runs/{created['run_id']}/trace")).json()
        assert [e["type"] for e in trace][0] == "run.started"
        assert [e["type"] for e in trace][-1] == "run.finished"


async def test_needs_input_then_confirm(config: AgentConfig, tmp_path):
    config.tracing_path = tmp_path / "traces"
    script = [
        assistant(
            tool_calls=[
                ToolCall(id="w1", name="write_file", arguments={"path": "out.txt", "content": "4"})
            ]
        ),
        assistant(content="Wrote it."),
    ]
    async with make_client(config, script) as client:
        created = (await client.post("/runs", json={"goal": "Write 4 to out.txt"})).json()
        data = await wait_for_status(client, created["run_id"], "needs_input")
        assert data["pending_confirmation"]["name"] == "write_file"

        confirm = await client.post(f"/runs/{created['run_id']}/confirm", json={"approved": True})
        assert confirm.status_code == 202

        data = await wait_for_status(client, created["run_id"], "succeeded")
        assert data["final_answer"] == "Wrote it."
        assert (config.workspace / "out.txt").read_text() == "4"


async def test_confirm_conflicts_when_not_paused(config: AgentConfig):
    script = [assistant(content="done")]
    async with make_client(config, script) as client:
        created = (await client.post("/runs", json={"goal": "Say done"})).json()
        await wait_for_status(client, created["run_id"], "succeeded")

        response = await client.post(f"/runs/{created['run_id']}/confirm", json={"approved": True})
        assert response.status_code == 409


async def test_unknown_run_is_404(config: AgentConfig):
    async with make_client(config, []) as client:
        assert (await client.get("/runs/nope")).status_code == 404
        assert (await client.post("/runs/nope/confirm", json={"approved": True})).status_code == 404


async def test_list_runs(config: AgentConfig):
    script = [assistant(content="one")]
    async with make_client(config, script) as client:
        created = (await client.post("/runs", json={"goal": "g"})).json()
        await wait_for_status(client, created["run_id"], "succeeded")
        runs = (await client.get("/runs")).json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == created["run_id"]
