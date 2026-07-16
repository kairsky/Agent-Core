"""HTTP API: start runs, poll status, confirm dangerous tools, read traces.

Runs execute as background asyncio tasks; the store keeps live AgentState
objects, so GET /runs/{id} reflects progress in real time. A run that pauses
on a dangerous tool reports status=needs_input and waits for
POST /runs/{id}/confirm.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_core.agent import Agent
from agent_core.state import AgentState


@dataclass
class RunRecord:
    state: AgentState
    task: asyncio.Task | None = None
    resumes: int = field(default=0)


class CreateRunRequest(BaseModel):
    goal: str


class ConfirmRequest(BaseModel):
    approved: bool


def _run_summary(record: RunRecord) -> dict[str, Any]:
    state = record.state
    return {
        "run_id": state.run_id,
        "status": state.status,
        "goal": state.goal,
        "steps": state.step,
        "tool_calls_total": state.tool_calls_total,
        "final_answer": state.final_answer if state.status == "succeeded" else None,
        "last_error": state.last_error,
        "pending_confirmation": state.scratchpad.get("pending_confirmation"),
        "usage": {
            "tokens_in": state.tokens_in,
            "tokens_out": state.tokens_out,
            "cost_usd": round(state.cost_usd, 6),
        },
    }


def create_app(agent: Agent):
    """Build the FastAPI app around an already-constructed Agent.

    The caller owns the agent lifecycle (MCP connect/close via async context).
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The HTTP API requires fastapi: pip install 'agent-core[api]'") from exc

    app = FastAPI(title="agent-core", version="0.1.0")
    runs: dict[str, RunRecord] = {}

    def get_record(run_id: str) -> RunRecord:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        return record

    @app.post("/runs", status_code=202)
    async def create_run(request: CreateRunRequest) -> dict[str, Any]:
        state = agent.prepare_state(request.goal)
        record = RunRecord(state=state)
        record.task = asyncio.create_task(agent.run_state(state))
        runs[state.run_id] = record
        return {"run_id": state.run_id, "status": state.status}

    @app.get("/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return [_run_summary(record) for record in runs.values()]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        return _run_summary(get_record(run_id))

    @app.post("/runs/{run_id}/confirm", status_code=202)
    async def confirm_run(run_id: str, request: ConfirmRequest) -> dict[str, Any]:
        record = get_record(run_id)
        if record.state.status != "needs_input":
            raise HTTPException(
                status_code=409,
                detail=f"run is not awaiting confirmation (status={record.state.status})",
            )
        record.resumes += 1
        record.task = asyncio.create_task(agent.resume(record.state, request.approved))
        return {"run_id": run_id, "status": "running", "approved": request.approved}

    @app.get("/runs/{run_id}/trace")
    async def get_trace(run_id: str) -> list[dict[str, Any]]:
        get_record(run_id)  # 404 for unknown runs
        trace_path = agent.trace_path_for(run_id)
        if trace_path is None or not trace_path.is_file():
            raise HTTPException(status_code=404, detail="tracing is disabled or trace missing")
        return [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return app
