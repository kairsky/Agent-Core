"""Eval runner: executes a suite task by task and scores the outcomes."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_core.agent import Agent, AgentResult
from agent_core.evals.schema import EvalSuite, EvalTask, Expectation
from agent_core.llm.base import LLMProvider


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    status: str
    reason: str | None  # why it failed, if it did
    steps: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    trace_path: Path | None


def check_expectation(expect: Expectation, result: AgentResult, workspace: Path) -> str | None:
    """Return a failure reason, or None if all checks pass."""
    if result.status != "succeeded":
        return f"status={result.status} ({result.state.last_error})"
    answer = result.final_answer or ""
    if expect.contains is not None and expect.contains not in answer:
        return f"answer does not contain {expect.contains!r}"
    if expect.regex is not None and re.search(expect.regex, answer) is None:
        return f"answer does not match /{expect.regex}/"
    if expect.file is not None:
        path = workspace / expect.file.path
        if not path.is_file():
            return f"expected file missing: {expect.file.path}"
        if expect.file.text is not None and expect.file.text not in path.read_text(
            encoding="utf-8"
        ):
            return f"file {expect.file.path} does not contain {expect.file.text!r}"
    if expect.max_steps is not None and result.state.step > expect.max_steps:
        return f"took {result.state.step} steps, expected <= {expect.max_steps}"
    return None


async def _approve_all(tool_name: str, arguments: dict) -> bool:
    return True


class EvalRunner:
    def __init__(
        self,
        suite: EvalSuite,
        llm_for_task: Callable[[EvalTask], LLMProvider],
        output_dir: Path,
    ) -> None:
        self._suite = suite
        self._llm_for_task = llm_for_task
        self._output_dir = output_dir

    async def run(self) -> list[EvalResult]:
        results = []
        for task in self._suite.tasks:
            results.append(await self._run_task(task))
        return results

    async def _run_task(self, task: EvalTask) -> EvalResult:
        workspace = self._output_dir / task.id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        config = self._suite.agent.model_copy(
            update={"workspace": workspace, "tracing_path": self._output_dir / task.id / "traces"}
        )
        confirm = _approve_all if self._suite.auto_approve else None

        started = time.monotonic()
        async with Agent(config, self._llm_for_task(task), confirm=confirm) as agent:
            result = await agent.run(task.goal)
        latency = time.monotonic() - started

        reason = check_expectation(task.expect, result, workspace)
        state = result.state
        return EvalResult(
            task_id=task.id,
            passed=reason is None,
            status=result.status,
            reason=reason,
            steps=state.step,
            tool_calls=state.tool_calls_total,
            tokens_in=state.tokens_in,
            tokens_out=state.tokens_out,
            cost_usd=state.cost_usd,
            latency_s=latency,
            trace_path=result.trace_path,
        )


def format_report(suite_name: str, results: list[EvalResult]) -> str:
    lines = [f"eval suite: {suite_name}", ""]
    header = (
        f"{'task':<20} {'result':<7} {'steps':>5} {'tools':>5} "
        f"{'tokens':>9} {'cost $':>8} {'time s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        tokens = r.tokens_in + r.tokens_out
        lines.append(
            f"{r.task_id:<20} {verdict:<7} {r.steps:>5} {r.tool_calls:>5} "
            f"{tokens:>9} {r.cost_usd:>8.4f} {r.latency_s:>7.1f}"
        )
        if r.reason:
            lines.append(f"{'':<20} reason: {r.reason}")
    passed = sum(r.passed for r in results)
    total_cost = sum(r.cost_usd for r in results)
    lines.append("-" * len(header))
    lines.append(f"passed {passed}/{len(results)}   total cost ${total_cost:.4f}")
    return "\n".join(lines)
