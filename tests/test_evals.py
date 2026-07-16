"""Eval harness: expectation checks and the runner, driven by scripted LLMs."""

from __future__ import annotations

from pathlib import Path

from agent_core.config import AgentConfig
from agent_core.evals import EvalRunner, EvalSuite, EvalTask, Expectation, format_report
from agent_core.evals.schema import FileExpectation
from agent_core.llm.scripted import ScriptedLLM
from agent_core.types import ToolCall, assistant


def make_suite(tasks: list[EvalTask], tmp_path: Path) -> EvalSuite:
    return EvalSuite(
        name="test",
        agent=AgentConfig(model="scripted", workspace=tmp_path / "unused"),
        auto_approve=True,
        tasks=tasks,
    )


async def test_passing_and_failing_tasks(tmp_path):
    tasks = [
        EvalTask(id="ok", goal="2+2?", expect=Expectation(contains="4")),
        EvalTask(id="wrong_answer", goal="2+2?", expect=Expectation(contains="5")),
    ]
    scripts = {
        "ok": [assistant(content="The answer is 4.")],
        "wrong_answer": [assistant(content="The answer is 4.")],
    }
    runner = EvalRunner(
        make_suite(tasks, tmp_path),
        llm_for_task=lambda task: ScriptedLLM(list(scripts[task.id])),
        output_dir=tmp_path / "runs",
    )
    results = await runner.run()

    assert results[0].passed
    assert not results[1].passed
    assert "does not contain" in results[1].reason


async def test_file_expectation_checks_task_workspace(tmp_path):
    task = EvalTask(
        id="write",
        goal="write 4 to out.txt",
        expect=Expectation(file=FileExpectation(path="out.txt", text="4")),
    )
    script = [
        assistant(
            tool_calls=[
                ToolCall(id="w", name="write_file", arguments={"path": "out.txt", "content": "4"})
            ]
        ),
        assistant(content="done"),
    ]
    runner = EvalRunner(
        make_suite([task], tmp_path),
        llm_for_task=lambda t: ScriptedLLM(list(script)),
        output_dir=tmp_path / "runs",
    )
    results = await runner.run()

    assert results[0].passed
    assert (tmp_path / "runs" / "write" / "workspace" / "out.txt").read_text() == "4"
    assert results[0].trace_path is not None  # each task gets its own trace


async def test_failed_run_reports_status(tmp_path):
    task = EvalTask(id="fail", goal="loop", expect=Expectation(contains="x"))
    endless = assistant(
        tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "1"})]
    )
    suite = make_suite([task], tmp_path)
    suite.agent.max_steps = 2
    runner = EvalRunner(
        suite, llm_for_task=lambda t: ScriptedLLM([endless] * 5), output_dir=tmp_path / "runs"
    )
    results = await runner.run()

    assert not results[0].passed
    assert "status=failed" in results[0].reason


async def test_max_steps_expectation(tmp_path):
    task = EvalTask(id="slow", goal="2+2?", expect=Expectation(contains="4", max_steps=1))
    script = [
        assistant(
            tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "2+2"})]
        ),
        assistant(content="4"),
    ]
    runner = EvalRunner(
        make_suite([task], tmp_path),
        llm_for_task=lambda t: ScriptedLLM(list(script)),
        output_dir=tmp_path / "runs",
    )
    results = await runner.run()

    assert not results[0].passed
    assert "took 2 steps" in results[0].reason


async def test_report_formatting(tmp_path):
    task = EvalTask(id="ok", goal="2+2?", expect=Expectation(contains="4"))
    runner = EvalRunner(
        make_suite([task], tmp_path),
        llm_for_task=lambda t: ScriptedLLM([assistant(content="4")]),
        output_dir=tmp_path / "runs",
    )
    results = await runner.run()
    report = format_report("test", results)

    assert "passed 1/1" in report
    assert "PASS" in report
