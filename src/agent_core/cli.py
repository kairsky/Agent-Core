"""CLI: `agent-core run "goal" --config agent.yaml` and `agent-core replay trace.jsonl`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from agent_core.agent import Agent
from agent_core.config import AgentConfig


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the nearest .env (cwd or ancestors) into os.environ."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        path = directory / ".env"
        if path.is_file():
            break
    else:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


async def _cli_confirm(tool_name: str, arguments: dict[str, Any]) -> bool:
    print(f"\n[HITL] Tool '{tool_name}' wants to run with arguments:")
    print(json.dumps(arguments, indent=2, ensure_ascii=False))
    answer = await asyncio.to_thread(input, f"Approve {tool_name}? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


async def _run(goal: str, config_path: Path | None, trace_dir: Path | None) -> int:
    if config_path is not None:
        config = AgentConfig.from_yaml(config_path)
    else:
        config = AgentConfig(model="gpt-4.1-mini")
    if trace_dir is not None:
        config.tracing_path = trace_dir
    if config.tracing_path is None:
        config.tracing_path = Path("traces")

    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is not set.\n"
            "Add it to a .env file in the current directory (see .env.example) "
            "or set the environment variable.",
            file=sys.stderr,
        )
        return 1

    from agent_core.llm.openai_provider import OpenAIProvider

    async with Agent(config, OpenAIProvider(), confirm=_cli_confirm) as agent:
        result = await agent.run(goal)

    print(f"\nrun_id: {result.run_id}")
    print(f"status: {result.status}")
    if result.trace_path:
        print(f"trace:  {result.trace_path}")
    if result.final_answer:
        print(f"\n{result.final_answer}")
    elif result.state.last_error:
        print(f"error: {result.state.last_error}", file=sys.stderr)
    return 0 if result.status == "succeeded" else 1


def _latest_trace(trace_dir: Path = Path("traces")) -> Path | None:
    traces = sorted(trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


async def _eval(suite_path: Path, output_dir: Path) -> int:
    from agent_core.evals import EvalRunner, EvalSuite, format_report

    suite = EvalSuite.from_yaml(suite_path)
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    from agent_core.llm.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    runner = EvalRunner(suite, lambda task: provider, output_dir)
    results = await runner.run()
    print(format_report(suite.name, results))
    return 0 if all(r.passed for r in results) else 1


def _replay(trace_path: Path | None) -> int:
    """Pretty-print a JSONL trace, one line per event."""
    if trace_path is None:
        trace_path = _latest_trace()
        if trace_path is None:
            print("no traces found in ./traces", file=sys.stderr)
            return 1
        print(f"replaying latest trace: {trace_path}\n")
    if not trace_path.is_file():
        print(f"no such trace: {trace_path}", file=sys.stderr)
        return 1
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload", {})
        summary = _summarize_payload(event["type"], payload)
        print(f"[step {event.get('step', 0):>2}] {event['type']:<22} {summary}")
    return 0


def _summarize_payload(event_type: str, payload: dict[str, Any]) -> str:
    match event_type:
        case "run.started":
            return f"goal={payload.get('goal', '')!r} model={payload.get('model')}"
        case "llm.completed":
            usage = payload.get("usage", {})
            return (
                f"tool_calls={payload.get('has_tool_calls')} "
                f"in={usage.get('in')} out={usage.get('out')}"
            )
        case "tool.requested":
            return f"{payload.get('name')}({json.dumps(payload.get('arguments', {}))[:80]})"
        case "tool.executed":
            status = "ok" if payload.get("ok") else f"FAIL:{payload.get('error_type')}"
            return (
                f"{payload.get('name')} {status} {payload.get('latency_ms')}ms "
                f"-> {payload.get('content_preview', '')!r}"
            )
        case "run.finished":
            return f"status={payload.get('status')} answer={payload.get('final_answer')!r}"
        case _:
            return json.dumps(payload, ensure_ascii=False)[:120]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-core", description="Minimal agent runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the agent on a goal")
    run_parser.add_argument("goal")
    run_parser.add_argument("--config", type=Path, default=None, help="path to agent.yaml")
    run_parser.add_argument("--trace-dir", type=Path, default=None, help="traces directory")

    replay_parser = subparsers.add_parser("replay", help="pretty-print a JSONL trace")
    replay_parser.add_argument(
        "trace", type=Path, nargs="?", default=None, help="trace file (default: latest in ./traces)"
    )

    eval_parser = subparsers.add_parser("eval", help="run an eval suite")
    eval_parser.add_argument("suite", type=Path, help="path to suite yaml")
    eval_parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_runs"), help="workspaces and traces go here"
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args.goal, args.config, args.trace_dir))
    if args.command == "eval":
        return asyncio.run(_eval(args.suite, args.output_dir))
    return _replay(args.trace)


if __name__ == "__main__":
    sys.exit(main())
