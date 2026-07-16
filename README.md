# agent-core

Minimal tool-using agent runtime: a hand-rolled ReAct loop, strict tool
execution, pluggable memory, JSONL traces, and an MCP adapter — no agent
framework as a core dependency.

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│ CLI / API   │────▶│                 AgentRuntime                 │
└─────────────┘     │     Loop → Memory → Tools → Tracer           │
                    └───────────┬───────────────────┬──────────────┘
                                │                   │
                    ┌───────────▼────────┐   ┌──────▼──────────┐
                    │  LLMProvider       │   │  ToolRuntime    │
                    │  (OpenAI / fake)   │   │  local + MCP    │
                    └────────────────────┘   └─────────────────┘
```

## Install

```bash
pip install -e ".[openai,mcp,dev]"   # from a clone
cp .env.example .env                  # add your OPENAI_API_KEY
```

## Quickstart

```bash
agent-core run "Calculate 17*19 and explain" --config examples/minimal_react/agent.yaml
agent-core replay traces/<run_id>.jsonl
```

Or fully offline, no API key:

```bash
python examples/minimal_react/run_offline.py
```

As a library:

```python
from agent_core import Agent, AgentConfig
from agent_core.llm.openai_provider import OpenAIProvider

config = AgentConfig(model="gpt-4.1-mini", tracing_path=Path("traces"))
async with Agent(config, OpenAIProvider()) as agent:
    result = await agent.run("What is 2+2? Use the calculator.")
print(result.status, result.final_answer)
```

## The loops

Two interchangeable loops, selected via `loop: react | plan_execute` in config.

**ReAct** (default): one run = one `AgentState` (single source of truth). Each step:

1. `memory.load(state)` — messages for the LLM (trimmed / summarized).
2. `llm.complete(messages, tools)` — tools filtered by policy.
3. No tool calls in the reply → **success**, that's the final answer.
4. Otherwise each tool call goes through: policy check → HITL confirmation
   (dangerous tools) → schema validation → timeout-guarded execution.
5. Results are appended as `tool` messages; `memory.commit(...)`.

Stop conditions: final answer, `max_steps`, budgets (tool calls / tokens / $),
unrecoverable LLM error after retries, or `needs_input` when a dangerous tool
awaits human confirmation.

**Plan-Execute**: the model first produces a JSON plan (traced as
`plan.created`), then works through each task with focused ReAct turns
(`plan.task_started` / `plan.task_finished`), and finishes with a final-answer
turn. Steps and budgets are shared with ReAct, so the same guardrails apply.

## Traces

One run = one `traces/{run_id}.jsonl`; one line = one event
(`run.started`, `step.started`, `llm.completed`, `tool.requested`,
`tool.executed`, `policy.denied`, `policy.confirmation`, `memory.compacted`,
`step.finished`, `run.finished`, `error`).

```json
{"schema_version":"1.0","run_id":"r1","ts":"...","type":"tool.executed","step":1,"payload":{"name":"calculator","ok":true,"latency_ms":3,"content_preview":"4"}}
```

`agent-core replay <trace>` pretty-prints a trace. Golden (normalized) traces
in `tests/golden/` pin the loop's behavior as a regression contract.

## Built-in tools

| Tool | Notes |
|---|---|
| `calculator` | AST-based arithmetic, no `eval`, rejects names/calls |
| `get_current_time` | UTC ISO 8601 |
| `http_get` | body capped at 100 KB |
| `read_file` | sandboxed to `workspace/` |
| `write_file` | sandboxed, `dangerous=True` → HITL confirmation |

## MCP in 5 minutes

```yaml
# agent.yaml
model: gpt-4.1-mini
mcp_servers:
  - name: fs
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
tools:
  allow: ["fs__*", "calculator"]
  deny: ["fs__delete*"]
```

```bash
agent-core run "Read hello.txt and summarize it" --config agent.yaml
```

MCP tools register as `{server}__{tool}` and are indistinguishable from local
tools inside the loop. See `examples/mcp_filesystem/`.

## Failure modes (what the model sees vs. what the trace records)

| Failure | `error_type` | Model sees | Trace records |
|---|---|---|---|
| Bad arguments | `validation_error` | schema hint, can retry | full arguments |
| Tool timeout | `timeout` | short timeout notice | latency |
| Handler exception | `execution_error` | exception class only | full repr in `error` event |
| Unknown tool | `not_found` | "does not exist" | — |
| Policy deny | `policy_denied` | "not allowed by policy" | `policy.denied` event |
| HITL rejected | `policy_denied` | "user declined" | `policy.confirmation` |
| MCP server died | `unavailable` | "server unavailable" | — |

Tracebacks never reach the model; runs never crash on a tool failure — the
model gets a short typed error and may adapt or finish.

## Evals

Measure agent behavior instead of guessing. A suite is a YAML file of goals
plus expectations (answer substring/regex, produced files, step ceilings):

```bash
agent-core eval evals/basic.yaml
```

```
task                 result  steps tools    tokens   cost $  time s
-------------------------------------------------------------------
arithmetic           PASS        2     1       484   0.0002     2.7
write_result         PASS        3     2       816   0.0004     3.2
...
passed 5/5   total cost $0.0012
```

Each task runs in its own workspace with its own trace under `eval_runs/`.
Exit code is non-zero if any task fails, so suites slot into CI directly.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pytest -q                          # deterministic, uses ScriptedLLM
UPDATE_GOLDEN=1 pytest tests/test_golden_traces.py   # regenerate golden traces
```

## Roadmap

- v1.1: `agent-core replay` diffing
- v2: vector memory, HTTP API (`needs_input` → `POST /runs/{id}/confirm`), OTel export
