# MCP filesystem example

Connects agent-core to the official MCP filesystem server. Tools appear to the
agent as `fs__read_file`, `fs__list_directory`, etc.

## Prerequisites

- Node.js (for `npx`)
- `pip install "agent-core[openai,mcp]"`
- `OPENAI_API_KEY` in the environment

## Run

```bash
mkdir -p workspace && echo "hello from mcp" > workspace/hello.txt
agent-core run "Read hello.txt and tell me what it says" --config agent.yaml
```

Destructive tools are blocked by the deny list (`fs__delete*`, `fs__move*`).
Inspect the trace afterwards:

```bash
agent-core replay traces/<run_id>.jsonl
```
