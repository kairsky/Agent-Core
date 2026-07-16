"""Agent configuration models (Pydantic v2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. Use the available tools when they help you accomplish "
    "the goal. When you have the final answer, reply with plain text and no tool calls."
)


class McpServerConfig(BaseModel):
    """One MCP server the agent connects to on startup."""

    name: str
    transport: Literal["stdio", "sse"] = "stdio"
    command: list[str] | None = None  # for stdio
    url: str | None = None  # for sse
    env: dict[str, str] | None = None


class ToolAccessConfig(BaseModel):
    """Glob-style allow/deny lists applied to tool names (e.g. "fs__*")."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_steps: int = 12
    max_tool_calls_per_step: int = 4
    max_total_tool_calls: int = 40
    max_tokens_budget: int | None = None
    max_cost_usd: float | None = None
    tool_timeout_s: float = 30.0
    parallel_tools: bool = False
    llm_max_retries: int = 2
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    loop: Literal["react", "plan_execute"] = "react"
    memory: Literal["buffer", "summary"] = "buffer"
    max_context_tokens: int | None = None  # trim/compact threshold; None = unlimited
    require_confirmation_for: list[str] = Field(default_factory=list)
    tools: ToolAccessConfig = Field(default_factory=ToolAccessConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    workspace: Path = Path("workspace")
    tracing_path: Path | None = None

    def digest(self) -> str:
        """Stable hash of the config, recorded in traces for reproducibility."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path) -> AgentConfig:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
