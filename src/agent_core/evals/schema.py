"""Eval definitions: a suite of goal + expectation pairs, loaded from YAML."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent_core.config import AgentConfig


class FileExpectation(BaseModel):
    path: str  # relative to the task workspace
    text: str | None = None  # required substring; None = existence check only


class Expectation(BaseModel):
    """All specified checks must pass; success status is always required."""

    contains: str | None = None  # substring of the final answer
    regex: str | None = None  # regex over the final answer
    file: FileExpectation | None = None
    max_steps: int | None = None  # efficiency bound, not a budget


class EvalTask(BaseModel):
    id: str
    goal: str
    expect: Expectation = Field(default_factory=Expectation)


class EvalSuite(BaseModel):
    name: str = "suite"
    agent: AgentConfig
    auto_approve: bool = False  # approve dangerous tools without a human
    tasks: list[EvalTask]

    @classmethod
    def from_yaml(cls, path: Path) -> EvalSuite:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
