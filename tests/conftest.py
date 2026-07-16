from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.config import AgentConfig


@pytest.fixture
def config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        model="scripted",
        workspace=tmp_path / "workspace",
        tool_timeout_s=0.5,
    )
