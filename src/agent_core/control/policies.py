"""Tool access policy: glob allow/deny lists + HITL confirmation rules."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from agent_core.config import AgentConfig
from agent_core.tools.base import ToolSpec


class ToolPolicy:
    def __init__(self, config: AgentConfig, specs_by_name: dict[str, ToolSpec] | None = None):
        self._allow_patterns = config.tools.allow
        self._deny_patterns = config.tools.deny
        self._confirm_names = set(config.require_confirmation_for)
        self._specs = specs_by_name or {}

    def allow(self, name: str) -> bool:
        """Deny wins over allow; a tool must match at least one allow pattern."""
        if any(fnmatch(name, pattern) for pattern in self._deny_patterns):
            return False
        return any(fnmatch(name, pattern) for pattern in self._allow_patterns)

    def needs_confirmation(self, name: str, args: dict[str, Any]) -> bool:
        if name in self._confirm_names:
            return True
        spec = self._specs.get(name)
        return spec is not None and spec.dangerous
