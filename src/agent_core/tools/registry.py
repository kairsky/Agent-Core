"""Tool registry: the only tool lookup surface the loop knows about."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from agent_core.tools.base import ToolHandler, ToolSpec


class ToolRegistry:
    def __init__(self, handlers: Iterable[ToolHandler] = ()) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        for handler in handlers:
            self.register(handler)

    def register(self, handler: ToolHandler) -> None:
        name = handler.spec.name
        if name in self._handlers:
            raise ValueError(f"tool already registered: {name}")
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        self._handlers.pop(name, None)

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def list_specs(self, allow: Callable[[str], bool] | None = None) -> list[ToolSpec]:
        specs = (handler.spec for handler in self._handlers.values())
        if allow is None:
            return list(specs)
        return [spec for spec in specs if allow(spec.name)]

    def __len__(self) -> int:
        return len(self._handlers)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers
