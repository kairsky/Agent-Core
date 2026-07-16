"""File tools sandboxed to the workspace directory.

read_file is safe; write_file is marked dangerous and goes through HITL confirmation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.tools.base import ToolContext, ToolResult, ToolSpec

MAX_READ_BYTES = 200_000


def resolve_in_workspace(workspace: Path, relative: str) -> Path:
    """Resolve a user-supplied path and refuse anything escaping the workspace."""
    workspace = workspace.resolve()
    candidate = (workspace / relative).resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise PermissionError(f"path escapes workspace: {relative}")
    return candidate


class ReadFileTool:
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file. Paths are relative to the agent workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "relative file path"}},
            "required": ["path"],
        },
    )

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_in_workspace(self._workspace, arguments["path"])
        except PermissionError as exc:
            return ToolResult.failure("permission_denied", str(exc))
        if not path.is_file():
            return ToolResult.failure("not_found", f"No such file: {arguments['path']}")
        text = path.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]
        return ToolResult(ok=True, content=text, data={"path": str(path), "chars": len(text)})


class WriteFileTool:
    spec = ToolSpec(
        name="write_file",
        description="Write UTF-8 text to a file inside the agent workspace (overwrites).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "relative file path"},
                "content": {"type": "string", "description": "text to write"},
            },
            "required": ["path", "content"],
        },
        dangerous=True,
    )

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            path = resolve_in_workspace(self._workspace, arguments["path"])
        except PermissionError as exc:
            return ToolResult.failure("permission_denied", str(exc))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"Wrote {len(arguments['content'])} chars to {arguments['path']}",
            data={"path": str(path)},
        )
