from __future__ import annotations

import pytest

from agent_core.tools.base import ToolContext
from agent_core.tools.builtin.files import ReadFileTool, WriteFileTool, resolve_in_workspace


def make_ctx(tmp_path) -> ToolContext:
    return ToolContext(run_id="r", step=1, cwd=tmp_path)


def test_resolve_rejects_escape(tmp_path):
    with pytest.raises(PermissionError):
        resolve_in_workspace(tmp_path, "../outside.txt")


def test_resolve_allows_nested(tmp_path):
    path = resolve_in_workspace(tmp_path, "sub/dir/file.txt")
    assert tmp_path.resolve() in path.parents


async def test_write_then_read_roundtrip(tmp_path):
    workspace = tmp_path / "ws"
    write_result = await WriteFileTool(workspace).run(
        {"path": "notes.txt", "content": "hello"}, make_ctx(tmp_path)
    )
    assert write_result.ok
    read_result = await ReadFileTool(workspace).run({"path": "notes.txt"}, make_ctx(tmp_path))
    assert read_result.ok
    assert read_result.content == "hello"


async def test_read_escape_is_permission_denied(tmp_path):
    result = await ReadFileTool(tmp_path / "ws").run({"path": "../secret"}, make_ctx(tmp_path))
    assert not result.ok
    assert result.error_type == "permission_denied"


def test_write_file_is_dangerous(tmp_path):
    assert WriteFileTool(tmp_path).spec.dangerous
