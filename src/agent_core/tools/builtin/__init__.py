from pathlib import Path

from agent_core.tools.base import ToolHandler
from agent_core.tools.builtin.calculator import CalculatorTool
from agent_core.tools.builtin.files import ReadFileTool, WriteFileTool
from agent_core.tools.builtin.http import HttpGetTool
from agent_core.tools.builtin.time_tool import GetCurrentTimeTool


def builtin_tools(workspace: Path) -> list[ToolHandler]:
    return [
        CalculatorTool(),
        GetCurrentTimeTool(),
        HttpGetTool(),
        ReadFileTool(workspace),
        WriteFileTool(workspace),
    ]


__all__ = [
    "CalculatorTool",
    "GetCurrentTimeTool",
    "HttpGetTool",
    "ReadFileTool",
    "WriteFileTool",
    "builtin_tools",
]
