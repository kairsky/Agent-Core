from __future__ import annotations

import pytest

from agent_core.tools.base import ToolContext
from agent_core.tools.builtin.calculator import CalculatorTool, evaluate_expression


def make_ctx(tmp_path) -> ToolContext:
    return ToolContext(run_id="r", step=1, cwd=tmp_path)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", 4),
        ("17 * 19", 323),
        ("(2 + 3) ** 2", 25),
        ("-4 / 2", -2),
        ("10 // 3", 3),
        ("10 % 3", 1),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: float):
    assert evaluate_expression(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd')",
        "exec('1')",
        "[1,2][0]",
        "'a' * 100",
        "x + 1",
        "2 ** 999999",
    ],
)
def test_rejects_non_arithmetic(expression: str):
    with pytest.raises((ValueError, SyntaxError)):
        evaluate_expression(expression)


async def test_tool_returns_error_result_not_exception(tmp_path):
    result = await CalculatorTool().run({"expression": "__import__('os')"}, make_ctx(tmp_path))
    assert not result.ok
    assert result.error_type == "execution_error"


async def test_tool_formats_integers(tmp_path):
    result = await CalculatorTool().run({"expression": "8 / 2"}, make_ctx(tmp_path))
    assert result.ok
    assert result.content == "4"
