"""Safe arithmetic evaluator built on the AST — no eval(), no names, no calls."""

from __future__ import annotations

import ast
import operator
from typing import Any

from agent_core.tools.base import ToolContext, ToolResult, ToolSpec

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_POWER = 10_000


def _evaluate(node: ast.expr) -> float:
    match node:
        case ast.Constant(value=int() | float() as value) if not isinstance(value, bool):
            return value
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPS:
            lhs, rhs = _evaluate(left), _evaluate(right)
            if isinstance(op, ast.Pow) and abs(rhs) > _MAX_POWER:
                raise ValueError("exponent too large")
            return _BINARY_OPS[type(op)](lhs, rhs)
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            return _UNARY_OPS[type(op)](_evaluate(operand))
        case _:
            raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def evaluate_expression(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _evaluate(tree.body)


class CalculatorTool:
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression (+, -, *, /, //, %, **, parentheses).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '17 * 19' or '(2+3)**2'"}
            },
            "required": ["expression"],
        },
    )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            value = evaluate_expression(arguments["expression"])
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            return ToolResult.failure("execution_error", f"Cannot evaluate expression: {exc}")
        text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return ToolResult(ok=True, content=text, data={"value": value})
