'''Built-in calculator tool for safe math expression evaluation.'''

from __future__ import annotations

import ast
import operator
from typing import Any

from ..base import Tool, ToolSpec, ToolParameter, ToolResult

# Allowed operators for safe evaluation
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    '''Safely evaluate an AST node containing only arithmetic operations.'''
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f'Unsupported operator: {op_type.__name__}')
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type == ast.Pow and right > 100:
            raise ValueError('Exponent too large (max 100)')
        return _OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f'Unsupported unary operator: {op_type.__name__}')
        return _OPERATORS[op_type](_safe_eval(node.operand))
    else:
        raise ValueError(f'Unsupported expression: {ast.dump(node)}')


class CalculatorTool(Tool):
    '''Safe math expression evaluator.'''

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name='calculator',
            description='Evaluate a mathematical expression safely. Supports +, -, *, /, //, %, **.',
            parameters=[
                ToolParameter(name='expression', type='string', description='Math expression to evaluate'),
            ],
            returns='string',
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        expression = kwargs.get('expression', '')

        if not expression:
            return ToolResult(
                tool_name='calculator',
                success=False,
                error='Expression is required',
            )

        try:
            tree = ast.parse(expression, mode='eval')
            result = _safe_eval(tree)
            return ToolResult(
                tool_name='calculator',
                success=True,
                output=str(result),
            )
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
            return ToolResult(
                tool_name='calculator',
                success=False,
                error=f'Evaluation error: {e}',
            )
