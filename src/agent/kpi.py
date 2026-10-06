"""Safe evaluation of KPI expressions over a frame of entity.attribute columns.

Expressions use entity.attribute names, numbers, arithmetic (+ - * / **),
comparisons, and/or/not, and the functions abs, log, exp, minimum and maximum.
Anything else (other names, calls, subscripts, attribute chains) is rejected.
"""

import ast
import operator
from functools import lru_cache
from typing import Any, Dict, Set

import numpy as np
import pandas as pd

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: np.logical_not}
_COMPARE = {
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_FUNCTIONS = {"abs": np.abs, "log": np.log, "exp": np.exp, "minimum": np.minimum, "maximum": np.maximum}
_AGGREGATIONS = {
    "sum": lambda v: float(np.nansum(v)),
    "mean": lambda v: float(np.nanmean(v)),
    "count": lambda v: float(np.count_nonzero(~pd.isna(v))),
    "min": lambda v: float(np.nanmin(v)),
    "max": lambda v: float(np.nanmax(v)),
}


@lru_cache(maxsize=256)
def _parse(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    _validate(tree.body, expression)
    return tree


def _validate(node: ast.AST, expression: str) -> None:
    allowed = (
        isinstance(node, ast.BinOp) and type(node.op) in _BINARY
        or isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY
        or isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops)
        or isinstance(node, ast.BoolOp)
        or isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
        or _is_attribute_name(node)
        or isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
        and not node.keywords
    )
    if not allowed:
        raise ValueError(f"Unsupported construct {ast.dump(node)[:60]!r} in KPI expression {expression!r}")
    if _is_attribute_name(node):
        return
    for child in (node.args if isinstance(node, ast.Call) else ast.iter_child_nodes(node)):
        if not isinstance(child, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
            _validate(child, expression)


def _is_attribute_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)


def referenced_attributes(expression: str) -> Set[str]:
    """The entity.attribute names an expression uses."""
    return {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(_parse(expression))
        if _is_attribute_name(node)
    }


def _evaluate(node: ast.AST, frame: pd.DataFrame) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if _is_attribute_name(node):
        name = f"{node.value.id}.{node.attr}"
        if name not in frame.columns:
            raise ValueError(f"KPI attribute {name!r} is not available")
        return frame[name].to_numpy()
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, frame), _evaluate(node.right, frame))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate(node.operand, frame))
    if isinstance(node, ast.Compare):
        left, result = _evaluate(node.left, frame), True
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, frame)
            result = np.logical_and(result, _COMPARE[type(op)](left, right))
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        values = [_evaluate(v, frame) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = combine(result, value)
        return result
    if isinstance(node, ast.Call):
        return _FUNCTIONS[node.func.id](*(_evaluate(a, frame) for a in node.args))
    raise ValueError(f"Unsupported construct {ast.dump(node)[:60]!r}")


def evaluate_kpi(kpi: Dict[str, Any], frame: pd.DataFrame) -> float:
    """aggregation(expression) over the rows of the frame."""
    aggregate = _AGGREGATIONS.get(kpi["aggregation"])
    if aggregate is None:
        raise ValueError(f"Unsupported aggregation {kpi['aggregation']!r} of KPI {kpi['kpi_name']!r}")
    values = np.broadcast_to(_evaluate(_parse(kpi["expression"]).body, frame), (len(frame),))
    return aggregate(values.astype(float))
