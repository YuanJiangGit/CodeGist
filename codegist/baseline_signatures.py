"""Render signature-only sketches for the baseline."""
from __future__ import annotations

import ast


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef,
               keep_docstrings: bool) -> list[ast.stmt]:
    doc = node.body[0] if node.body and _is_docstring_stmt(node.body[0]) else None
    body: list[ast.stmt] = []
    if keep_docstrings and doc is not None:
        body.append(doc)
    body.append(ast.Expr(ast.Constant(...)))
    return body


def _walk(node: ast.AST, keep_docstrings: bool) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.body = _stub_body(node, keep_docstrings)
        return  # the old body (incl. nested defs) is gone — nothing to recurse
    if not keep_docstrings and isinstance(node, (ast.Module, ast.ClassDef)):
        if node.body and _is_docstring_stmt(node.body[0]):
            node.body = node.body[1:] or [ast.Expr(ast.Constant(...))]
    for child in ast.iter_child_nodes(node):
        _walk(child, keep_docstrings)


def signatures_only(src: str, keep_docstrings: bool = True) -> str:
    """Return the stub-style skeleton of ``src``. Raises SyntaxError as-is."""
    tree = ast.parse(src)
    _walk(tree, keep_docstrings)
    return ast.unparse(tree)
