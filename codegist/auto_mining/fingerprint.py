"""Fingerprint Python regions by their abstract AST shape."""
from __future__ import annotations

import ast
import hashlib
from typing import Any, Tuple


_OP_CATEGORY = {
    # comparison
    "Eq": "?cmp_op", "NotEq": "?cmp_op",
    "Lt": "?cmp_op", "LtE": "?cmp_op", "Gt": "?cmp_op", "GtE": "?cmp_op",
    # identity / membership
    "Is": "?ident_op", "IsNot": "?ident_op",
    "In": "?member_op", "NotIn": "?member_op",
    # arithmetic
    "Add": "?arith_op", "Sub": "?arith_op", "Mult": "?arith_op",
    "Div": "?arith_op", "Mod": "?arith_op", "Pow": "?arith_op",
    "FloorDiv": "?arith_op",
    # bitwise
    "LShift": "?bit_op", "RShift": "?bit_op",
    "BitOr": "?bit_op", "BitXor": "?bit_op", "BitAnd": "?bit_op",
    # logical
    "And": "?logic_op", "Or": "?logic_op",
    "Not": "?logic_unary_op",
    # unary numeric
    "USub": "?unary_arith_op", "UAdd": "?unary_arith_op",
    "Invert": "?bit_unary_op",
}

# Leaves we keep verbatim — only Name-context markers, since these
# carry real semantic info (Load vs Store).
_KEEP_VERBATIM = {"Load", "Store", "Del"}


def _is_docstring(stmt: Any) -> bool:
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _digest(node: Any, depth_limit: int) -> Tuple:
    if depth_limit <= 0:
        return ("?subexpr",)
    if isinstance(node, ast.Name):
        return ("?id",)
    if isinstance(node, ast.Constant):
        return ("?lit",)
    if isinstance(node, ast.arg):
        return ("?param",)
    if not isinstance(node, ast.AST):
        return ("?scalar",)
    cls = type(node).__name__
    if cls in _OP_CATEGORY:
        return (_OP_CATEGORY[cls],)
    if cls in _KEEP_VERBATIM:
        return (cls,)
    fields: list = []
    for fname, value in ast.iter_fields(node):
        if isinstance(value, list):
            items = [v for v in value if isinstance(v, ast.AST)]
            # Ignore leading docstrings when fingerprinting bodies.
            if items and _is_docstring(items[0]):
                items = items[1:]
            child_digests = [_digest(v, depth_limit - 1) for v in items]
            if not child_digests:
                fields.append((fname, "list", 0))
                continue
            first = child_digests[0]
            # Keep call arity in the digest to protect positional and keyword arguments.
            collapse_safe = not (cls == "Call" and fname in ("args", "keywords"))
            if collapse_safe and all(c == first for c in child_digests):
                fields.append((fname, "*", first))
            else:
                fields.append((fname, "tuple", tuple(child_digests)))
        elif isinstance(value, ast.AST):
            fields.append((fname, _digest(value, depth_limit - 1)))
        elif value is None:
            fields.append((fname, None))
        # ignore scalar attributes
    return (cls, tuple(fields))


def fingerprint(node: ast.AST, region_type: str,
                depth_limit: int = 6) -> str:
    """Return a stable string key including the region_type prefix."""
    fp = _digest(node, depth_limit)
    raw = repr((region_type, fp))
    h = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    return f"{region_type}::{h}"


def fingerprint_for_stmt_range(parent: ast.AST, body_attr: str,
                               first_idx: int, last_idx: int,
                               region_type: str,
                               depth_limit: int = 6) -> str:
    body = getattr(parent, body_attr)
    stmts = body[first_idx:last_idx + 1]
    # Strip a leading docstring inside the range too, for the same
    # reason as inside _digest's list-handler.
    if stmts and _is_docstring(stmts[0]):
        stmts = stmts[1:]
    fps = tuple(_digest(s, depth_limit) for s in stmts)
    raw = repr((region_type, "range", fps))
    h = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    return f"{region_type}::{h}"
