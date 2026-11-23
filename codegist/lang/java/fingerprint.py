"""Fingerprint Java regions by their abstract tree shape."""
from __future__ import annotations

import hashlib
from typing import Tuple

from .shadow import JNode

_ID_KINDS = {"identifier", "type_identifier"}
_LIT_KINDS = {
    "decimal_integer_literal", "hex_integer_literal", "octal_integer_literal",
    "binary_integer_literal", "decimal_floating_point_literal",
    "hex_floating_point_literal", "string_literal", "character_literal",
    "true", "false", "null_literal",
}
_PARAM_KINDS = {"formal_parameter", "spread_parameter"}

OP_CATEGORY_JAVA = {
    "==": "?cmp_op", "!=": "?cmp_op", "<": "?cmp_op", "<=": "?cmp_op",
    ">": "?cmp_op", ">=": "?cmp_op",
    "+": "?arith_op", "-": "?arith_op", "*": "?arith_op", "/": "?arith_op",
    "%": "?arith_op",
    "&&": "?logic_op", "||": "?logic_op", "!": "?logic_unary_op",
    "&": "?bit_op", "|": "?bit_op", "^": "?bit_op",
    "<<": "?bit_op", ">>": "?bit_op", ">>>": "?bit_op",
    "++": "?update_op", "--": "?update_op",
}

# Arity is load-bearing here — a single-arg template must not fire on a
# multi-arg call (the f(a, b) -> f(a) silent-drop failure).
_NO_COLLAPSE_KINDS = {"argument_list"}


def op_category(op: str) -> str:
    return OP_CATEGORY_JAVA.get(op, "?op")


def _digest(node: JNode, depth_limit: int) -> Tuple:
    if depth_limit <= 0:
        return ("?subexpr",)
    if node.kind in _ID_KINDS:
        return ("?id",)
    if node.kind in _LIT_KINDS:
        return ("?lit",)
    if node.kind in _PARAM_KINDS:
        return ("?param",)
    head = (node.kind,)
    if node.op:
        head = (node.kind, op_category(node.op))
    child_digests = [
        (c.field or "_", _digest(c, depth_limit - 1)) for c in node.children
    ]
    if not child_digests:
        return head + ((),)
    collapse_safe = node.kind not in _NO_COLLAPSE_KINDS
    if collapse_safe and len(child_digests) > 1:
        first = child_digests[0]
        if all(c == first for c in child_digests):
            return head + ((("*",) + first),)
    return head + (tuple(child_digests),)


def fingerprint_java(node: JNode, region_type: str,
                     depth_limit: int = 6) -> str:
    fp = _digest(node, depth_limit)
    raw = repr((region_type, fp))
    h = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    return f"{region_type}::{h}"


def fingerprint_java_range(stmts: list, region_type: str,
                           depth_limit: int = 6) -> str:
    fps = tuple(_digest(s, depth_limit) for s in stmts)
    raw = repr((region_type, "range", fps))
    h = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    return f"{region_type}::{h}"
