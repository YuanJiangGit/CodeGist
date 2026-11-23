"""Group Python regions by canonical key."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .regions import Region


# operators we want to keep in the key; everything else (names/literals)
# becomes a hole symbol "?id" / "?lit".
_OP_NAMES = {
    "Add", "Sub", "Mult", "Div", "Mod", "Pow", "FloorDiv",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot",
    "In", "NotIn", "And", "Or", "Not", "USub", "UAdd",
}


def _digest(node: ast.AST) -> tuple:
    """Recursive structural digest with identifiers/literals abstracted."""
    if isinstance(node, ast.Name):
        return ("?id",)
    if isinstance(node, ast.Constant):
        return ("?lit", type(node.value).__name__)
    if isinstance(node, ast.arg):
        return ("?param",)
    cls = type(node).__name__
    if cls in _OP_NAMES:
        return (cls,)
    children: list = []
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            children.append(tuple(_digest(v) for v in value if isinstance(v, ast.AST)))
        elif isinstance(value, ast.AST):
            children.append(_digest(value))
    return (cls, tuple(children))


_KNOWN_SUB_PATTERNS = {
    "init_assign_params", "dataclass_fields",
    "one_line_return",
    "if_return_guard", "if_raise_guard",
    "accumulator_for_return", "with_open_body",
    "import_group", "entry_main",
    "len_eq_zero", "len_ne_zero", "is_none", "is_not_none",
    "name_plus_one", "name_minus_one", "name_cmp_zero",
}


def canonical_key(region: Region) -> str:
    if region.sub_pattern in _KNOWN_SUB_PATTERNS:
        return f"{region.region_type}::{region.sub_pattern}"
    digest = _digest(region.node)
    return f"{region.region_type}::{region.sub_pattern}::{hash(digest):x}"


@dataclass
class Bucket:
    key: str
    region_type: str
    sub_pattern: str
    members: List[Region]


def group(regions: Iterable[Region], support_threshold: int) -> List[Bucket]:
    """Bucket regions by canonical key, drop buckets below support_threshold."""
    raw: Dict[str, List[Region]] = defaultdict(list)
    for r in regions:
        raw[canonical_key(r)].append(r)
    out: List[Bucket] = []
    for key, members in raw.items():
        if len(members) < support_threshold:
            continue
        out.append(Bucket(
            key=key,
            region_type=members[0].region_type,
            sub_pattern=members[0].sub_pattern,
            members=members,
        ))
    out.sort(key=lambda b: (b.region_type, b.sub_pattern, -len(b.members)))
    return out
