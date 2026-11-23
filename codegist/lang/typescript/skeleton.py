"""Build and match abstract TypeScript tree skeletons."""
from __future__ import annotations

from typing import Any, List

from ...auto_mining.skeleton import Skeleton, _Builder
from .fingerprint import _ID_KINDS, _LIT_KINDS, _PARAM_KINDS, op_category
from .shadow import TSNode

_NO_ARITY_TOLERANCE = {"arguments"}

# Keep argument, parameter, and type lists outside wildcard leaf holes.
_LEAF_HOLE_BLOCK = {"arguments", "formal_parameters",
                    "type_arguments", "type_parameters"}


def _grouped_children(node: TSNode):
    order: List[str] = []
    groups: dict = {}
    for c in node.children:
        key = c.field or "_"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    return [(k, groups[k]) for k in order]


def _build(node: TSNode, depth_limit: int, b: _Builder) -> Skeleton:
    if depth_limit <= 0:
        return Skeleton(kind="expr", hole_id=b.alloc())
    if node.kind in _ID_KINDS:
        return Skeleton(kind="id", hole_id=b.alloc())
    if node.kind in _LIT_KINDS:
        return Skeleton(kind="lit", hole_id=b.alloc())
    if node.kind in _PARAM_KINDS:
        return Skeleton(kind="param", hole_id=b.alloc())
    sk = Skeleton(kind="node", node_class=node.kind, fields={})
    if node.op:
        sk.fields["__op__"] = Skeleton(kind="op", hole_id=b.alloc(),
                                       op_category=op_category(node.op))
    for key, children in _grouped_children(node):
        sk.fields[key] = [_build(c, depth_limit - 1, b) for c in children]
    return sk


def build_skeleton_ts(node: TSNode, depth_limit: int = 6) -> Skeleton:
    return _build(node, depth_limit, _Builder())


def skeleton_matches_ts(sk: Skeleton, node: Any, depth_limit: int = 6) -> bool:
    if sk.kind in ("id", "lit", "param", "expr"):
        if isinstance(node, TSNode) and node.kind in _LEAF_HOLE_BLOCK:
            return False
        return True
    if sk.kind == "op":
        if not isinstance(node, str):
            return False
        return op_category(node) == sk.op_category
    if sk.kind != "node":
        return False
    if not isinstance(node, TSNode):
        return False
    if node.kind != sk.node_class:
        return False
    if depth_limit <= 0:
        return True
    groups = dict(_grouped_children(node))
    # An empty body in the skeleton matches only an empty body.
    if not sk.fields and node.children:
        return False
    for key, sub in sk.fields.items():
        if key == "__op__":
            if not skeleton_matches_ts(sub, node.op, depth_limit - 1):
                return False
            continue
        if isinstance(sub, list):
            children = groups.get(key, [])
            if node.kind in _NO_ARITY_TOLERANCE or sk.node_class in _NO_ARITY_TOLERANCE:
                if len(children) != len(sub):
                    return False
            elif bool(sub) != bool(children):
                return False
            for i, child in enumerate(children):
                sub_sk = sub[i] if i < len(sub) else sub[-1]
                if not skeleton_matches_ts(sub_sk, child, depth_limit - 1):
                    return False
        elif isinstance(sub, Skeleton):
            children = groups.get(key, [])
            child = children[0] if children else None
            if child is None:
                return False
            if not skeleton_matches_ts(sub, child, depth_limit - 1):
                return False
        elif sub is None:
            if groups.get(key):
                return False
    return True
