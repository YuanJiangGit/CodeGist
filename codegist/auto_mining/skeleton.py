"""Build and match abstract Python AST skeletons."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .fingerprint import _KEEP_VERBATIM, _OP_CATEGORY, _is_docstring


@dataclass
class Skeleton:
    kind: str                 # "node" | "id" | "lit" | "param" | "expr" | "op"
    hole_id: int = -1
    # for kind == "node":
    node_class: str = ""
    fields: dict = field(default_factory=dict)
    # for kind == "op":
    op_category: str = ""


@dataclass
class _Builder:
    next_hole_id: int = 0

    def alloc(self) -> int:
        i = self.next_hole_id
        self.next_hole_id += 1
        return i


def _build(node: Any, depth_limit: int, b: _Builder) -> Skeleton:
    if depth_limit <= 0:
        return Skeleton(kind="expr", hole_id=b.alloc())
    if isinstance(node, ast.Name):
        return Skeleton(kind="id", hole_id=b.alloc())
    if isinstance(node, ast.Constant):
        # No lit_type — literal-type variation is collapsed in the
        # fingerprint, so the skeleton must follow suit for consistency.
        return Skeleton(kind="lit", hole_id=b.alloc())
    if isinstance(node, ast.arg):
        return Skeleton(kind="param", hole_id=b.alloc())
    if not isinstance(node, ast.AST):
        return Skeleton(kind="expr", hole_id=b.alloc())
    cls = type(node).__name__
    if cls in _OP_CATEGORY:
        return Skeleton(kind="op", hole_id=b.alloc(),
                        op_category=_OP_CATEGORY[cls])
    if cls in _KEEP_VERBATIM:
        return Skeleton(kind="node", node_class=cls, fields={})
    sk = Skeleton(kind="node", node_class=cls, fields={})
    for fname, value in ast.iter_fields(node):
        if isinstance(value, list):
            children = [v for v in value if isinstance(v, ast.AST)]
            sk.fields[fname] = [_build(c, depth_limit - 1, b) for c in children]
        elif isinstance(value, ast.AST):
            sk.fields[fname] = _build(value, depth_limit - 1, b)
        elif value is None:
            sk.fields[fname] = None
        # scalar attrs ignored
    return sk


def build_skeleton(node: ast.AST, depth_limit: int = 6) -> Skeleton:
    """Build a Skeleton from a single representative instance."""
    return _build(node, depth_limit, _Builder())


# ---- hole extraction ----

@dataclass
class HoleValue:
    hole_id: int
    kind: str
    value: Optional[ast.AST]   # the concrete sub-node in *this* instance


def extract_holes(skeleton: Skeleton, node: ast.AST,
                  depth_limit: int = 6) -> List[HoleValue]:
    """Walk skeleton + instance in lock-step, collecting hole sub-nodes."""
    out: List[HoleValue] = []
    _extract(skeleton, node, depth_limit, out)
    return out


def _extract(sk: Skeleton, node: Any, depth_limit: int,
             out: List[HoleValue]) -> None:
    if sk.kind in ("id", "lit", "param", "expr", "op"):
        out.append(HoleValue(sk.hole_id, sk.kind,
                             node if isinstance(node, ast.AST) else None))
        return
    if sk.kind != "node":
        return
    if depth_limit <= 0 or not isinstance(node, ast.AST):
        return
    for fname, sub in sk.fields.items():
        if isinstance(sub, list):
            value = getattr(node, fname, None) or []
            children = [v for v in value if isinstance(v, ast.AST)]
            # Match extra list entries against the final skeleton item.
            for i, child in enumerate(children):
                sub_sk = sub[i] if i < len(sub) else sub[-1]
                _extract(sub_sk, child, depth_limit - 1, out)
        elif isinstance(sub, Skeleton):
            child = getattr(node, fname, None)
            _extract(sub, child, depth_limit - 1, out)
        # None field → nothing to do


# ---- structural match predicate (the online Match component) ----

def _skeleton_is_docstring(sk: Any) -> bool:
    if not isinstance(sk, Skeleton) or sk.kind != "node" or sk.node_class != "Expr":
        return False
    val = sk.fields.get("value")
    return isinstance(val, Skeleton) and val.kind == "lit"


def skeleton_matches(sk: Skeleton, node: Any, depth_limit: int = 6) -> bool:
    """Match fixed AST structure and require equal call argument and keyword counts."""
    if sk.kind in ("id", "lit", "param", "expr"):
        # Pure wildcard: any sub-tree (or even a non-AST scalar) is fine.
        return True
    if sk.kind == "op":
        # Operator holes still require a matching category.
        if not isinstance(node, ast.AST):
            return False
        cls = type(node).__name__
        return _OP_CATEGORY.get(cls) == sk.op_category
    if sk.kind != "node":
        return False
    # Fixed node: class must agree.
    if not isinstance(node, ast.AST):
        return False
    if type(node).__name__ != sk.node_class:
        return False
    if depth_limit <= 0:
        # Past the structural horizon: treat as matched (wildcard tail).
        return True
    for fname, sub in sk.fields.items():
        if isinstance(sub, list):
            value = getattr(node, fname, None) or []
            children = [v for v in value if isinstance(v, ast.AST)]
            # Ignore leading docstrings on both sides to match fingerprinting.
            if sub and _skeleton_is_docstring(sub[0]):
                sub = sub[1:]
            if children and _is_docstring(children[0]):
                children = children[1:]
            # Exact call arity prevents a rewrite from dropping extra arguments.
            if sk.node_class == "Call" and fname in ("args", "keywords"):
                if len(children) != len(sub):
                    return False
            elif bool(sub) != bool(children):
                # One side empty, the other not → incompatible shape (and
                # guards the sub[-1] index below against an empty skeleton).
                return False
            for i, child in enumerate(children):
                sub_sk = sub[i] if i < len(sub) else sub[-1]
                if not skeleton_matches(sub_sk, child, depth_limit - 1):
                    return False
        elif isinstance(sub, Skeleton):
            child = getattr(node, fname, None)
            if not skeleton_matches(sub, child, depth_limit - 1):
                return False
        elif sub is None:
            # Require absent fields to stay absent, such as a bare return's value.
            if getattr(node, fname, None) is not None:
                return False
    return True


# ---- (de)serialization for the schema set ----

def skeleton_to_dict(sk: Skeleton) -> dict:
    def _enc(v: Any):
        if isinstance(v, Skeleton):
            return skeleton_to_dict(v)
        if isinstance(v, list):
            return [_enc(x) for x in v]
        return None  # the only other case is an absent (None) field

    out: dict = {"kind": sk.kind}
    if sk.hole_id != -1:
        out["hole_id"] = sk.hole_id
    if sk.kind == "node":
        out["node_class"] = sk.node_class
        out["fields"] = {k: _enc(v) for k, v in sk.fields.items()}
    elif sk.kind == "op":
        out["op_category"] = sk.op_category
    return out


def skeleton_from_dict(d: dict) -> Skeleton:
    """Inverse of :func:`skeleton_to_dict`."""
    def _dec(v: Any):
        if isinstance(v, dict):
            return skeleton_from_dict(v)
        if isinstance(v, list):
            return [_dec(x) for x in v]
        return None

    sk = Skeleton(kind=d["kind"], hole_id=d.get("hole_id", -1))
    if sk.kind == "node":
        sk.node_class = d.get("node_class", "")
        sk.fields = {k: _dec(v) for k, v in (d.get("fields") or {}).items()}
    elif sk.kind == "op":
        sk.op_category = d.get("op_category", "")
    return sk
