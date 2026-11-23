"""Infer Match predicates for baseline regions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .grouping import Bucket


@dataclass
class Match:
    region_type: str
    sub_pattern: str
    holes: List[Dict[str, Any]]  # name + kind ("identifier" | "literal" | "expr")
    notes: str


def infer_match(bucket: Bucket) -> Match:
    sp = bucket.sub_pattern
    rt = bucket.region_type
    if sp == "init_assign_params":
        return Match(rt, sp,
                     holes=[{"name": "class_name", "kind": "identifier"},
                            {"name": "fields", "kind": "identifier_list"}],
                     notes="class C[:doc]: def __init__(self, p1..pk)[:doc]: "
                           "self.attri = pi  (docstring + private prefix allowed)")
    if sp == "dataclass_fields":
        return Match(rt, sp,
                     holes=[{"name": "class_name", "kind": "identifier"},
                            {"name": "fields", "kind": "identifier_list"}],
                     notes="@dataclass\\nclass C: name: T1; ... (AnnAssign body)")
    if sp == "one_line_return":
        return Match(rt, sp,
                     holes=[{"name": "func_name", "kind": "identifier"},
                            {"name": "params", "kind": "identifier_list"},
                            {"name": "ret_expr", "kind": "expr"}],
                     notes="def f(...): return <expr>")
    if sp == "if_return_guard":
        return Match(rt, sp,
                     holes=[{"name": "cond", "kind": "expr"},
                            {"name": "ret_expr", "kind": "expr"}],
                     notes="if <cond>: return <expr>  (no else)")
    if sp == "if_raise_guard":
        return Match(rt, sp,
                     holes=[{"name": "cond", "kind": "expr"},
                            {"name": "raise_expr", "kind": "expr"}],
                     notes="if <cond>: raise <Exception>(...)  (no else)")
    if sp == "accumulator_for_return":
        return Match(rt, sp,
                     holes=[{"name": "acc", "kind": "identifier"},
                            {"name": "init", "kind": "expr"},
                            {"name": "iter", "kind": "expr"},
                            {"name": "op", "kind": "operator"}],
                     notes="acc = <init>; for x in xs: acc = acc <op> ...; "
                           "return acc")
    if sp == "with_open_body":
        return Match(rt, sp,
                     holes=[{"name": "path_expr", "kind": "expr"},
                            {"name": "mode", "kind": "literal"},
                            {"name": "body", "kind": "expr"}],
                     notes="def f(...): with open(<path>, <mode>) as f: "
                           "<return|expr>")
    if sp == "import_group":
        return Match(rt, sp,
                     holes=[{"name": "imports", "kind": "stmt_list"}],
                     notes="contiguous top-level Import/ImportFrom prefix")
    if sp == "entry_main":
        return Match(rt, sp, holes=[], notes='if __name__ == "__main__": ...')
    # expression-level sub_patterns
    if sp in {"len_eq_zero", "len_ne_zero"}:
        return Match(rt, sp,
                     holes=[{"name": "var", "kind": "identifier"}],
                     notes="len(<id>) == 0 / != 0")
    if sp in {"is_none", "is_not_none"}:
        return Match(rt, sp,
                     holes=[{"name": "var", "kind": "identifier"}],
                     notes="<id> is None / is not None")
    if sp in {"name_plus_one", "name_minus_one"}:
        return Match(rt, sp,
                     holes=[{"name": "var", "kind": "identifier"}],
                     notes="<id> + 1 / - 1")
    if sp == "name_cmp_zero":
        return Match(rt, sp,
                     holes=[{"name": "var", "kind": "identifier"},
                            {"name": "op", "kind": "operator"}],
                     notes="<id> <op> 0")
    return Match(rt, sp, holes=[], notes="unspecified")
