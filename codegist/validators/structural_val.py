"""Check that sketches retain the required Python identifiers."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Set

from ..grouping import Bucket
from ..regions import Region
from ..write_infer import WriteTemplate


_BUILTINS_TO_IGNORE = {
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "print", "range", "enumerate", "zip", "map", "filter", "sum", "min",
    "max", "abs", "any", "all", "sorted", "reversed", "open", "type",
    "isinstance", "issubclass", "iter", "next", "repr",
}


def _free_names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and n.id not in _BUILTINS_TO_IGNORE}


def _expected_tokens(r: Region) -> Set[str]:
    sp = r.sub_pattern
    if sp == "init_assign_params":
        return {r.extra["class_name"], *r.extra.get("fields", [])}
    if sp == "dataclass_fields":
        return {r.extra["class_name"], *r.extra.get("fields", [])}
    if sp == "one_line_return":
        fn: ast.FunctionDef = r.node  # type: ignore[assignment]
        ret = fn.body[0]
        names = {fn.name, *[a.arg for a in fn.args.args]}
        if ret.value is not None:  # type: ignore[attr-defined]
            names |= _free_names(ret.value)  # type: ignore[attr-defined]
        return names
    if sp == "if_return_guard":
        if_node: ast.If = r.node  # type: ignore[assignment]
        names = _free_names(if_node.test)
        ret = if_node.body[0]
        if ret.value is not None:  # type: ignore[attr-defined]
            names |= _free_names(ret.value)  # type: ignore[attr-defined]
        return names
    if sp == "if_raise_guard":
        if_node = r.node  # type: ignore[assignment]
        names = _free_names(if_node.test)  # type: ignore[attr-defined]
        raise_stmt = if_node.body[0]  # type: ignore[attr-defined]
        if isinstance(raise_stmt, ast.Raise) and raise_stmt.exc is not None:
            names |= _free_names(raise_stmt.exc)
        return names
    if sp == "accumulator_for_return":
        return {r.extra["acc"]}
    if sp == "with_open_body":
        fn = r.node  # type: ignore[assignment]
        return {fn.name, *[a.arg for a in fn.args.args]}  # type: ignore[attr-defined]
    if sp == "import_group":
        try:
            block = ast.parse(r.source)
        except SyntaxError:
            return set()
        names: Set[str] = set()
        for stmt in block.body:
            if isinstance(stmt, ast.Import):
                for a in stmt.names:
                    names.update(a.name.split("."))
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    names.update(stmt.module.split("."))
                names.update(a.name for a in stmt.names)
        return names
    if sp == "entry_main":
        return set()  # degenerate sketch — nothing semantic to preserve
    if sp in {"len_eq_zero", "len_ne_zero", "is_none", "is_not_none",
              "name_plus_one", "name_minus_one", "name_cmp_zero"}:
        return _free_names(r.node)
    return set()


@dataclass
class StructuralReport:
    instances: int
    failures: List[dict]


def validate(bucket: Bucket, write: WriteTemplate) -> tuple[bool, StructuralReport]:
    failures: List[dict] = []
    for r in bucket.members:
        try:
            rewritten = write.render_default(r)
        except Exception as e:
            failures.append({"file": r.file, "lineno": r.lineno,
                             "missing": [], "error": str(e)})
            continue
        expected = _expected_tokens(r)
        missing = [tok for tok in expected if tok not in rewritten]
        if missing:
            failures.append({"file": r.file, "lineno": r.lineno,
                             "missing": missing})
    ok = len(failures) == 0
    return ok, StructuralReport(instances=len(bucket.members), failures=failures)
