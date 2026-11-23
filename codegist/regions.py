"""Extract typed regions using the baseline Python detectors."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


REGION_TYPES = (
    "module",
    "class",
    "function",
    "control_flow",
    "data_flow",
    "api_idiom",
    "expression",
)


@dataclass
class Region:
    region_type: str
    sub_pattern: str
    node: ast.AST
    source: str
    file: str
    lineno: int
    span_kind: str = "node"
    parent_node: ast.AST | None = None
    body_attr: str = ""
    first_idx: int = -1
    last_idx: int = -1
    extra: dict = field(default_factory=dict)


# ---------- helpers ----------

def _is_docstring(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _skip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if body and _is_docstring(body[0]):
        return body[1:]
    return body


def _is_self_attr_assign(stmt: ast.stmt) -> Optional[tuple[str, str]]:
    """Return (attr_name, value_name) if stmt is `self.X = Y` with Y a Name."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
        return None
    tgt = stmt.targets[0]
    if not (isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
        return None
    if not isinstance(stmt.value, ast.Name):
        return None
    return tgt.attr, stmt.value.id


def _has_decorator(node: ast.AST, name: str) -> bool:
    decos = getattr(node, "decorator_list", None) or []
    for d in decos:
        if isinstance(d, ast.Name) and d.id == name:
            return True
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == name:
            return True
        if isinstance(d, ast.Attribute) and d.attr == name:
            return True
    return False


def _source_segment(src: str, node: ast.AST) -> Optional[str]:
    try:
        return ast.get_source_segment(src, node)
    except Exception:
        return None


def _multi_segment(src_lines: list[str], first: ast.stmt, last: ast.stmt) -> str:
    start = first.lineno - 1
    end = (last.end_lineno or last.lineno)
    return "".join(src_lines[start:end])


# ---------- class-level detectors ----------

def _detect_class(node: ast.ClassDef, src: str, file: str) -> Optional[Region]:
    """Detect classes whose initializer assigns parameters to instance fields."""
    cls_body = _skip_docstring(node.body)
    if len(cls_body) != 1:
        return None
    init = cls_body[0]
    if not (isinstance(init, ast.FunctionDef) and init.name == "__init__"):
        return None
    params = [a.arg for a in init.args.args]
    if not params or params[0] != "self":
        return None
    param_names = set(params[1:])
    if not param_names:
        return None
    init_body = _skip_docstring(init.body)
    if not init_body:
        return None
    fields: list[str] = []
    for stmt in init_body:
        pair = _is_self_attr_assign(stmt)
        if pair is None:
            return None
        attr, val_name = pair
        if val_name not in param_names:
            return None
        fields.append(attr)
    seg = _source_segment(src, node)
    if seg is None:
        return None
    return Region("class", "init_assign_params", node, seg, file, node.lineno,
                  extra={"class_name": node.name, "fields": fields})


def _detect_class_dataclass(node: ast.ClassDef, src: str, file: str) -> Optional[Region]:
    """@dataclass\nclass C:\n    name: T1\n    other: T2..."""
    if not _has_decorator(node, "dataclass"):
        return None
    body = _skip_docstring(node.body)
    if not body:
        return None
    fields: list[str] = []
    for stmt in body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(stmt.target.id)
        else:
            return None  # any non-AnnAssign stmt disqualifies this sketch
    seg = _source_segment(src, node)
    if seg is None:
        return None
    return Region("class", "dataclass_fields", node, seg, file, node.lineno,
                  extra={"class_name": node.name, "fields": fields})


# ---------- function-level ----------

def _detect_function(node: ast.FunctionDef, src: str, file: str) -> Optional[Region]:
    """def f(...)[: doc]: return <expr>"""
    body = _skip_docstring(node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    if body[0].value is None:
        return None
    seg = _source_segment(src, node)
    if seg is None:
        return None
    return Region("function", "one_line_return", node, seg, file, node.lineno,
                  extra={"func_name": node.name,
                         "params": [a.arg for a in node.args.args]})


# ---------- control-flow ----------

def _detect_control_flow_guard(
    if_node: ast.If, src: str, file: str
) -> Optional[Region]:
    """if <cond>: return <expr>   (no else, no docstring)"""
    if if_node.orelse:
        return None
    if len(if_node.body) != 1 or not isinstance(if_node.body[0], ast.Return):
        return None
    seg = _source_segment(src, if_node)
    if seg is None:
        return None
    return Region("control_flow", "if_return_guard", if_node, seg, file,
                  if_node.lineno)


def _detect_control_flow_raise_guard(
    if_node: ast.If, src: str, file: str
) -> Optional[Region]:
    """if <cond>: raise <Exc>(...)    (no else)"""
    if if_node.orelse:
        return None
    if len(if_node.body) != 1 or not isinstance(if_node.body[0], ast.Raise):
        return None
    seg = _source_segment(src, if_node)
    if seg is None:
        return None
    return Region("control_flow", "if_raise_guard", if_node, seg, file,
                  if_node.lineno)


# ---------- data-flow ----------

def _detect_data_flow_accumulator(
    fn: ast.FunctionDef, src: str, file: str
) -> Optional[Region]:
    """Detect accumulator loops using plain or augmented assignment."""
    body = _skip_docstring(fn.body)
    if len(body) < 3:
        return None
    raw_body = fn.body  # for stmt_range indices we need raw-body offsets
    body_offset = len(raw_body) - len(body)  # 0 or 1 (docstring offset)
    for i in range(len(body) - 2):
        s_init, s_for, s_ret = body[i], body[i + 1], body[i + 2]
        if not (isinstance(s_init, ast.Assign)
                and len(s_init.targets) == 1
                and isinstance(s_init.targets[0], ast.Name)):
            continue
        acc_name = s_init.targets[0].id
        if not isinstance(s_for, ast.For):
            continue
        update_op: Optional[str] = None
        for stmt in ast.walk(s_for):
            # plain Assign: acc = acc <op> ...
            if (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == acc_name
                    and isinstance(stmt.value, ast.BinOp)
                    and isinstance(stmt.value.left, ast.Name)
                    and stmt.value.left.id == acc_name):
                update_op = type(stmt.value.op).__name__
                break
            # AugAssign: acc <op>= ...
            if (isinstance(stmt, ast.AugAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == acc_name):
                update_op = type(stmt.op).__name__
                break
        if update_op is None:
            continue
        if not (isinstance(s_ret, ast.Return)
                and isinstance(s_ret.value, ast.Name)
                and s_ret.value.id == acc_name):
            continue
        src_lines = src.splitlines(keepends=True)
        seg = _multi_segment(src_lines, s_init, s_ret)
        return Region("data_flow", "accumulator_for_return",
                      fn, seg, file, s_init.lineno,
                      span_kind="stmt_range",
                      parent_node=fn, body_attr="body",
                      first_idx=body_offset + i,
                      last_idx=body_offset + i + 2,
                      extra={"acc": acc_name, "op": update_op})
    return None


# ---------- api idiom ----------

def _detect_api_idiom_with_open(
    fn: ast.FunctionDef, src: str, file: str
) -> Optional[Region]:
    """def f(...)[: doc]: with open(<expr>, <mode>) as f: return <expr>"""
    body = _skip_docstring(fn.body)
    if len(body) != 1 or not isinstance(body[0], ast.With):
        return None
    w = body[0]
    if len(w.items) != 1:
        return None
    call = w.items[0].context_expr
    if not (isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name) and call.func.id == "open"):
        return None
    if len(w.body) != 1:
        return None
    body0 = w.body[0]
    if not (isinstance(body0, ast.Return) or isinstance(body0, ast.Expr)):
        return None
    seg = _source_segment(src, fn)
    if seg is None:
        return None
    mode = ""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        mode = str(call.args[1].value)
    return Region("api_idiom", "with_open_body", fn, seg, file, fn.lineno,
                  extra={"mode": mode})


# ---------- expression-level ----------

def _classify_expression(expr: ast.AST) -> Optional[str]:
    """Return the sub_pattern key for a stylised Compare/BinOp, else None."""
    if (isinstance(expr, ast.Compare) and len(expr.ops) == 1
            and len(expr.comparators) == 1):
        left, op, right = expr.left, expr.ops[0], expr.comparators[0]
        if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "len" and len(left.args) == 1
                and isinstance(left.args[0], ast.Name)
                and isinstance(right, ast.Constant) and right.value == 0):
            if isinstance(op, ast.Eq):
                return "len_eq_zero"
            if isinstance(op, ast.NotEq):
                return "len_ne_zero"
        if (isinstance(left, ast.Name) and isinstance(right, ast.Constant)
                and right.value == 0
                and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                    ast.Gt, ast.GtE))):
            return "name_cmp_zero"
        if (isinstance(left, ast.Name)
                and isinstance(op, (ast.Is, ast.IsNot))
                and isinstance(right, ast.Constant) and right.value is None):
            return "is_none" if isinstance(op, ast.Is) else "is_not_none"
    elif (isinstance(expr, ast.BinOp) and isinstance(expr.left, ast.Name)
          and isinstance(expr.right, ast.Constant) and expr.right.value == 1):
        if isinstance(expr.op, ast.Add):
            return "name_plus_one"
        if isinstance(expr.op, ast.Sub):
            return "name_minus_one"
    return None


def _detect_expression(expr: ast.AST, src: str, file: str) -> Optional[Region]:
    sub = _classify_expression(expr)
    if sub is None:
        return None
    seg = _source_segment(src, expr)
    if seg is None:
        return None
    line = getattr(expr, "lineno", 0) or 0
    return Region("expression", sub, expr, seg, file, line)


# ---------- module-level ----------

def _detect_module_imports(
    tree: ast.Module, src: str, file: str
) -> Optional[Region]:
    start = 0
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        start = 1
    imports: list[ast.stmt] = []
    idx = start
    while idx < len(tree.body) and isinstance(
            tree.body[idx], (ast.Import, ast.ImportFrom)):
        imports.append(tree.body[idx])
        idx += 1
    if len(imports) < 2:
        return None
    src_lines = src.splitlines(keepends=True)
    seg = _multi_segment(src_lines, imports[0], imports[-1])
    return Region("module", "import_group", tree, seg, file, imports[0].lineno,
                  span_kind="stmt_range",
                  parent_node=tree, body_attr="body",
                  first_idx=start, last_idx=idx - 1,
                  extra={"count": len(imports)})


def _detect_module_entry(
    tree: ast.Module, src: str, file: str
) -> Optional[Region]:
    for stmt in tree.body:
        if not isinstance(stmt, ast.If):
            continue
        t = stmt.test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1
                and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "__main__"):
            continue
        if len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Expr):
            continue
        seg = _source_segment(src, stmt)
        if seg is None:
            continue
        return Region("module", "entry_main", stmt, seg, file, stmt.lineno)
    return None


# ---------- driver ----------

def extract_from_tree(tree: ast.Module, src: str, file: str) -> List[Region]:
    regions: List[Region] = []

    mi = _detect_module_imports(tree, src, file)
    if mi:
        regions.append(mi)
    me = _detect_module_entry(tree, src, file)
    if me:
        regions.append(me)

    # First pass: class / function / control-flow / api-idiom / data-flow
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            r = _detect_class_dataclass(node, src, file)
            if r:
                regions.append(r)
                continue  # don't also try the manual-init detector
            r = _detect_class(node, src, file)
            if r:
                regions.append(r)
        elif isinstance(node, ast.FunctionDef):
            r = _detect_function(node, src, file)
            if r:
                regions.append(r)
            r = _detect_data_flow_accumulator(node, src, file)
            if r:
                regions.append(r)
            r = _detect_api_idiom_with_open(node, src, file)
            if r:
                regions.append(r)
        elif isinstance(node, ast.If):
            r = _detect_control_flow_guard(node, src, file)
            if r:
                regions.append(r)
            r = _detect_control_flow_raise_guard(node, src, file)
            if r:
                regions.append(r)

    # Second pass: stylised expressions, anywhere in the AST.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Compare, ast.BinOp)):
            r = _detect_expression(node, src, file)
            if r:
                regions.append(r)
    return regions


def extract_from_source(src: str, file: str) -> List[Region]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return extract_from_tree(tree, src, file)


def iter_corpus(corpus_dir: Path, transform=None,
                exclude: list[str] | None = None) -> Iterator[Region]:
    """Yield regions from Python files, with optional filtering and preprocessing."""
    from ._excludes import is_excluded
    excludes = list(exclude) if exclude else []
    for path in sorted(corpus_dir.rglob("*.py")):
        if excludes and is_excluded(path, excludes):
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if transform is not None:
            try:
                src = transform(src)
            except Exception:
                pass
        for r in extract_from_source(src, str(path)):
            yield r


def count_corpus(corpus_dir: Path,
                 exclude: list[str] | None = None) -> tuple[int, int]:
    """Return (kept, skipped) ``.py`` file counts after applying excludes."""
    from ._excludes import is_excluded
    excludes = list(exclude) if exclude else []
    kept = skipped = 0
    for path in corpus_dir.rglob("*.py"):
        if excludes and is_excluded(path, excludes):
            skipped += 1
        else:
            kept += 1
    return kept, skipped
