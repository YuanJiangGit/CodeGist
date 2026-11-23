"""Built-in Write transforms for Python regions."""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..write_infer import RenderedTemplate
from .extract import AutoRegion
from .skeleton import Skeleton


def _hole(i: int) -> str:
    return f"\x01CG_HOLE_{i}\x01"


def _param_names(fn: ast.AST) -> str:
    a = fn.args
    parts: list = [arg.arg for arg in a.posonlyargs + a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [arg.arg for arg in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return ",".join(parts)


_OP_TO_SYM = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
    "Pow": "**", "FloorDiv": "//",
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
    "Gt": ">", "GtE": ">=",
}


@dataclass
class WriteTransform:
    name: str
    region_types: tuple
    matches: Callable[[Skeleton, AutoRegion], bool]
    render: Callable[[AutoRegion], RenderedTemplate]


# ---- skeleton predicates ----

def _is_node(sk: Optional[Skeleton], cls: str) -> bool:
    return isinstance(sk, Skeleton) and sk.kind == "node" and sk.node_class == cls


def _is_hole(sk: Optional[Skeleton], kinds: tuple = ("id", "lit", "param", "expr")) -> bool:
    return isinstance(sk, Skeleton) and sk.kind in kinds


def _field(sk: Skeleton, name: str):
    return sk.fields.get(name) if sk.kind == "node" else None


def _body_only(sk: Skeleton, fname: str, want_cls: str) -> Optional[Skeleton]:
    body = _field(sk, fname)
    if isinstance(body, list) and len(body) == 1 and _is_node(body[0], want_cls):
        return body[0]
    if isinstance(body, list) and len(body) == 2 and _is_hole(body[0]) and _is_node(body[1], want_cls):
        # docstring tolerated as a leading expression hole
        return body[1]
    return None


def _skip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:]
    return body


def _has_decorator(node: ast.AST, names: set) -> bool:
    for d in getattr(node, "decorator_list", []) or []:
        if isinstance(d, ast.Name) and d.id in names:
            return True
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in names:
            return True
    return False


# ---- T1: function with single-return body -> arrow form ----

def _m_arrow(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = _skip_docstring(r.node.body)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    return body[0].value is not None


def _r_arrow(r: AutoRegion) -> RenderedTemplate:
    fn = r.node
    body = _skip_docstring(fn.body)
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return RenderedTemplate(skeleton=r.source)
    ret = body[-1]
    params = _param_names(fn)
    h0 = _hole(0)
    return RenderedTemplate(skeleton=f"def {fn.name}({params})=>{h0}",
                            inner_nodes=[ret.value], marks=[h0])


T_ARROW_DEF = WriteTransform("arrow_def", ("function",), _m_arrow, _r_arrow)


# ---- T2: class with __init__ that purely assigns params to fields ----

def _m_class_init(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, ast.ClassDef):
        return False
    cls = r.node
    body = _skip_docstring(cls.body)
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "__init__":
        return False
    init = body[0]
    params = [a.arg for a in init.args.args]
    if not params or params[0] != "self":
        return False
    pset = set(params[1:])
    if not pset:
        return False
    init_body = _skip_docstring(init.body)
    if not init_body:
        return False
    for stmt in init_body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            return False
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"
                and isinstance(stmt.value, ast.Name) and stmt.value.id in pset):
            return False
    return True


def _r_class_init(r: AutoRegion) -> RenderedTemplate:
    cls = r.node
    init = _skip_docstring(cls.body)[0]
    init_body = _skip_docstring(init.body)
    fields = [s.targets[0].attr for s in init_body]
    return RenderedTemplate(skeleton=f"class {cls.name}{{{','.join(fields)}}}")


T_CLASS_INIT = WriteTransform("class_init_sketch", ("class",),
                              _m_class_init, _r_class_init)


# ---- T3: @dataclass class with AnnAssign-only body ----

def _m_dataclass(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, ast.ClassDef):
        return False
    cls = r.node
    if not _has_decorator(cls, {"dataclass"}):
        return False
    body = _skip_docstring(cls.body)
    if not body:
        return False
    return all(isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
               for s in body)


def _r_dataclass(r: AutoRegion) -> RenderedTemplate:
    cls = r.node
    body = _skip_docstring(cls.body)
    fields = [s.target.id for s in body]
    return RenderedTemplate(skeleton=f"class {cls.name}{{{','.join(fields)}}}")


T_DATACLASS = WriteTransform("dataclass_sketch", ("class",),
                             _m_dataclass, _r_dataclass)


# ---- T4 / T5: if-guard family (return / raise + sugar) ----

def _m_if_return(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, ast.If):
        return False
    if_node = r.node
    if if_node.orelse:
        return False
    if len(if_node.body) != 1 or not isinstance(if_node.body[0], ast.Return):
        return False
    return True


def _m_if_raise(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, ast.If):
        return False
    if_node = r.node
    if if_node.orelse:
        return False
    if len(if_node.body) != 1 or not isinstance(if_node.body[0], ast.Raise):
        return False
    return True


def _sugar_cond(cond: ast.AST) -> Optional[str]:
    if (isinstance(cond, ast.Compare) and len(cond.ops) == 1
            and isinstance(cond.comparators[0], ast.Constant)
            and cond.comparators[0].value == 0
            and isinstance(cond.left, ast.Call)
            and isinstance(cond.left.func, ast.Name)
            and cond.left.func.id == "len"
            and len(cond.left.args) == 1):
        v = ast.unparse(cond.left.args[0])
        if isinstance(cond.ops[0], ast.Eq):
            return f"empty({v})"
        if isinstance(cond.ops[0], ast.NotEq):
            return f"!empty({v})"
    if (isinstance(cond, ast.Compare) and len(cond.ops) == 1
            and isinstance(cond.ops[0], (ast.Is, ast.IsNot))
            and isinstance(cond.comparators[0], ast.Constant)
            and cond.comparators[0].value is None):
        v = ast.unparse(cond.left)
        return f"none({v})" if isinstance(cond.ops[0], ast.Is) else f"!none({v})"
    return None


def _r_if_return_inline(r: AutoRegion) -> RenderedTemplate:
    n = r.node
    ret = n.body[0]
    h0, h1 = _hole(0), _hole(1)
    inner = [n.test]
    marks = [h0]
    body_expr = "return"
    if ret.value is not None:
        inner.append(ret.value)
        marks.append(h1)
        body_expr = f"return {h1}"
    return RenderedTemplate(skeleton=f"if {h0}: {body_expr}",
                            inner_nodes=inner, marks=marks)


def _r_if_return_sugar(r: AutoRegion) -> RenderedTemplate:
    n = r.node
    sugar = _sugar_cond(n.test)
    if sugar is None:
        return _r_if_return_inline(r)
    ret = n.body[0]
    if ret.value is None:
        return RenderedTemplate(skeleton=f"if {sugar}: return")
    h0 = _hole(0)
    return RenderedTemplate(skeleton=f"if {sugar}: return {h0}",
                            inner_nodes=[ret.value], marks=[h0])


def _r_if_raise_inline(r: AutoRegion) -> RenderedTemplate:
    n = r.node
    raise_stmt = n.body[0]
    h0, h1 = _hole(0), _hole(1)
    inner = [n.test]
    marks = [h0]
    if raise_stmt.exc is not None:
        inner.append(raise_stmt.exc)
        marks.append(h1)
        return RenderedTemplate(skeleton=f"if {h0}: raise {h1}",
                                inner_nodes=inner, marks=marks)
    return RenderedTemplate(skeleton=f"if {h0}: raise",
                            inner_nodes=inner, marks=marks)


def _r_if_raise_sugar(r: AutoRegion) -> RenderedTemplate:
    n = r.node
    sugar = _sugar_cond(n.test)
    if sugar is None:
        return _r_if_raise_inline(r)
    raise_stmt = n.body[0]
    if raise_stmt.exc is None:
        return RenderedTemplate(skeleton=f"if {sugar}: raise")
    h0 = _hole(0)
    return RenderedTemplate(skeleton=f"if {sugar}: raise {h0}",
                            inner_nodes=[raise_stmt.exc], marks=[h0])


T_IF_RETURN_INLINE = WriteTransform("if_return_inline", ("control_flow",),
                                    _m_if_return, _r_if_return_inline)
T_IF_RETURN_SUGAR = WriteTransform("if_return_sugar", ("control_flow",),
                                   _m_if_return, _r_if_return_sugar)
T_IF_RAISE_INLINE = WriteTransform("if_raise_inline", ("control_flow",),
                                   _m_if_raise, _r_if_raise_inline)
T_IF_RAISE_SUGAR = WriteTransform("if_raise_sugar", ("control_flow",),
                                  _m_if_raise, _r_if_raise_sugar)


# ---- T6: accumulator loop (stmt_range of size 3) ----

def _accumulator_match_local(stmts: list) -> Optional[dict]:
    """Return {'acc','op'} if stmts is the 3-stmt accumulator shape."""
    if len(stmts) != 3:
        return None
    s_init, s_for, s_ret = stmts
    if not (isinstance(s_init, ast.Assign) and len(s_init.targets) == 1
            and isinstance(s_init.targets[0], ast.Name)):
        return None
    acc_name = s_init.targets[0].id
    if not isinstance(s_for, ast.For):
        return None
    update_op = None
    for stmt in ast.walk(s_for):
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == acc_name
                and isinstance(stmt.value, ast.BinOp)
                and isinstance(stmt.value.left, ast.Name)
                and stmt.value.left.id == acc_name):
            update_op = type(stmt.value.op).__name__
            break
        if (isinstance(stmt, ast.AugAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == acc_name):
            update_op = type(stmt.op).__name__
            break
    if update_op is None:
        return None
    if not (isinstance(s_ret, ast.Return)
            and isinstance(s_ret.value, ast.Name)
            and s_ret.value.id == acc_name):
        return None
    return {"acc": acc_name, "op": update_op}


def _m_accumulator(sk: Skeleton | None, r: AutoRegion) -> bool:
    if r.span_kind != "stmt_range":
        return False
    if r.region_type != "data_flow":
        return False
    body = getattr(r.parent_node, r.body_attr, None) or []
    stmts = body[r.first_idx:r.last_idx + 1]
    info = _accumulator_match_local(stmts)
    if info is None:
        return False
    r.extra["acc"] = info["acc"]
    r.extra["op"] = info["op"]
    return True


def _r_accumulator(r: AutoRegion) -> RenderedTemplate:
    acc = r.extra.get("acc")
    op = _OP_TO_SYM.get(r.extra.get("op", ""), r.extra.get("op", ""))
    try:
        block = ast.parse(textwrap.dedent(r.source))
    except SyntaxError:
        return RenderedTemplate(skeleton=r.source)
    if len(block.body) < 3:
        return RenderedTemplate(skeleton=r.source)
    s_init, s_for = block.body[0], block.body[1]
    if not (isinstance(s_init, ast.Assign) and isinstance(s_for, ast.For)):
        return RenderedTemplate(skeleton=r.source)
    init_src = ast.unparse(s_init.value)
    iter_src = ast.unparse(s_for.iter)
    return RenderedTemplate(
        skeleton=f"{acc} = reduce({op}, {iter_src}, init={init_src})")


T_ACCUMULATOR = WriteTransform("accumulator_reduce", ("data_flow",),
                               _m_accumulator, _r_accumulator)


# ---- T7: function whose body is `with open(...) as f: return ...` ----

def _m_with_open(sk: Skeleton | None, r: AutoRegion) -> bool:
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    fn = r.node
    body = _skip_docstring(fn.body)
    if len(body) != 1 or not isinstance(body[0], ast.With):
        return False
    w = body[0]
    if len(w.items) != 1:
        return False
    call = w.items[0].context_expr
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "open"):
        return False
    return len(w.body) == 1


def _r_with_open(r: AutoRegion) -> RenderedTemplate:
    fn = r.node
    params = _param_names(fn)
    body = _skip_docstring(fn.body)
    w: ast.With = body[0]
    call = w.items[0].context_expr
    path = ast.unparse(call.args[0])
    mode = ""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        mode = str(call.args[1].value)
    h0 = w.body[0]
    if isinstance(h0, ast.Return) and h0.value is not None:
        m = _hole(0)
        return RenderedTemplate(
            skeleton=f"def {fn.name}({params})=>open({path},'{mode}').do({m})",
            inner_nodes=[h0.value], marks=[m])
    m = _hole(0)
    return RenderedTemplate(
        skeleton=f"def {fn.name}({params}): open({path},'{mode}').do({m})",
        inner_nodes=[h0], marks=[m])


T_WITH_OPEN = WriteTransform("with_open_arrow", ("function",),
                             _m_with_open, _r_with_open)


# ---- T8: contiguous Import / ImportFrom prefix ----

def _m_import_group(sk: Skeleton | None, r: AutoRegion) -> bool:
    if r.span_kind != "stmt_range":
        return False
    if r.region_type != "module":
        return False
    body = getattr(r.parent_node, r.body_attr, None) or []
    stmts = body[r.first_idx:r.last_idx + 1]
    if len(stmts) < 2:
        return False
    return all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in stmts)


def _r_import_group(r: AutoRegion) -> RenderedTemplate:
    body = getattr(r.parent_node, r.body_attr, None) or []
    stmts = body[r.first_idx:r.last_idx + 1]
    items: list[str] = []
    for s in stmts:
        if isinstance(s, ast.Import):
            items.extend(a.name for a in s.names)
        elif isinstance(s, ast.ImportFrom):
            mod = s.module or ""
            items.extend(f"{mod}.{a.name}" for a in s.names)
    return RenderedTemplate(skeleton="imports{" + ",".join(items) + "}")


T_IMPORT_GROUP = WriteTransform("import_group_sketch", ("module",),
                                _m_import_group, _r_import_group)


# ---- T9: expression-level — Compare/BinOp idioms ----

def _m_expression(sk: Skeleton | None, r: AutoRegion) -> bool:
    if r.region_type != "expression":
        return False
    expr = r.node
    return _classify_expression(expr) is not None


def _classify_expression(expr) -> Optional[str]:
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
    if (isinstance(expr, ast.BinOp) and isinstance(expr.left, ast.Name)
            and isinstance(expr.right, ast.Constant) and expr.right.value == 1):
        if isinstance(expr.op, ast.Add):
            return "name_plus_one"
        if isinstance(expr.op, ast.Sub):
            return "name_minus_one"
    return None


def _r_expression(r: AutoRegion) -> RenderedTemplate:
    expr = r.node
    sub = _classify_expression(expr)
    if sub == "len_eq_zero":
        v = ast.unparse(expr.left.args[0])
        return RenderedTemplate(skeleton=f"empty({v})")
    if sub == "len_ne_zero":
        v = ast.unparse(expr.left.args[0])
        return RenderedTemplate(skeleton=f"!empty({v})")
    if sub == "is_none":
        v = ast.unparse(expr.left)
        return RenderedTemplate(skeleton=f"none({v})")
    if sub == "is_not_none":
        v = ast.unparse(expr.left)
        return RenderedTemplate(skeleton=f"!none({v})")
    if sub == "name_plus_one":
        v = ast.unparse(expr.left)
        return RenderedTemplate(skeleton=f"{v}++")
    if sub == "name_minus_one":
        v = ast.unparse(expr.left)
        return RenderedTemplate(skeleton=f"{v}--")
    if sub == "name_cmp_zero":
        v = ast.unparse(expr.left)
        op = _OP_TO_SYM.get(type(expr.ops[0]).__name__, "?")
        return RenderedTemplate(skeleton=f"{v}{op}0")
    return RenderedTemplate(skeleton=r.source)


T_EXPRESSION = WriteTransform("expression_idiom", ("expression",),
                              _m_expression, _r_expression)


ALL_TRANSFORMS: List[WriteTransform] = [
    T_ARROW_DEF, T_CLASS_INIT, T_DATACLASS,
    T_IF_RETURN_INLINE, T_IF_RETURN_SUGAR,
    T_IF_RAISE_INLINE, T_IF_RAISE_SUGAR,
    T_ACCUMULATOR, T_WITH_OPEN, T_IMPORT_GROUP,
    T_EXPRESSION,
]
