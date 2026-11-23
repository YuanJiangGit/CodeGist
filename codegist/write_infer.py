"""Infer and render baseline Write templates."""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List

from .grouping import Bucket
from .regions import Region


# Use control characters that cannot occur in valid source outside literals.
def _hole(i: int) -> str:
    return f"\x01CG_HOLE_{i}\x01"


@dataclass
class RenderedTemplate:
    """A sketch skeleton with parallel lists of hole markers and AST nodes."""
    skeleton: str
    inner_nodes: List[ast.AST] = field(default_factory=list)
    marks: List[str] = field(default_factory=list)

    def materialize(self, texts: List[str]) -> str:
        out = self.skeleton
        for m, t in zip(self.marks, texts):
            out = out.replace(m, t, 1)
        return out

    def materialize_default(self) -> str:
        """Fill holes with unparsed AST nodes for scoring and validation."""
        defaults = [_safe_unparse(n) for n in self.inner_nodes]
        return self.materialize(defaults)


def _safe_unparse(n: ast.AST) -> str:
    try:
        return ast.unparse(n)
    except Exception:
        return "<?>"


@dataclass
class WriteTemplate:
    name: str            # short id, unique within a bucket
    template: str        # printable template (with $holes)
    render: Callable[[Region], RenderedTemplate]

    def render_default(self, r: Region) -> str:
        return self.render(r).materialize_default()


_OP_TO_SYM = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
    "Pow": "**", "FloorDiv": "//",
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
    "Gt": ">", "GtE": ">=",
}


# ---- per-sub_pattern renderers ----

def _render_class_init(r: Region) -> RenderedTemplate:
    fields = r.extra.get("fields", [])
    return RenderedTemplate(
        skeleton=f"class {r.extra['class_name']}{{{','.join(fields)}}}",
    )


def _render_dataclass_fields(r: Region) -> RenderedTemplate:
    fields = r.extra.get("fields", [])
    return RenderedTemplate(
        skeleton=f"class {r.extra['class_name']}{{{','.join(fields)}}}",
    )


def _render_if_raise_inline(r: Region) -> RenderedTemplate:
    if_node: ast.If = r.node  # type: ignore[assignment]
    raise_stmt = if_node.body[0]
    h0, h1 = _hole(0), _hole(1)
    skel = f"if {h0}: raise {h1}"
    # raise_stmt.exc is the exception expression (Call or Name); may be None
    inner_nodes: list = [if_node.test]
    marks = [h0]
    if isinstance(raise_stmt, ast.Raise) and raise_stmt.exc is not None:
        inner_nodes.append(raise_stmt.exc)
        marks.append(h1)
    else:
        skel = f"if {h0}: raise"
    return RenderedTemplate(skeleton=skel, inner_nodes=inner_nodes, marks=marks)


def _render_if_raise_sugar(r: Region) -> RenderedTemplate:
    if_node: ast.If = r.node  # type: ignore[assignment]
    cond = if_node.test
    sugar = None
    if (isinstance(cond, ast.Compare) and len(cond.ops) == 1
            and isinstance(cond.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(cond.comparators[0], ast.Constant)
            and cond.comparators[0].value == 0
            and isinstance(cond.left, ast.Call)
            and isinstance(cond.left.func, ast.Name)
            and cond.left.func.id == "len"
            and len(cond.left.args) == 1):
        v = ast.unparse(cond.left.args[0])
        sugar = f"empty({v})" if isinstance(cond.ops[0], ast.Eq) else f"!empty({v})"
    elif (isinstance(cond, ast.Compare) and len(cond.ops) == 1
          and isinstance(cond.ops[0], (ast.Is, ast.IsNot))
          and isinstance(cond.comparators[0], ast.Constant)
          and cond.comparators[0].value is None):
        v = ast.unparse(cond.left)
        sugar = f"none({v})" if isinstance(cond.ops[0], ast.Is) else f"!none({v})"
    if sugar is None:
        return _render_if_raise_inline(r)
    raise_stmt = if_node.body[0]
    h0 = _hole(0)
    if isinstance(raise_stmt, ast.Raise) and raise_stmt.exc is not None:
        skel = f"if {sugar}: raise {h0}"
        return RenderedTemplate(skeleton=skel,
                                inner_nodes=[raise_stmt.exc], marks=[h0])
    return RenderedTemplate(skeleton=f"if {sugar}: raise")


def _render_one_line_return(r: Region) -> RenderedTemplate:
    fn: ast.FunctionDef = r.node  # type: ignore[assignment]
    params = ",".join(a.arg for a in fn.args.args)
    ret = fn.body[0]
    h0 = _hole(0)
    return RenderedTemplate(
        skeleton=f"def {fn.name}({params})=>{h0}",
        inner_nodes=[ret.value],  # type: ignore[arg-type, list-item]
        marks=[h0],
    )


def _render_if_return_inline(r: Region) -> RenderedTemplate:
    if_node: ast.If = r.node  # type: ignore[assignment]
    ret = if_node.body[0]
    h0, h1 = _hole(0), _hole(1)
    skel = f"if {h0}: return {h1}"
    inner = [if_node.test, ret.value]  # type: ignore[list-item]
    return RenderedTemplate(skeleton=skel, inner_nodes=inner, marks=[h0, h1])


def _render_if_return_sugar(r: Region) -> RenderedTemplate:
    """Shorten simple conditions and leave the return expression open for rewriting."""
    if_node: ast.If = r.node  # type: ignore[assignment]
    cond = if_node.test
    sugar = None
    if (isinstance(cond, ast.Compare) and len(cond.ops) == 1
            and isinstance(cond.ops[0], (ast.Eq, ast.NotEq))
            and isinstance(cond.comparators[0], ast.Constant)
            and cond.comparators[0].value == 0
            and isinstance(cond.left, ast.Call)
            and isinstance(cond.left.func, ast.Name)
            and cond.left.func.id == "len"
            and len(cond.left.args) == 1):
        v = ast.unparse(cond.left.args[0])
        if isinstance(cond.ops[0], ast.Eq):
            sugar = f"empty({v})"
        else:
            sugar = f"!empty({v})"
    elif (isinstance(cond, ast.Compare) and len(cond.ops) == 1
          and isinstance(cond.ops[0], (ast.Is, ast.IsNot))
          and isinstance(cond.comparators[0], ast.Constant)
          and cond.comparators[0].value is None):
        v = ast.unparse(cond.left)
        sugar = f"none({v})" if isinstance(cond.ops[0], ast.Is) else f"!none({v})"
    if sugar is None:
        return _render_if_return_inline(r)
    ret = if_node.body[0]
    h0 = _hole(0)
    return RenderedTemplate(
        skeleton=f"if {sugar}: return {h0}",
        inner_nodes=[ret.value],  # type: ignore[list-item]
        marks=[h0],
    )


def _render_accumulator(r: Region) -> RenderedTemplate:
    acc = r.extra["acc"]
    op = _OP_TO_SYM.get(r.extra["op"], r.extra["op"])
    try:
        block = ast.parse(textwrap.dedent(r.source))
    except SyntaxError:
        return RenderedTemplate(skeleton=r.source)
    if len(block.body) < 3:
        return RenderedTemplate(skeleton=r.source)
    s_init, s_for, _s_ret = block.body[0], block.body[1], block.body[2]
    if not (isinstance(s_init, ast.Assign) and isinstance(s_for, ast.For)):
        return RenderedTemplate(skeleton=r.source)
    init_src = ast.unparse(s_init.value)
    iter_src = ast.unparse(s_for.iter)
    return RenderedTemplate(
        skeleton=f"{acc} = reduce({op}, {iter_src}, init={init_src})",
    )


def _render_with_open(r: Region) -> RenderedTemplate:
    fn: ast.FunctionDef = r.node  # type: ignore[assignment]
    params = ",".join(a.arg for a in fn.args.args)
    w: ast.With = fn.body[0]  # type: ignore[assignment]
    call = w.items[0].context_expr
    path_src = ast.unparse(call.args[0])  # type: ignore[arg-type, index]
    mode = r.extra.get("mode", "")
    body0 = w.body[0]
    h0 = _hole(0)
    if isinstance(body0, ast.Return) and body0.value is not None:
        skel = (f"def {fn.name}({params})=>open({path_src},'{mode}')"
                f".do({h0})")
        return RenderedTemplate(skeleton=skel, inner_nodes=[body0.value],
                                marks=[h0])
    # body is a bare Expr statement — keep the whole stmt as the hole
    skel = (f"def {fn.name}({params}): open({path_src},'{mode}').do({h0})")
    return RenderedTemplate(skeleton=skel, inner_nodes=[body0], marks=[h0])


def _render_import_group(r: Region) -> RenderedTemplate:
    try:
        block = ast.parse(r.source)
    except SyntaxError:
        return RenderedTemplate(skeleton=r.source)
    items: list[str] = []
    for stmt in block.body:
        if isinstance(stmt, ast.Import):
            items.extend(a.name for a in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            items.extend(f"{mod}.{a.name}" for a in stmt.names)
    return RenderedTemplate(skeleton="imports{" + ",".join(items) + "}")


def _render_entry_main(r: Region) -> RenderedTemplate:
    return RenderedTemplate(skeleton="main_entry")


def _render_len_zero(sym: str) -> Callable[[Region], RenderedTemplate]:
    def _r(r: Region) -> RenderedTemplate:
        var = ast.unparse(r.node.left.args[0])  # type: ignore[attr-defined]
        return RenderedTemplate(skeleton=f"{sym}({var})")
    return _r


def _render_is_none(neg: bool) -> Callable[[Region], RenderedTemplate]:
    def _r(r: Region) -> RenderedTemplate:
        var = ast.unparse(r.node.left)  # type: ignore[attr-defined]
        s = f"!none({var})" if neg else f"none({var})"
        return RenderedTemplate(skeleton=s)
    return _r


def _render_name_plus_one(sign: str) -> Callable[[Region], RenderedTemplate]:
    def _r(r: Region) -> RenderedTemplate:
        var = ast.unparse(r.node.left)  # type: ignore[attr-defined]
        s = f"{var}{sign}{sign}" if sign == "+" else f"{var}--"
        return RenderedTemplate(skeleton=s)
    return _r


def _render_name_cmp_zero(r: Region) -> RenderedTemplate:
    cmp = r.node
    var = ast.unparse(cmp.left)  # type: ignore[attr-defined]
    op = _OP_TO_SYM.get(type(cmp.ops[0]).__name__, "?")  # type: ignore[attr-defined]
    return RenderedTemplate(skeleton=f"{var}{op}0")


# ---- dispatcher ----

def synthesize(bucket: Bucket) -> List[WriteTemplate]:
    sp = bucket.sub_pattern
    if sp == "init_assign_params":
        return [WriteTemplate("class_sketch", "class $C{$f1,$f2,...}",
                              _render_class_init)]
    if sp == "dataclass_fields":
        return [WriteTemplate("dataclass_sketch",
                              "class $C{$f1,$f2,...}",
                              _render_dataclass_fields)]
    if sp == "one_line_return":
        return [WriteTemplate("arrow_def", "def $f($params)=>$expr",
                              _render_one_line_return)]
    if sp == "if_return_guard":
        return [
            WriteTemplate("inline_guard", "if $cond: return $ret",
                          _render_if_return_inline),
            WriteTemplate("sugar_guard",
                          "if empty($v)/none($v): return $ret",
                          _render_if_return_sugar),
        ]
    if sp == "if_raise_guard":
        return [
            WriteTemplate("inline_raise_guard", "if $cond: raise $exc",
                          _render_if_raise_inline),
            WriteTemplate("sugar_raise_guard",
                          "if empty($v)/none($v): raise $exc",
                          _render_if_raise_sugar),
        ]
    if sp == "accumulator_for_return":
        return [WriteTemplate("reduce_sketch",
                              "$acc = reduce($op, $iter, init=$init)",
                              _render_accumulator)]
    if sp == "with_open_body":
        return [WriteTemplate("open_arrow",
                              "def $f($params)=>open($p,'$m').do($body)",
                              _render_with_open)]
    if sp == "import_group":
        return [WriteTemplate("imports_sketch", "imports{$m1,$m2,...}",
                              _render_import_group)]
    if sp == "entry_main":
        return [WriteTemplate("main_entry_tag", "main_entry",
                              _render_entry_main)]
    if sp == "len_eq_zero":
        return [WriteTemplate("empty_sugar", "empty($v)",
                              _render_len_zero("empty"))]
    if sp == "len_ne_zero":
        return [WriteTemplate("nonempty_sugar", "!empty($v)",
                              _render_len_zero("!empty"))]
    if sp == "is_none":
        return [WriteTemplate("none_sugar", "none($v)",
                              _render_is_none(neg=False))]
    if sp == "is_not_none":
        return [WriteTemplate("not_none_sugar", "!none($v)",
                              _render_is_none(neg=True))]
    if sp == "name_plus_one":
        return [WriteTemplate("inc_sugar", "$v++", _render_name_plus_one("+"))]
    if sp == "name_minus_one":
        return [WriteTemplate("dec_sugar", "$v--", _render_name_plus_one("-"))]
    if sp == "name_cmp_zero":
        return [WriteTemplate("cmp_zero", "$v <op> 0", _render_name_cmp_zero)]
    return []
