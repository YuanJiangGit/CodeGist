"""Synthesize and check Write templates with an LLM."""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..llm_client import LLMClient
from .extract import AutoRegion

log = logging.getLogger("codegist.auto_mining.llm_synth")


# ---- AST helpers ----

def _skip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def _call_target_name(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _short_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _is_named_decorator(d: ast.expr, name: str) -> bool:
    if isinstance(d, ast.Name) and d.id == name:
        return True
    if isinstance(d, ast.Call) and _call_target_name(d) == name:
        return True
    if isinstance(d, ast.Attribute) and d.attr == name:
        return True
    return False


def _summarize_decorators(decos: List[ast.expr]) -> str:
    if not decos:
        return ""
    names = []
    for d in decos:
        if isinstance(d, ast.Name):
            names.append("@" + d.id)
        elif isinstance(d, ast.Call):
            n = _call_target_name(d)
            if n:
                names.append("@" + n)
        elif isinstance(d, ast.Attribute):
            names.append("@" + d.attr)
    return " ".join(names) if names else ""


def _compact_annotation(node: Optional[ast.AST]) -> str:
    """Shorten common type annotations using standard Python syntax."""
    if node is None:
        return ""
    try:
        if isinstance(node, ast.Subscript):
            base = _short_name(node.value)
            sl = node.slice
            if base == "Optional":
                return _compact_annotation(sl) + "|None"
            if base in ("List", "MutableSequence"):
                return "list[" + _compact_annotation(sl) + "]"
            if base == "Set":
                return "set[" + _compact_annotation(sl) + "]"
            if base == "FrozenSet":
                return "frozenset[" + _compact_annotation(sl) + "]"
            if base in ("Sequence", "Iterable"):
                return base + "[" + _compact_annotation(sl) + "]"
            if base in ("Dict", "MutableMapping"):
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    return ("dict[" + _compact_annotation(sl.elts[0])
                            + "," + _compact_annotation(sl.elts[1]) + "]")
            if base == "Mapping":
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    return ("Mapping[" + _compact_annotation(sl.elts[0])
                            + "," + _compact_annotation(sl.elts[1]) + "]")
            if base == "Tuple":
                if isinstance(sl, ast.Tuple):
                    return ("tuple[" + ",".join(_compact_annotation(e)
                                                for e in sl.elts) + "]")
                return "tuple[" + _compact_annotation(sl) + "]"
            if base == "Union":
                if isinstance(sl, ast.Tuple):
                    return "|".join(_compact_annotation(e) for e in sl.elts)
                return _compact_annotation(sl)
            if base == "Callable":
                return "Callable"
            inner = (_compact_annotation(sl)
                     if not isinstance(sl, ast.Tuple)
                     else ",".join(_compact_annotation(e) for e in sl.elts))
            return base + "[" + inner + "]"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return _compact_annotation(node.left) + "|" + _compact_annotation(node.right)
        return ast.unparse(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return ""


def _compact_default(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return "None"
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            if len(v) > 12:
                return '"..."'
            return repr(v)
    if isinstance(node, ast.Name):
        return node.id
    return "..."


def _func_signature(fn: ast.AST, include_name: bool) -> str:
    a = fn.args
    parts: list = []
    posonly = list(a.posonlyargs)
    regular = list(a.args)
    n_pos = len(posonly) + len(regular)
    pos_defaults = list(a.defaults)
    pad = n_pos - len(pos_defaults)
    for i, arg in enumerate(posonly + regular):
        s = arg.arg
        if arg.annotation is not None:
            s += ":" + _compact_annotation(arg.annotation)
        di = i - pad
        if di >= 0 and di < len(pos_defaults):
            try:
                s += "=" + _compact_default(pos_defaults[di])
            except Exception:
                pass
        parts.append(s)
        if posonly and i == len(posonly) - 1:
            parts.append("/")
    if a.vararg:
        s = "*" + a.vararg.arg
        if a.vararg.annotation is not None:
            s += ":" + _compact_annotation(a.vararg.annotation)
        parts.append(s)
    elif a.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(a.kwonlyargs):
        s = arg.arg
        if arg.annotation is not None:
            s += ":" + _compact_annotation(arg.annotation)
        if i < len(a.kw_defaults) and a.kw_defaults[i] is not None:
            try:
                s += "=" + _compact_default(a.kw_defaults[i])
            except Exception:
                pass
        parts.append(s)
    if a.kwarg:
        s = "**" + a.kwarg.arg
        if a.kwarg.annotation is not None:
            s += ":" + _compact_annotation(a.kwarg.annotation)
        parts.append(s)
    sig = "(" + ",".join(parts) + ")"
    ret = getattr(fn, "returns", None)
    if ret is not None:
        sig += "->" + _compact_annotation(ret)
    if include_name:
        sig = fn.name + sig
    return sig


def _short_expr(node: ast.AST, limit: int = 72) -> str:
    try:
        s = ast.unparse(node)
    except Exception:
        return "?"
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _ordered_calls(node: ast.AST) -> List[str]:
    """List call targets in source order, including repeats."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    calls.sort(key=lambda c: (getattr(c, "lineno", 0), getattr(c, "col_offset", 0)))
    out: List[str] = []
    for c in calls:
        nm = _call_target_name(c)
        if nm and len(nm) < 40:
            out.append(nm)
    return out


def _full_expr(node: ast.AST) -> str:
    """Render an expression with normalized whitespace and no length cap."""
    try:
        return " ".join(ast.unparse(node).split())
    except Exception:
        return "?"


_KEEP_BUDGET = 48   # option B: keep a stmt's expr VERBATIM (args / literals)
                    # when its unparse is at most this long; else fall back to
                    # ordered call names. Tunable: smaller = more CR, less recall.


def _keep_or_calls(node: ast.AST) -> List[str]:
    """Render short expressions in full and summarize longer ones by their calls."""
    s = _full_expr(node)
    if len(s) <= _KEEP_BUDGET:
        return [s]
    calls = _ordered_calls(node)
    return calls if calls else [s]


def _param_names(args: ast.arguments) -> str:
    """Compact comma-joined parameter names of a (nested) def signature."""
    names: List[str] = []
    for a in list(getattr(args, "posonlyargs", [])) + list(args.args):
        names.append(a.arg)
    if args.vararg:
        names.append("*" + args.vararg.arg)
    for a in args.kwonlyargs:
        names.append(a.arg)
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return ",".join(names)


def _target_names(node: ast.AST) -> List[str]:
    """Assigned name(s) of an assignment target (Name / self.attr / tuple)."""
    out: List[str] = []
    if isinstance(node, ast.Name):
        out.append(node.id)
    elif isinstance(node, ast.Attribute):
        try:
            s = ast.unparse(node)
        except Exception:
            s = node.attr
        if len(s) < 40:
            out.append(s)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            out.extend(_target_names(e))
    elif isinstance(node, ast.Starred):
        out.extend(_target_names(node.value))
    return out


def _body_buckets(body: List[ast.stmt]) -> str:
    """Group a long body's dependencies by kind."""
    writes: List[str] = []
    calls: List[str] = []
    controls: List[str] = []
    effects: List[str] = []
    defs: List[str] = []
    ret: Optional[str] = None

    def add(seq: List[str], v: str) -> None:
        if v and v not in seq:
            seq.append(v)

    def _calls_of(node: Optional[ast.AST]) -> None:
        if node is not None:
            for nm in _ordered_calls(node):
                add(calls, nm)

    def walk(stmts: List[ast.stmt]) -> None:
        nonlocal ret
        for s in _skip_docstring(stmts):
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(defs, "def " + s.name)          # keep nested-def boundary
                continue
            if isinstance(s, ast.ClassDef):
                add(defs, "class " + s.name)
                continue
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    for nm in _target_names(t):
                        add(writes, nm)
                _calls_of(s.value)
            elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
                for nm in _target_names(s.target):
                    add(writes, nm)
                _calls_of(s.value)
            elif isinstance(s, ast.Expr):
                if isinstance(s.value, (ast.Yield, ast.YieldFrom)):
                    add(effects, "yield")
                _calls_of(s.value)
            elif isinstance(s, ast.Return):
                if s.value is not None:
                    if ret is None:
                        ret = _full_expr(s.value)    # verbatim terminal return
                    _calls_of(s.value)
            elif isinstance(s, ast.Raise):
                t = _exc_type_name(s.exc) if s.exc is not None else None
                add(effects, "raise " + t if t else "raise")
                _calls_of(s.exc)
            elif isinstance(s, ast.Assert):
                add(effects, "assert")
                _calls_of(s.test)
            elif isinstance(s, ast.If):
                add(controls, "if " + _short_expr(s.test, 64))
                _calls_of(s.test)
                walk(s.body); walk(s.orelse)
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                add(controls, "for " + _short_expr(s.target) + " in "
                    + _short_expr(s.iter))
                _calls_of(s.iter)
                walk(s.body); walk(s.orelse)
            elif isinstance(s, ast.While):
                add(controls, "while " + _short_expr(s.test, 64))
                _calls_of(s.test)
                walk(s.body); walk(s.orelse)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                add(controls, "with")
                for it in s.items:
                    _calls_of(it.context_expr)
                walk(s.body)
            elif isinstance(s, ast.Try):
                walk(s.body)
                for h in s.handlers:
                    walk(h.body)
                walk(s.orelse); walk(s.finalbody)
            else:
                _calls_of(s)

    walk(body)
    segs: List[str] = []
    if writes:
        segs.append("writes[" + ",".join(writes) + "]")
    if controls:
        segs.append("controls[" + ";".join(controls) + "]")
    if calls:
        segs.append("calls[" + ",".join(calls) + "]")
    if effects:
        segs.append("effects[" + ",".join(effects) + "]")
    if defs:
        segs.append("defs[" + ",".join(defs) + "]")
    if ret is not None:
        segs.append("return[" + ret + "]")
    return " ".join(segs)


def _stmt_ops(stmt: ast.stmt, depth: int = 0) -> List[str]:
    if depth > 4:
        return _ordered_calls(stmt)
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return ["return"]
        return ["return " + _full_expr(stmt.value)]
    if isinstance(stmt, ast.Raise):
        return ["raise " + _full_expr(stmt.exc)] if stmt.exc is not None else ["raise"]
    if isinstance(stmt, ast.Assert):
        s = _full_expr(stmt.test)
        return ["assert " + s] if len(s) <= _KEEP_BUDGET else ["assert"] + _ordered_calls(stmt.test)
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        if stmt.value is None:
            return []
        if isinstance(stmt, ast.Assign):
            tgts = [nm for t in stmt.targets for nm in _target_names(t)]
        else:
            tgts = _target_names(stmt.target)
        parts = _keep_or_calls(stmt.value)
        if tgts and parts:
            parts = [",".join(tgts) + "=" + parts[0]] + parts[1:]
        return parts
    if isinstance(stmt, ast.Expr):
        # B: short call statements stay VERBATIM (recovers assert/log/call args);
        # long ones collapse to ordered call names.
        return _keep_or_calls(stmt.value)
    if isinstance(stmt, ast.If):
        out = ["if " + _short_expr(stmt.test) + ":"]
        for s in _skip_docstring(stmt.body):
            out += _stmt_ops(s, depth + 1)
        if stmt.orelse:
            out.append("else:")
            for s in stmt.orelse:
                out += _stmt_ops(s, depth + 1)
        return out
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        out = ["for " + _short_expr(stmt.target) + " in "
               + _short_expr(stmt.iter) + ":"]
        for s in stmt.body:
            out += _stmt_ops(s, depth + 1)
        return out
    if isinstance(stmt, ast.While):
        out = ["while " + _short_expr(stmt.test) + ":"]
        for s in stmt.body:
            out += _stmt_ops(s, depth + 1)
        return out
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        # with-contexts (locks, plates, open(...)) are semantically central and
        # rarely huge — render them VERBATIM rather than capping to ``…``.
        ctx = ",".join(_full_expr(i.context_expr) for i in stmt.items)
        out = ["with " + ctx + ":"]
        for s in stmt.body:
            out += _stmt_ops(s, depth + 1)
        return out
    if isinstance(stmt, ast.Try):
        out = ["try:"]
        for s in stmt.body:
            out += _stmt_ops(s, depth + 1)
        for h in stmt.handlers:
            out.append("except " + (_short_expr(h.type) if h.type is not None
                                    else "*") + ":")
            for s in h.body:
                out += _stmt_ops(s, depth + 1)
        return out
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Keep nested function boundaries and signatures in the parent sketch.
        return ["def " + stmt.name + "(" + _param_names(stmt.args) + "):"]
    if isinstance(stmt, ast.ClassDef):
        return ["class " + stmt.name + ":"]
    # any other statement type: surface whatever calls it contains.
    return _ordered_calls(stmt)


def _body_chain(body: List[ast.stmt], max_ops: int = 24) -> str:
    """Join dependencies in order, grouping them by kind above the length limit."""
    body = _skip_docstring(body)
    parts: List[str] = []
    for stmt in body:
        parts.extend(_stmt_ops(stmt))
    # collapse immediate duplicates (e.g. repeated .append) and cap length.
    chain: List[str] = []
    for p in parts:
        if p and (not chain or chain[-1] != p):
            chain.append(p)
    if len(chain) > max_ops:
        return _body_buckets(body)
    return "->".join(chain)


def _body_summary(body: List[ast.stmt]) -> str:
    return _body_chain(body)


def _exc_type_name(exc: ast.AST) -> Optional[str]:
    if isinstance(exc, ast.Call):
        return _call_target_name(exc)
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def _extract_self_assign_attrs(stmts: List[ast.stmt]) -> List[str]:
    attrs: list = []
    for s in stmts:
        if isinstance(s, ast.Assign) and len(s.targets) == 1:
            t = s.targets[0]
            if (isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                attrs.append(t.attr)
    return attrs


def _method_sketch(fn: ast.AST) -> str:
    prefix = ""
    if any(_is_named_decorator(d, "property") for d in fn.decorator_list):
        prefix = "@prop "
    elif any(_is_named_decorator(d, "staticmethod") for d in fn.decorator_list):
        prefix = "@static "
    elif any(_is_named_decorator(d, "classmethod") for d in fn.decorator_list):
        prefix = "@cls "
    elif any(_is_named_decorator(d, "cached_property") for d in fn.decorator_list):
        prefix = "@cached_prop "
    if isinstance(fn, ast.AsyncFunctionDef):
        prefix = "async " + prefix
    sig = _func_signature(fn, include_name=True)
    body = _skip_docstring(fn.body)
    if (len(body) == 1 and isinstance(body[0], ast.Return)
            and body[0].value is not None):
        try:
            ret_src = ast.unparse(body[0].value)
        except Exception:
            ret_src = "..."
        return f"{prefix}{sig}=>{ret_src}"
    if (len(body) == 1 and isinstance(body[0], ast.Return)
            and body[0].value is None):
        return f"{prefix}{sig}=>None"
    if (len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr))):
        return f"{prefix}{sig}"
    body_tag = _body_summary(body)
    return f"{prefix}{sig}: {body_tag}"


# Slot vocabulary

SlotExtractor = Callable[[AutoRegion], str]


# ---- class slots ----

def _slot_class_name(r: AutoRegion) -> str:
    return r.node.name if isinstance(r.node, ast.ClassDef) else ""


def _slot_bases(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    return ",".join(_short_name(b) for b in r.node.bases)


def _slot_decorators(r: AutoRegion) -> str:
    decos = getattr(r.node, "decorator_list", None) or []
    return _summarize_decorators(decos)


def _find_init(n: ast.ClassDef):
    for s in _skip_docstring(n.body):
        if (isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and s.name == "__init__"):
            return s
    return None


def _slot_init_params(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    init = _find_init(r.node)
    if init is None:
        return ""
    a = init.args
    params: list = []
    for arg in list(a.posonlyargs) + list(a.args):
        if arg.arg == "self":
            continue
        params.append(arg.arg)
    if a.vararg:
        params.append("*" + a.vararg.arg)
    for arg in a.kwonlyargs:
        params.append(arg.arg)
    if a.kwarg:
        params.append("**" + a.kwarg.arg)
    return ",".join(params)


def _slot_init_fields(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    init = _find_init(r.node)
    if init is None:
        return ""
    return ",".join(_extract_self_assign_attrs(_skip_docstring(init.body)))


def _slot_class_vars(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    names: list = []
    for s in _skip_docstring(r.node.body):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
        elif (isinstance(s, ast.AnnAssign)
              and isinstance(s.target, ast.Name)):
            names.append(s.target.id)
    return ",".join(names)


def _slot_method_names(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    return ",".join(
        s.name for s in _skip_docstring(r.node.body)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
        and s.name != "__init__")


def _slot_method_sigs(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.ClassDef):
        return ""
    sigs = []
    for s in _skip_docstring(r.node.body):
        if (isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                and s.name != "__init__"):
            sigs.append(_method_sketch(s))
    return ";".join(sigs)


# ---- function slots ----

def _slot_func_name(r: AutoRegion) -> str:
    return (r.node.name
            if isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "")


def _slot_params(r: AutoRegion) -> str:
    """Render parameter names and kind markers without annotations or defaults."""
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    a = r.node.args
    parts: list = [arg.arg for arg in a.posonlyargs + a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [arg.arg for arg in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return ",".join(parts)


def _slot_sig(r: AutoRegion) -> str:
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    return _func_signature(r.node, include_name=False)


def _slot_return_type(r: AutoRegion) -> str:
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    return _compact_annotation(r.node.returns) if r.node.returns else ""


def _slot_return_expr(r: AutoRegion) -> str:
    if not isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    body = _skip_docstring(r.node.body)
    if (len(body) == 1 and isinstance(body[0], ast.Return)
            and body[0].value is not None):
        try:
            return ast.unparse(body[0].value)
        except Exception:
            return "..."
    return ""


def _slot_body_summary(r: AutoRegion) -> str:
    """Body summary works for both functions and any node with a .body list."""
    body = getattr(r.node, "body", None)
    if not (isinstance(body, list) and body):
        return ""
    body = _skip_docstring(body)
    return _body_summary(body) if body else ""


def _slot_calls(r: AutoRegion) -> str:
    body = getattr(r.node, "body", None)
    if not (isinstance(body, list) and body):
        return ""
    names: List[str] = []
    for stmt in _skip_docstring(body):
        for n in _ordered_calls(stmt):
            if not names or names[-1] != n:
                names.append(n)
    return "->".join(names)


def _func_body_blocks(r: AutoRegion) -> List[ast.stmt]:
    n = r.node
    if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    return list(_skip_docstring(n.body))


def _slot_body_blocks(r: AutoRegion) -> str:
    """Render the flat statement chain used as a slot fallback."""
    parts: List[str] = []
    for b in _func_body_blocks(r):
        try:
            parts.append(ast.unparse(b))
        except Exception:
            parts.append("...")
    return "->".join(parts)


# ---- control_flow slots ----

def _slot_cond(r: AutoRegion) -> str:
    n = r.node
    if isinstance(n, (ast.If, ast.While)):
        try:
            return ast.unparse(n.test)
        except Exception:
            return "?"
    return ""


# Structural idiom slots

def _branch_value(stmt: ast.stmt):
    """Value carried by a single-statement branch (Return or simple Assign)."""
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        return stmt.value
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        return stmt.value
    return None


def _slot_tern_true(r: AutoRegion) -> str:
    """The value produced by the if-branch (for an `if/else` two-way region)."""
    n = r.node
    if isinstance(n, ast.If) and len(n.body) == 1:
        v = _branch_value(n.body[0])
        if v is not None:
            try:
                return ast.unparse(v)
            except Exception:
                return "?"
    return ""


def _slot_tern_false(r: AutoRegion) -> str:
    """The value produced by the else-branch."""
    n = r.node
    if isinstance(n, ast.If) and len(n.orelse) == 1:
        v = _branch_value(n.orelse[0])
        if v is not None:
            try:
                return ast.unparse(v)
            except Exception:
                return "?"
    return ""


def _slot_tern_target(r: AutoRegion) -> str:
    """The common assignment target, when both branches assign (else empty)."""
    n = r.node
    if (isinstance(n, ast.If) and len(n.body) == 1
            and isinstance(n.body[0], ast.Assign)
            and len(n.body[0].targets) == 1):
        try:
            return ast.unparse(n.body[0].targets[0])
        except Exception:
            return "?"
    return ""


def _slot_loop_target(r: AutoRegion) -> str:
    n = r.node
    if isinstance(n, ast.For):
        try:
            return ast.unparse(n.target)
        except Exception:
            return "?"
    return ""


def _slot_iter_expr(r: AutoRegion) -> str:
    n = r.node
    if isinstance(n, ast.For):
        try:
            return ast.unparse(n.iter)
        except Exception:
            return "?"
    return ""


def _slot_else_summary(r: AutoRegion) -> str:
    orelse = getattr(r.node, "orelse", None)
    if not (isinstance(orelse, list) and orelse):
        return ""
    return _body_summary(_skip_docstring(orelse))


def _slot_except_types(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.Try):
        return ""
    types: list = []
    for h in r.node.handlers:
        if h.type is not None:
            try:
                types.append(ast.unparse(h.type))
            except Exception:
                pass
        else:
            types.append("*")
    return ",".join(types)


def _slot_ctx_expr(r: AutoRegion) -> str:
    if not isinstance(r.node, ast.With):
        return ""
    parts: list = []
    for item in r.node.items:
        try:
            parts.append(ast.unparse(item.context_expr))
        except Exception:
            pass
    return ",".join(parts)


def _slot_full_test(r: AutoRegion) -> str:
    n = r.node
    if isinstance(n, (ast.If, ast.While)):
        try:
            return ast.unparse(n.test)
        except Exception:
            return "?"
    if isinstance(n, ast.IfExp):
        try:
            return ast.unparse(n.test)
        except Exception:
            return "?"
    return ""


# ---- data_flow slots (stmt_range regions) ----

def _stmt_range_slice(r: AutoRegion) -> List[ast.stmt]:
    if r.span_kind == "node":
        # ALG2 block-level: a single-statement data_flow region — treat the
        # node itself as a 1-statement range so the data_flow slots apply.
        return [r.node] if isinstance(r.node, ast.stmt) else []
    if r.parent_node is None:
        return []
    body = getattr(r.parent_node, r.body_attr, None) or []
    return list(body[r.first_idx:r.last_idx + 1])


def _slot_dict_build(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    if len(stmts) < 2:
        return ""
    s0 = stmts[0]
    if not (isinstance(s0, ast.Assign) and len(s0.targets) == 1
            and isinstance(s0.targets[0], ast.Name)
            and isinstance(s0.value, ast.Dict) and not s0.value.keys):
        return ""
    d = s0.targets[0].id
    pairs: List[str] = []
    for s in stmts[1:]:
        tgt = s.targets[0] if (isinstance(s, ast.Assign) and len(s.targets) == 1) else None
        if not (isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Name) and tgt.value.id == d):
            return ""
        try:
            k = ast.unparse(tgt.slice)
            v = ast.unparse(s.value)
        except Exception:
            return ""
        pairs.append(f"{k}:{v}")
    return f"{d}={{{','.join(pairs)}}}"


def _slot_first_targets(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    if not stmts:
        return ""
    first = stmts[0]
    if isinstance(first, ast.Assign):
        names = []
        for t in first.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, ast.Tuple):
                for sub in t.elts:
                    if isinstance(sub, ast.Name):
                        names.append(sub.id)
        return ",".join(names)
    if isinstance(first, ast.AnnAssign) and isinstance(first.target, ast.Name):
        return first.target.id
    return ""


def _slot_last_expr(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    if not stmts:
        return ""
    last = stmts[-1]
    try:
        if isinstance(last, ast.Return) and last.value is not None:
            return ast.unparse(last.value)
        if isinstance(last, ast.Assign) and last.value is not None:
            return ast.unparse(last.value)
        if isinstance(last, ast.Expr):
            return ast.unparse(last.value)
    except Exception:
        return "?"
    return ""


def _slot_call_chain(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    return _body_chain(stmts) if stmts else ""


def _slot_n_stmts(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    return str(len(stmts)) if stmts else ""


def _slot_range_body_summary(r: AutoRegion) -> str:
    stmts = _stmt_range_slice(r)
    return _body_summary(stmts) if stmts else ""


def _slot_stmt_chain(r: AutoRegion) -> str:
    """Render the chain for a single statement, such as an assignment or loop."""
    n = r.node
    return _body_chain([n]) if isinstance(n, ast.stmt) else ""


# ---- api_idiom slots (ast.Call) ----

def _slot_call_target(r: AutoRegion) -> str:
    n = r.node
    if not isinstance(n, ast.Call):
        return ""
    f = n.func
    if isinstance(f, ast.Name):
        return f.id
    # Keep complex callee references intact, including subscripts.
    try:
        return ast.unparse(f)
    except Exception:
        return getattr(f, "attr", "") or "?"


def _slot_n_args(r: AutoRegion) -> str:
    n = r.node
    if not isinstance(n, ast.Call):
        return ""
    return str(len(n.args))


def _slot_kw_keys(r: AutoRegion) -> str:
    n = r.node
    if not isinstance(n, ast.Call):
        return ""
    return ",".join(k.arg for k in n.keywords if k.arg)


def _slot_first_arg(r: AutoRegion) -> str:
    n = r.node
    if not isinstance(n, ast.Call) or not n.args:
        return ""
    try:
        return ast.unparse(n.args[0])
    except Exception:
        return "?"


def _slot_rest_args(r: AutoRegion) -> str:
    n = r.node
    if not isinstance(n, ast.Call) or len(n.args) <= 1:
        return ""
    parts: list = []
    for a in n.args[1:]:
        try:
            parts.append(ast.unparse(a))
        except Exception:
            parts.append("?")
    return ", ".join(parts)


def _slot_call_args_combined(r: AutoRegion) -> str:
    """Render positional and keyword arguments with one comma separator."""
    n = r.node
    if not isinstance(n, ast.Call):
        return ""
    parts: list = []
    for a in n.args:
        try:
            parts.append(ast.unparse(a))
        except Exception:
            parts.append("?")
    for kw in n.keywords:
        try:
            v = ast.unparse(kw.value)
        except Exception:
            v = "?"
        if kw.arg is None:
            parts.append(f"**{v}")    # **kwargs splat
        else:
            parts.append(f"{kw.arg}={v}")
    return ", ".join(parts)


# ---- expression slots ----

_OP_SYM = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
    "Pow": "**", "FloorDiv": "//",
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
    "Gt": ">", "GtE": ">=", "Is": "is", "IsNot": "is not",
    "In": "in", "NotIn": "not in",
    "And": "and", "Or": "or", "Not": "not", "USub": "-", "UAdd": "+",
    "Invert": "~", "BitAnd": "&", "BitOr": "|", "BitXor": "^",
    "LShift": "<<", "RShift": ">>",
}


def _slot_var(r: AutoRegion) -> str:
    n = r.node
    if isinstance(n, ast.Compare):
        try:
            return ast.unparse(n.left)
        except Exception:
            return "?"
    if isinstance(n, ast.BinOp):
        try:
            return ast.unparse(n.left)
        except Exception:
            return "?"
    if isinstance(n, ast.UnaryOp):
        try:
            return ast.unparse(n.operand)
        except Exception:
            return "?"
    if isinstance(n, ast.BoolOp) and n.values:
        try:
            return ast.unparse(n.values[0])
        except Exception:
            return "?"
    return ""


def _slot_op_sym(r: AutoRegion) -> str:
    n = r.node
    op_node = None
    if isinstance(n, ast.Compare) and n.ops:
        op_node = n.ops[0]
    elif isinstance(n, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
        op_node = n.op
    if op_node is None:
        return ""
    return _OP_SYM.get(type(op_node).__name__, type(op_node).__name__)


def _slot_lit(r: AutoRegion) -> str:
    """Render the right operand, including nonliteral expressions."""
    n = r.node
    if isinstance(n, ast.Compare) and n.comparators:
        c = n.comparators[0]
        if isinstance(c, ast.Constant):
            return repr(c.value)
        try:
            return ast.unparse(c)
        except Exception:
            return ""
    if isinstance(n, ast.BinOp):
        if isinstance(n.right, ast.Constant):
            return repr(n.right.value)
        try:
            return ast.unparse(n.right)
        except Exception:
            return ""
    return ""


# ---- module slots ----

def _slot_imports(r: AutoRegion) -> str:
    """All top-level Import/ImportFrom names — comma joined as ``a,b,c.d``."""
    if not isinstance(r.node, ast.Module):
        return ""
    items: list = []
    for stmt in r.node.body:
        if isinstance(stmt, ast.Import):
            items.extend(a.name for a in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            mod = stmt.module or ""
            items.extend(f"{mod}.{a.name}" if mod else a.name
                         for a in stmt.names)
    return ",".join(items)


def _slot_globals(r: AutoRegion) -> str:
    """Top-level Assign / AnnAssign target names."""
    if not isinstance(r.node, ast.Module):
        return ""
    names: list = []
    for stmt in r.node.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
        elif (isinstance(stmt, ast.AnnAssign)
              and isinstance(stmt.target, ast.Name)):
            names.append(stmt.target.id)
    return ",".join(names)


def _slot_entry_call(r: AutoRegion) -> str:
    """The single expression inside ``if __name__ == "__main__": ...``."""
    if not isinstance(r.node, ast.Module):
        return ""
    for stmt in r.node.body:
        if (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
                and isinstance(stmt.test.left, ast.Name)
                and stmt.test.left.id == "__name__"):
            if stmt.body and isinstance(stmt.body[0], ast.Expr):
                try:
                    return ast.unparse(stmt.body[0].value)
                except Exception:
                    pass
    return ""


# ---- the menu, exposed both to SynthLLM and to the renderer ----

SLOT_VOCAB: Dict[str, Dict[str, SlotExtractor]] = {
    "class": {
        "{class_name}":    _slot_class_name,
        "{bases}":         _slot_bases,
        "{decorators}":    _slot_decorators,
        "{init_params}":   _slot_init_params,
        "{init_fields}":   _slot_init_fields,
        "{class_vars}":    _slot_class_vars,
        "{method_names}":  _slot_method_names,
        "{method_sigs}":   _slot_method_sigs,
    },
    "function": {
        "{func_name}":     _slot_func_name,
        "{params}":        _slot_params,
        "{sig}":           _slot_sig,
        "{return_type}":   _slot_return_type,
        "{return_expr}":   _slot_return_expr,
        "{body_blocks}":   _slot_body_blocks,
        "{calls}":         _slot_calls,
    },
    "control_flow": {
        "{cond}":          _slot_cond,
        "{full_test}":     _slot_full_test,
        "{loop_target}":   _slot_loop_target,
        "{iter_expr}":     _slot_iter_expr,
        "{body_chain}":    _slot_body_summary,
        "{stmt_chain}":    _slot_stmt_chain,
        "{else_chain}":    _slot_else_summary,
        "{except_types}":  _slot_except_types,
        "{ctx_expr}":      _slot_ctx_expr,
        # idiom (2nd mining line): two-branch if/else -> ternary
        "{tern_true}":     _slot_tern_true,
        "{tern_false}":    _slot_tern_false,
        "{tern_target}":   _slot_tern_target,
    },
    "data_flow": {
        "{first_targets}":   _slot_first_targets,
        "{last_expr}":       _slot_last_expr,
        "{call_chain}":      _slot_call_chain,
        "{stmt_chain}":      _slot_stmt_chain,
        "{n_stmts}":         _slot_n_stmts,
        "{body_chain}":      _slot_range_body_summary,
        # idiom (statement-block): dict init + d[k]=v fills -> dict literal
        "{dict_build}":      _slot_dict_build,
    },
    "api_idiom": {
        "{call_target}":         _slot_call_target,
        "{n_args}":              _slot_n_args,
        "{kw_keys}":             _slot_kw_keys,
        "{first_arg}":           _slot_first_arg,
        "{rest_args}":           _slot_rest_args,
        "{call_args_combined}":  _slot_call_args_combined,
    },
    "expression": {
        "{var}":           _slot_var,
        "{op_sym}":        _slot_op_sym,
        "{lit}":           _slot_lit,
    },
    "module": {
        "{imports}":       _slot_imports,
        "{globals}":       _slot_globals,
        "{entry_call}":    _slot_entry_call,
    },
}


_SLOT_DESCRIPTIONS = {
    "{class_name}":      "name of the class",
    "{bases}":           "comma-separated base class names",
    "{decorators}":      "decorator names like @dataclass",
    "{init_params}":     "__init__ parameters (excluding self)",
    "{init_fields}":     "attribute names assigned in __init__ as self.X = X",
    "{class_vars}":      "class-level variable / annotated field names",
    "{method_names}":    "non-__init__ method names",
    "{method_sigs}":     "non-__init__ method signatures: name(params)[->ret]",
    "{func_name}":       "name of the function",
    "{params}":          "parameter names",
    "{sig}":             "compact signature: (params)[->ret]",
    "{return_type}":     "compact return type",
    "{return_expr}":     "return expression (for one-line return functions)",
    "{body_chain}":      "ORDERED dependency chain of the body: the calls it "
                         "makes IN SOURCE ORDER + control-flow shape "
                         "(if/for/try) + return/raise, joined by -> "
                         "(e.g. norm1->attn->dropout->mlp->return). Keeps "
                         "WHICH calls in WHAT order; drops glue. NEVER a vague "
                         "'# N calls' count.",
    "{calls}":           "ordered call-target names in the body (source order, "
                         "joined ->), without control-flow shape",
    "{stmt_chain}":      "the WHOLE single statement compressed to its own "
                         "dependency chain (target=value / return X / if cond:->"
                         "... ), keeping the statement head + structure, dropping "
                         "only glue — v9 fidelity. Use as a block-level template "
                         "(`{stmt_chain}`) on single-statement data_flow / "
                         "control_flow regions to fill combiner holes.",
    "{body_blocks}":     "the body's top-level statements, EACH kept as its own "
                         "block joined by -> ; unlike {body_chain} this leaves "
                         "every statement an OPEN hole for block-level schemas to "
                         "compress (the combiner template only de-newlines). Use "
                         "in `def {sig}: {body_blocks}` to build a combiner.",
    "{cond}":            "if/while test expression",
    "{full_test}":       "full unparse of test condition (If/While/IfExp) — "
                         "use when the condition's exact form matters",
    "{loop_target}":     "for-loop target variable",
    "{iter_expr}":       "for-loop iterable expression",
    "{else_chain}":      "ordered dependency chain of the else branch",
    "{except_types}":    "try/except handler type names",
    "{ctx_expr}":        "with-block context manager expression",
    "{tern_true}":       "value of the if-branch when it is a single return/assign "
                         "(for rewriting a two-branch if/else into a ternary)",
    "{tern_false}":      "value of the else-branch when it is a single return/assign",
    "{tern_target}":     "common assignment target when both branches assign to it "
                         "(empty for the return-ternary form)",
    "{first_targets}":   "LHS names of the first stmt in the range",
    "{last_expr}":       "expression value of the last stmt in the range",
    "{call_chain}":      "ORDERED dependency chain across the stmt range: "
                         "calls in source order + control shape + return, "
                         "joined -> (NOT a deduped set, NOT a count)",
    "{n_stmts}":         "stmt count in the range",
    "{call_target}":     "qualified call name (e.g. json.loads, requests.get)",
    "{n_args}":          "number of positional arguments",
    "{kw_keys}":         "keyword-argument names",
    "{first_arg}":       "first positional argument expression",
    "{rest_args}":       "ALL positional args after the first, comma-joined "
                         "(use when arity may be >1 and tail args carry info)",
    "{call_args_combined}":
                         "ALL positional args AND all keyword args, joined "
                         "with a single comma separator. Use this — not a "
                         "concatenation of separate slots — whenever you need "
                         "to preserve the full call signature (e.g. "
                         "``{call_target}({call_args_combined})`` always "
                         "renders as valid Python regardless of arity).",
    "{var}":             "main operand expression (LHS for binop/compare)",
    "{op_sym}":          "operator symbol (==, +, is, ...)",
    "{lit}":             "literal value on the right (for cmp-to-literal patterns)",
    "{imports}":         "comma-joined top-level import names",
    "{globals}":         "top-level global / annotated variable names",
    "{entry_call}":      "expression inside the __main__ guard",
}


def supported_region_types() -> List[str]:
    return list(SLOT_VOCAB.keys())


def _slot_menu(rt: str) -> str:
    if rt not in SLOT_VOCAB:
        return "(none)"
    return "\n".join(f"  {s}  — {_SLOT_DESCRIPTIONS.get(s, '?')}"
                     for s in SLOT_VOCAB[rt])


def render_template(template: str, region: AutoRegion) -> str:
    rt = region.region_type
    if rt not in SLOT_VOCAB:
        return template
    out = template
    for slot, extractor in SLOT_VOCAB[rt].items():
        if slot in out:
            try:
                val = extractor(region)
            except Exception:
                val = ""
            out = out.replace(slot, val)
    return out


# Single-node slots can expose their AST nodes for nested rewrites.

def _inner_return_expr(r: AutoRegion) -> Optional[ast.AST]:
    n = r.node
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = _skip_docstring(n.body)
        if (len(body) == 1 and isinstance(body[0], ast.Return)
                and body[0].value is not None):
            return body[0].value
    return None


def _inner_test(r: AutoRegion) -> Optional[ast.AST]:
    n = r.node
    return n.test if isinstance(n, (ast.If, ast.While, ast.IfExp)) else None


def _inner_iter(r: AutoRegion) -> Optional[ast.AST]:
    n = r.node
    return n.iter if isinstance(n, ast.For) else None


def _inner_first_arg(r: AutoRegion) -> Optional[ast.AST]:
    n = r.node
    if isinstance(n, ast.Call) and n.args:
        return n.args[0]
    return None


def _inner_last_expr(r: AutoRegion) -> Optional[ast.AST]:
    stmts = _stmt_range_slice(r)
    if not stmts:
        return None
    last = stmts[-1]
    if isinstance(last, ast.Return) and last.value is not None:
        return last.value
    if isinstance(last, ast.Assign) and last.value is not None:
        return last.value
    if isinstance(last, ast.Expr):
        return last.value
    return None


_INNER_NODE_SLOTS: Dict[str, Dict[str, Callable[[AutoRegion], Optional[ast.AST]]]] = {
    "function":     {"{return_expr}": _inner_return_expr},
    "control_flow": {"{cond}": _inner_test,
                     "{full_test}": _inner_test,
                     "{iter_expr}": _inner_iter},
    "api_idiom":    {"{first_arg}": _inner_first_arg},
    "data_flow":    {"{last_expr}": _inner_last_expr},
}


# Each top-level function statement becomes a separate hole.
_MULTI_HOLE_SLOTS: Dict[str, Dict[str, Callable[[AutoRegion], List[ast.AST]]]] = {
    "function": {"{body_blocks}": _func_body_blocks},
}

# A newline after a compound statement keeps the next statement outside its suite.
_CG_COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
                ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
if hasattr(ast, "Match"):
    _CG_COMPOUND = _CG_COMPOUND + (ast.Match,)
if hasattr(ast, "TryStar"):
    _CG_COMPOUND = _CG_COMPOUND + (ast.TryStar,)


def _cg_hole(i: int) -> str:
    # \x01 never appears in well-formed source, so markers can't collide with
    # surrounding skeleton text. Mirrors codegist.write_infer._hole.
    return f"\x01CG_HOLE_{i}\x01"


def render_template_structured(template: str, region: AutoRegion):
    """Render a template with AST holes for eligible slots."""
    from ..write_infer import RenderedTemplate
    rt = region.region_type
    if rt not in SLOT_VOCAB:
        return RenderedTemplate(skeleton=template)
    inner_map = _INNER_NODE_SLOTS.get(rt, {})
    out = template
    inner_nodes: List[ast.AST] = []
    marks: List[str] = []
    for slot, nodes_fn in _MULTI_HOLE_SLOTS.get(rt, {}).items():
        if slot not in out:
            continue
        try:
            block_nodes = nodes_fn(region)
        except Exception:
            block_nodes = []
        if len(block_nodes) >= 2:
            hole_marks: List[str] = []
            for bnode in block_nodes:
                m = _cg_hole(len(marks))
                marks.append(m)
                inner_nodes.append(bnode)
                hole_marks.append(m)
            joined = hole_marks[0]
            for k in range(1, len(hole_marks)):
                sep = "\n" if isinstance(block_nodes[k - 1], _CG_COMPOUND) else "->"
                joined += sep + hole_marks[k]
            out = out.replace(slot, joined, 1)
    for slot, extractor in SLOT_VOCAB[rt].items():
        if slot not in out:
            continue
        try:
            val = extractor(region)
        except Exception:
            val = ""
        node = None
        node_fn = inner_map.get(slot)
        if node_fn is not None:
            try:
                node = node_fn(region)
            except Exception:
                node = None
            if node is not None:
                try:
                    if ast.unparse(node) != val:
                        node = None     # mismatch → flatten, never lose info
                except Exception:
                    node = None
        if node is not None:
            # One distinct hole per occurrence, all anchored on the same node.
            while slot in out:
                m = _cg_hole(len(marks))
                out = out.replace(slot, m, 1)
                marks.append(m)
                inner_nodes.append(node)
        else:
            out = out.replace(slot, val)
    return RenderedTemplate(skeleton=out, inner_nodes=inner_nodes, marks=marks)


def _is_pure_chain_template(template: Optional[str]) -> bool:
    """Check whether a template represents the body entirely by a chain slot."""
    if not template:
        return False
    if "{body_chain}" not in template and "{call_chain}" not in template:
        return False
    residual = re.sub(r"\{[A-Za-z_][A-Za-z_0-9]*\}", " ", template)
    residual = re.sub(
        r"\b(def|class|if|elif|else|for|while|in|return|try|except|finally|"
        r"with|raise|yield|async|await|None|True|False|and|or|not|is|lambda|"
        r"imports|globals)\b", " ", residual)
    residual = re.sub(r"[\s:\->=\(\)\.,;\{\}\[\]\*&|%^~<+/@]", "", residual)
    return residual == ""


def _validate_template(template: str, rt: str) -> Optional[str]:
    if not template or not isinstance(template, str):
        return "empty template"
    used = set(re.findall(r"\{[A-Za-z_][A-Za-z_0-9]*\}", template))
    allowed = set(SLOT_VOCAB.get(rt, {}).keys())
    bad = used - allowed
    if bad:
        return f"unknown slots: {','.join(sorted(bad))}"
    # Reject angle-bracket placeholders while allowing sketch arrows.
    if re.search(r"<[A-Za-z_][A-Za-z_0-9]*>", template):
        return ("angle-bracket placeholder (e.g. <param>, <Exc>) is invented "
                "shorthand — use a {slot} or real code instead")
    return None


# Prompts

_PROMPT_SYNTH = """You are designing a compact pseudo-code "sketch" for a recurring Python pattern.

Below are {n} concrete instances that share the same AST structure (region_type={rt}).

{instances_block}

GOAL: design a single template that compresses these into the shortest form that still lets
another LLM:
  (a) identify what the {rt} is — by name when applicable;
  (b) understand its structural composition — fields, params, methods, signatures;
  (c) recognise its key calls / side effects at a high level.

DESIGN BUDGET (drop recoverable GLUE, KEEP the dependency signature):
  The compressed output is a lightweight code-like SKETCH (it need not be executable).
  The rule for what to drop: drop ONLY what a reader could RECONSTRUCT from what remains.
  - DROP (low-information, recoverable): docstrings; type annotations; pure formatting;
    a constructor that only assigns params to fields (keep just the field NAMES);
    implied keywords (``return`` -> ``=>``); and inside a body, the syntactic GLUE —
    intermediate-variable bookkeeping (``residual = x``), exact statement arrangement,
    throwaway local temp names.
  - KEEP (high-information, NOT reconstructable): class / function names; parameter and
    field names; and a body's DEPENDENCY SIGNATURE — WHICH calls it makes in WHAT ORDER,
    the control-flow shape (if / for / try), and what it returns / raises.
  - For a non-trivial body, compress it to that ordered dependency chain via the
    ``{{body_chain}}`` (function / control_flow) or ``{{call_chain}}`` (data_flow) slot;
    it renders the calls in source order, e.g. ``norm1->attn->dropout->mlp->return``.
    This keeps the calls + order while dropping the glue.
  - STRUCTURAL-BLOCK IDIOM: when a contiguous block of statements has a shorter,
    EQUIVALENT standard-Python form, prefer that form via its slot. The clearest case:
    a dict initialised empty and then filled by constant-key subscript assignments
    (``d={{}}``; ``d[k1]=v1``; ``d[k2]=v2``; ...) IS a dict literal — emit ``{{dict_build}}``
    (renders ``d={{k1:v1, k2:v2, ...}}``). This is lossless and stays ordinary Python.
  - NEVER collapse a body to a COUNT or prose ("# 5 calls", "# does forward pass"):
    losing WHICH calls and their order destroys the dependency signature — the one thing
    a downstream LLM needs to recognise the code. Under-compressing is fine; losing the
    signature is not.
  - STAY CODE-LIKE: real Python identifiers + ordinary code punctuation only; no ``<x>``
    placeholders, no invented markers, no natural-language prose.

EXAMPLES (original -> good sketch):
  GOOD — glue dropped, dependency signature kept:
    class User: __init__(self,name,age): self.name=name; self.age=age
        ->  class User{{name,age}}
    def get_name(user): return user.name
        ->  def get_name(user)=>user.name
    def forward(self,x): residual=x; x=self.norm1(x); a=self.attn(x); x=residual+self.drop(a); return x
        ->  def forward(self,x): {{body_chain}}   (renders norm1->attn->drop->return: calls in order, glue gone)
    state={{}}; state["model"]=model.state_dict(); state["step"]=step
        ->  {{dict_build}}   (renders state={{"model":model.state_dict(),"step":step}}: lossless dict literal)
  BAD — dependency signature destroyed (DO NOT do this):
    def forward(self,x): ...the body above...
        ->  "def {{func_name}}: # does a forward pass"   <-- WRONG: literal prose/count loses
            WHICH calls + their order. Use "def {{func_name}}{{sig}}: {{body_chain}}" instead.

AVAILABLE SLOTS for region_type={rt}:
{slot_menu}

Rules:
  - Use ONLY slots from the menu. Slots not on the menu make your output invalid.
  - Literal characters are allowed: ``def``, ``class``, ``=>``, ``;``, ``{{``, ``}}``, etc.
  - Aim for the shortest template that satisfies the goal.

OUTPUT FORMAT — JSON object, nothing else:
{{
  "template": "<template string>",
  "rationale": "<one short sentence: what's preserved, what's dropped>"
}}

REFERENCE EXAMPLES (across region_types):
  - one-line return function:        "def {{func_name}}({{params}})=>{{return_expr}}"
  - multi-line function body:        "def {{func_name}}{{sig}}: {{body_chain}}"
  - class with pure self-assign init: "class {{class_name}}{{{{{{init_fields}}}}}}"
  - class with methods + init:        "class {{class_name}}{{{{init({{init_params}}); methods{{{{{{method_sigs}}}}}}}}}}"
  - if-return guard:                  "if {{cond}}: return"
  - for-loop body chain:             "for {{loop_target}} in {{iter_expr}}: {{body_chain}}"
  - try-except chain:                "try: {{body_chain}} except {{except_types}}: {{else_chain}}"
  - stmt-chain (data_flow):           "{{first_targets}}=...{{call_chain}}->{{last_expr}}"
  - API call:                         "{{call_target}}({{call_args_combined}})"
  - expression idiom:                 "{{var}}{{op_sym}}{{lit}}"
  - module:                           "imports{{{{{{imports}}}}}}; globals{{{{{{globals}}}}}}"

{feedback_block}Design your template now."""


_PROMPT_GENERALITY = """You are validating whether a candidate compression template, designed for one
specific bucket of AST shapes, can SAFELY apply to OTHER shapes of the same
region_type without losing semantic information.

TEMPLATE under test (region_type={rt}):
  {template}

The template was designed by looking at instances from a SINGLE bucket. We
now want to test whether broadcasting it across the entire region_type is
safe. Below are {n} instances drawn from NEIGHBOURING buckets (same
region_type, different fingerprints). For each, the template's rendered
output is shown beside the original.

{instances_block}

For EACH neighbour, decide whether applying this template to it dropped ONLY recoverable
boilerplate, or whether it destroyed distinctive content:

  SAFE   — on this neighbour the template KEPT the dependency signature (identity, structural
           names, and the body's calls IN ORDER + control-flow shape) and dropped only
           recoverable glue (docstrings, type hints, self.x=x init with field names kept,
           the return keyword, intermediate-variable bookkeeping, formatting). A body rendered
           as its ORDERED CALL CHAIN (e.g. norm1->attn->dropout->return) is SAFE.
  UNSAFE — on this neighbour the template DESTROYED the dependency signature: lost WHICH calls
           or their ORDER (body collapsed to a vague count/prose), wrong identity, dropped
           structural names, or a load-bearing condition lost.

Then aggregate:

  ALL instances SAFE          → BROAD    (safe to apply across the region_type)
  ANY signature-loss instance → NARROW (template only safe on its own bucket)

Dropping recoverable glue is fine (mark SAFE); destroying a neighbour's dependency signature
(which-calls/order, identity, names) is not (mark UNSAFE). For each UNSAFE instance, say what was lost.

OUTPUT FORMAT — JSON object, nothing else:
{{
  "verdict": "BROAD" | "NARROW",
  "per_instance": [
    {{"i": 1, "safe": true, "why": "all identifiers and args preserved"}},
    {{"i": 2, "safe": false, "why": "dropped second arg `f`"}},
    ...
  ],
  "summary": "<one short sentence>"
}}"""


_PROMPT_REFINE = """The compression template you previously designed was accepted by the
single-instance safety critic, but was rejected by the cross-bucket
generality critic — it loses information on OTHER shapes within the same
region_type.

PREVIOUS TEMPLATE (region_type={rt}):
  {old_template}

GENERALITY-CRITIC FEEDBACK — these neighbouring instances were rendered
unsafely:

{failures_block}

YOUR TASK: design a NEW template that:
  1. Preserves the information lost in the failures above.
  2. STILL saves tokens vs the original (otherwise compression is pointless).

Slots like ``{{call_args_combined}}`` / ``{{rest_args}}`` / ``{{full_test}}``
may help when you need to keep variable-length sequences or full expressions.
For api_idiom calls always use ``{{call_target}}({{call_args_combined}})`` —
never try to concatenate separate args/kwargs slots, that pattern is no
longer supported because it produced invalid Python on mixed calls.

AVAILABLE SLOTS for region_type={rt}:
{slot_menu}

Rules:
  - Use ONLY slots from the menu. Slots not on the menu make your output invalid.
  - You may use any standard literal characters.

IMPORTANT: If you judge that NO template can simultaneously be safe across
the failing instances AND save tokens — say so explicitly. Don't force a
bad compromise.

OUTPUT FORMAT — JSON object, nothing else:

If you can refine:
{{
  "template": "<new template using only menu slots>",
  "rationale": "<one short sentence: how it handles the failing cases>"
}}

If you have to give up:
{{
  "give_up": true,
  "reason": "<one short sentence: why no safe compact form exists for this shape>"
}}"""


_PROMPT_CRITIC = """ORIGINAL Python code:
```python
{original}
```

PROPOSED COMPRESSED SKETCH:
```
{rendered}
```

This is a lightweight code-like SKETCH (it need not be executable). Judge by UNDERSTANDING.
The test: the sketch may drop recoverable GLUE, but must keep the DEPENDENCY SIGNATURE.

ACCEPT if it kept the dependency signature — the identity (class / function name), structural
names (params / fields), and, for a body, the calls it makes IN ORDER + the control-flow shape
(if/for/try) + what it returns/raises — while dropping only recoverable glue (docstrings, type
hints, formatting, repetitive ``self.x = x`` init when field names are kept, intermediate-
variable bookkeeping, the ``return`` keyword). A body compressed to its ORDERED CALL CHAIN
(e.g. ``norm1->attn->dropout->return``) is GOOD — accept it; the glue it dropped is recoverable.
REJECT if it destroyed the dependency signature: lost WHICH calls or their ORDER (e.g. a body
collapsed to a vague ``# N calls`` count or prose like ``# does X``), lost the identity, or
dropped structural names / a load-bearing condition. UNDER-compression is fine; losing the
signature is not.
ALSO reject if it stops being CODE: prose / natural-language, or invented placeholder tokens
or shorthand (``<x>``, ``CALL(...)``). Real Python identifiers, calls, and ``->`` chains are fine.

OUTPUT FORMAT — JSON object, nothing else:
{{
  "verdict": "YES" or "NO",
  "reason": "<one short sentence>",
  "suggestion": "<if NO: a specific slot from the menu to add back or a structural hint. empty if YES>"
}}"""


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# Template synthesis and critique

@dataclass
class LLMSynthResult:
    template: str
    rationale: str
    rounds: int                       # Phase-A safety rounds used
    critic_verdict: str               # final safety verdict YES/NO/UNKNOWN
    critic_reason: str
    applies_broadly: bool = False
    refined: bool = False
    generality_verdict: str = ""      # BROAD / NARROW / SKIPPED / ""
    generality_reasons: List[dict] = field(default_factory=list)


@dataclass
class _GeneralityResult:
    verdict: str
    per_instance: List[dict]
    summary: str


@dataclass
class _RefineResult:
    template: str
    rationale: str
    gave_up: bool
    reason: str


def _safety_check(template: str, in_bucket_instance: AutoRegion,
                  llm: LLMClient) -> tuple[bool, str, str]:
    try:
        rendered = render_template(template, in_bucket_instance)
    except Exception as e:
        return False, f"render error: {e}", ""
    prompt = _PROMPT_CRITIC.format(
        original=in_bucket_instance.source.strip(),
        rendered=rendered,
    )
    try:
        reply = llm.chat(prompt)
    except Exception as e:
        log.warning("safety LLM call failed: %s", e)
        return False, f"llm error: {e}", ""
    parsed = _parse_json(reply)
    if not parsed:
        return False, "malformed critic reply", ""
    verdict = (parsed.get("verdict") or "").strip().upper()
    reason = (parsed.get("reason") or "").strip()
    suggestion = (parsed.get("suggestion") or "").strip()
    return verdict == "YES", reason, suggestion


def generality_check(template: str, rt: str,
                     neighbours: List[AutoRegion],
                     llm: LLMClient) -> _GeneralityResult:
    if not neighbours:
        return _GeneralityResult(verdict="NARROW", per_instance=[],
                                 summary="no neighbours to test")
    instance_blocks: list = []
    for i, nb in enumerate(neighbours, start=1):
        orig = nb.source.strip()
        try:
            rendered = render_template(template, nb)
        except Exception as e:
            rendered = f"<render error: {e}>"
        instance_blocks.append(
            f"=== Instance {i} ===\n"
            f"Original:\n```python\n{orig}\n```\n"
            f"Rendered (template applied):\n```\n{rendered}\n```"
        )
    prompt = _PROMPT_GENERALITY.format(
        rt=rt, template=template, n=len(neighbours),
        instances_block="\n\n".join(instance_blocks),
    )
    try:
        reply = llm.chat(prompt)
    except Exception as e:
        log.warning("generality LLM call failed: %s", e)
        return _GeneralityResult(verdict="NARROW", per_instance=[],
                                 summary=f"llm error: {e}")
    parsed = _parse_json(reply)
    if not parsed:
        return _GeneralityResult(verdict="NARROW", per_instance=[],
                                 summary="malformed critic reply")
    verdict = (parsed.get("verdict") or "NARROW").strip().upper()
    if verdict not in ("BROAD", "NARROW"):
        verdict = "NARROW"
    per_instance = parsed.get("per_instance") or []
    if not isinstance(per_instance, list):
        per_instance = []
    summary = (parsed.get("summary") or "").strip()
    return _GeneralityResult(verdict=verdict, per_instance=per_instance,
                             summary=summary)


def synth_refine(old_template: str, rt: str,
                 unsafe_pairs: List[tuple],
                 llm: LLMClient) -> Optional[_RefineResult]:
    if not unsafe_pairs:
        return None
    failure_lines: list = []
    for nb, info in unsafe_pairs[:5]:
        orig = nb.source.strip()
        try:
            rendered = render_template(old_template, nb)
        except Exception:
            rendered = "<render error>"
        why = info.get("why", "") if isinstance(info, dict) else ""
        failure_lines.append(
            f"  - Original:  {orig}\n"
            f"    Rendered:  {rendered}\n"
            f"    Why unsafe: {why}"
        )
    prompt = _PROMPT_REFINE.format(
        rt=rt, old_template=old_template,
        failures_block="\n".join(failure_lines),
        slot_menu=_slot_menu(rt),
    )
    try:
        reply = llm.chat(prompt)
    except Exception as e:
        log.warning("refine LLM call failed: %s", e)
        return None
    parsed = _parse_json(reply)
    if not parsed:
        return None
    if parsed.get("give_up"):
        return _RefineResult(
            template="", rationale="", gave_up=True,
            reason=(parsed.get("reason") or "").strip(),
        )
    template = (parsed.get("template") or "").strip()
    rationale = (parsed.get("rationale") or "").strip()
    if not template:
        return None
    err = _validate_template(template, rt)
    if err:
        log.info("refine: invalid template (%s)", err)
        return None
    return _RefineResult(template=template, rationale=rationale,
                         gave_up=False, reason="")


def synth_loop(
    bucket_instances: List[AutoRegion],
    rt: str,
    llm: LLMClient,
    neighbours: Optional[List[AutoRegion]] = None,
    max_rounds: int = 2,
    initial_feedback: str = "",
) -> Optional[LLMSynthResult]:
    """Design, check, and refine a template for a bucket using related examples."""
    if rt not in SLOT_VOCAB or not bucket_instances:
        return None

    sample = bucket_instances[:3]
    instances_block = "\n\n".join(
        f"=== Instance {i+1} ===\n```python\n{r.source.strip()}\n```"
        for i, r in enumerate(sample))
    slot_menu = _slot_menu(rt)

    # ---- Phase A: design + single-instance safety ----
    feedback = initial_feedback.strip()
    template = ""
    rationale = ""
    safety_ok = False
    safety_reason = ""
    rounds_used = 0
    for round_no in range(1, max_rounds + 1):
        rounds_used = round_no
        feedback_block = ""
        if feedback:
            feedback_block = f"FEEDBACK FROM PREVIOUS ROUND:\n{feedback}\n\n"
        prompt = _PROMPT_SYNTH.format(
            n=len(sample), rt=rt,
            instances_block=instances_block,
            slot_menu=slot_menu, feedback_block=feedback_block,
        )
        try:
            reply = llm.chat(prompt)
        except Exception as e:
            log.warning("synth LLM call failed: %s", e)
            return None
        parsed = _parse_json(reply)
        if not parsed or "template" not in parsed:
            log.info("round %d: malformed synth output", round_no)
            feedback = ("Previous reply was not valid JSON. Output ONLY a "
                        "JSON object with keys 'template' and 'rationale'.")
            continue
        new_template = (parsed.get("template") or "").strip()
        new_rationale = (parsed.get("rationale") or "").strip()
        err = _validate_template(new_template, rt)
        if err:
            log.info("round %d: invalid template (%s)", round_no, err)
            feedback = (f"Previous template was invalid: {err}. Use only "
                        f"slots from the menu and standard literals.")
            continue
        template = new_template
        rationale = new_rationale
        safety_ok, safety_reason, suggestion = _safety_check(
            template, sample[0], llm)
        if safety_ok:
            log.info("round %d: safety YES (%s)", round_no, safety_reason)
            break
        log.info("round %d: safety NO (%s)", round_no, safety_reason)
        feedback = (f"Critic rejected previous template. Reason: "
                    f"{safety_reason}. Suggestion: {suggestion}. Revise.")

    if not template:
        return None
    if not safety_ok:
        return LLMSynthResult(
            template=template, rationale=rationale, rounds=rounds_used,
            critic_verdict="NO",
            critic_reason=safety_reason or "exhausted rounds",
        )

    # ---- Phase B: generality ----
    if not neighbours:
        log.info("phase B SKIPPED: no cross-bucket neighbours for %s", rt)
        return LLMSynthResult(
            template=template, rationale=rationale, rounds=rounds_used,
            critic_verdict="YES", critic_reason=safety_reason,
            applies_broadly=False, refined=False,
            generality_verdict="SKIPPED", generality_reasons=[],
        )

    gen = generality_check(template, rt, neighbours, llm)
    if gen.verdict == "BROAD":
        log.info("phase B BROAD (%s)", gen.summary)
        return LLMSynthResult(
            template=template, rationale=rationale, rounds=rounds_used,
            critic_verdict="YES", critic_reason=safety_reason,
            applies_broadly=True, refined=False,
            generality_verdict="BROAD", generality_reasons=gen.per_instance,
        )
    log.info("phase B NARROW (%s) — attempting refine", gen.summary)

    # ---- Phase C: refine ----
    unsafe_pairs: list = []
    for info in gen.per_instance:
        if not isinstance(info, dict) or info.get("safe"):
            continue
        idx = info.get("i")
        if isinstance(idx, int) and 1 <= idx <= len(neighbours):
            unsafe_pairs.append((neighbours[idx - 1], info))
    refined = synth_refine(template, rt, unsafe_pairs, llm)

    fallback = LLMSynthResult(
        template=template, rationale=rationale, rounds=rounds_used,
        critic_verdict="YES", critic_reason=safety_reason,
        applies_broadly=False, refined=False,
        generality_verdict="NARROW", generality_reasons=gen.per_instance,
    )
    if refined is None or refined.gave_up:
        if refined and refined.gave_up:
            log.info("phase C: refine gave up (%s)", refined.reason)
        else:
            log.info("phase C: refine malformed")
        return fallback

    safety2_ok, safety2_reason, _ = _safety_check(refined.template, sample[0], llm)
    if not safety2_ok:
        log.info("phase C: refined template fails safety (%s) — fallback",
                 safety2_reason)
        return fallback
    gen2 = generality_check(refined.template, rt, neighbours, llm)
    if gen2.verdict == "BROAD":
        log.info("phase C: refine BROAD (%s)", gen2.summary)
        return LLMSynthResult(
            template=refined.template, rationale=refined.rationale,
            rounds=rounds_used, critic_verdict="YES",
            critic_reason=safety2_reason,
            applies_broadly=True, refined=True,
            generality_verdict="BROAD", generality_reasons=gen2.per_instance,
        )
    log.info("phase C: refine still NARROW (%s) — fallback to original",
             gen2.summary)
    return fallback
