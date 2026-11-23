"""Extract slot values and render Java templates."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...auto_mining.extract import AutoRegion
from .render import compact_render_java
from .shadow import JNode

_SLOT_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_KEEP_BUDGET = 60   # stmt kept verbatim-compact when at most this many chars

_HOLE_MARK = "\x02JH{}\x02"


def _compact(text: str) -> str:
    return compact_render_java(text)


def _stmts_of_body(node: JNode) -> List[JNode]:
    body = node.child_by_field("body")
    if body is None:
        return []
    return list(body.children)


# ---------------------------------------------------------------- stmt ops

_CALLEE_KINDS = {"method_invocation", "object_creation_expression"}


def _callee_name(call: JNode) -> str:
    if call.kind == "object_creation_expression":
        t = call.child_by_field("type")
        return ("new " + _compact(t.text)) if t is not None else "new"
    name = call.child_by_field("name")
    obj = call.child_by_field("object")
    if name is None:
        return ""
    if obj is not None and obj.kind in ("identifier", "field_access"):
        return _compact(obj.text) + "." + name.text
    return name.text


def _ordered_callees(node: JNode) -> List[str]:
    out: List[str] = []
    for n in node.walk():
        if n.kind in _CALLEE_KINDS:
            nm = _callee_name(n)
            if nm and len(nm) < 40 and (not out or out[-1] != nm):
                out.append(nm)
    return out


def _stmt_ops_java(stmt: JNode, depth: int = 0) -> List[str]:
    if depth > 4:
        return [c + "()" for c in _ordered_callees(stmt)]
    text = _compact(stmt.text)
    if stmt.kind in ("return_statement", "throw_statement"):
        return [text if text.endswith(";") else text + ";"]
    if stmt.kind == "local_variable_declaration" or stmt.kind == "expression_statement":
        if len(text) <= _KEEP_BUDGET:
            return [text]
        callees = _ordered_callees(stmt)
        if not callees:
            return [text]
        decl = stmt.children[0] if stmt.children else None
        target = ""
        if stmt.kind == "local_variable_declaration":
            for c in stmt.children:
                if c.kind == "variable_declarator":
                    nm = c.child_by_field("name")
                    if nm is not None:
                        target = nm.text
                    break
        elif decl is not None and decl.kind == "assignment_expression":
            left = decl.child_by_field("left")
            if left is not None:
                target = _compact(left.text)
        chain = "+".join(c + "()" for c in callees) if len(callees) > 1 \
            else callees[0] + "()"
        return [(target + "=" + chain + ";") if target else chain + ";"]
    if stmt.kind == "if_statement":
        cond = stmt.child_by_field("condition")
        head = "if" + (_compact(cond.text) if cond is not None else "(?)")
        out = [head + "{"]
        cons = stmt.child_by_field("consequence")
        for s in (cons.children if cons is not None and cons.kind == "block"
                  else ([cons] if cons is not None else [])):
            out += _stmt_ops_java(s, depth + 1)
        out.append("}")
        alt = stmt.child_by_field("alternative")
        if alt is not None:
            out.append("else{")
            for s in (alt.children if alt.kind == "block" else [alt]):
                out += _stmt_ops_java(s, depth + 1)
            out.append("}")
        return out
    if stmt.kind in ("for_statement", "enhanced_for_statement",
                     "while_statement", "do_statement"):
        body = stmt.child_by_field("body")
        head_end = body.start if body is not None else stmt.end
        head = _compact(stmt._src[stmt.start:head_end])
        out = [head + "{"]
        for s in (body.children if body is not None and body.kind == "block"
                  else ([body] if body is not None else [])):
            out += _stmt_ops_java(s, depth + 1)
        out.append("}")
        return out
    if stmt.kind in ("try_statement", "try_with_resources_statement"):
        out = ["try{"]
        body = stmt.child_by_field("body")
        if body is not None:
            for s in body.children:
                out += _stmt_ops_java(s, depth + 1)
        for c in stmt.children:
            if c.kind == "catch_clause":
                param = next((x for x in c.walk()
                              if x.kind == "catch_formal_parameter"), None)
                out.append("}catch(" + (_compact(param.text) if param else "?") + "){")
                cb = c.child_by_field("body")
                if cb is not None:
                    for s in cb.children:
                        out += _stmt_ops_java(s, depth + 1)
            elif c.kind == "finally_clause":
                out.append("}finally{")
                fb = next((x for x in c.children if x.kind == "block"), None)
                if fb is not None:
                    for s in fb.children:
                        out += _stmt_ops_java(s, depth + 1)
        out.append("}")
        return out
    if len(text) <= _KEEP_BUDGET:
        return [text]
    callees = _ordered_callees(stmt)
    return [c + "();" for c in callees] if callees else [text]


def _body_chain_java(stmts: List[JNode], max_ops: int = 32) -> str:
    parts: List[str] = []
    for s in stmts:
        parts.extend(_stmt_ops_java(s))
    if len(parts) > max_ops:
        callees = []
        for s in stmts:
            callees.extend(_ordered_callees(s))
        return " ".join(c + "();" for c in callees) or " ".join(parts[:max_ops])
    out: List[str] = []
    for p in parts:
        if out and out[-1].endswith("{"):
            out.append(p)
        elif p in ("}", "}catch", "}finally") or p.startswith("}"):
            out.append(p)
        else:
            out.append((" " if out else "") + p)
    return "".join(out)


# ---------------------------------------------------------------- slots

SlotFn = Callable[[AutoRegion], str]


def _node(r: AutoRegion) -> JNode:
    return r.node  # type: ignore[return-value]


def _slot_src(r: AutoRegion) -> str:
    return r.source


def _slot_func_name(r: AutoRegion) -> str:
    n = _node(r).child_by_field("name")
    return n.text if n is not None else ""


def _slot_ret_type(r: AutoRegion) -> str:
    n = _node(r).child_by_field("type")
    return _compact(n.text) if n is not None else ""


def _slot_sig(r: AutoRegion) -> str:
    node = _node(r)
    params = node.child_by_field("parameters")
    sig = _compact(params.text) if params is not None else "()"
    for c in node.children:
        if c.kind == "throws":
            sig += " " + _compact(c.text)
    return sig


def _slot_body_chain(r: AutoRegion) -> str:
    return _body_chain_java(_stmts_of_body(_node(r)))


def _slot_body_blocks(r: AutoRegion) -> str:
    parts = []
    for s in _stmts_of_body(_node(r)):
        parts.append(_compact(s.text))
    return " ".join(parts)


def _slot_stmt_chain(r: AutoRegion) -> str:
    node = _node(r)
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        stmts = list(r.parent_node.children)[r.first_idx:r.last_idx + 1]
        return _body_chain_java(stmts)
    return _body_chain_java([node])


def _slot_cond(r: AutoRegion) -> str:
    n = _node(r).child_by_field("condition")
    return _compact(n.text) if n is not None else ""


def _slot_class_name(r: AutoRegion) -> str:
    n = _node(r).child_by_field("name")
    return n.text if n is not None else ""


def _slot_bases(r: AutoRegion) -> str:
    node = _node(r)
    out: List[str] = []
    sc = node.child_by_field("superclass")
    if sc is not None:
        out.append(_compact(sc.text).replace("extends", "").strip())
    for c in node.children:
        if c.kind in ("super_interfaces", "interfaces"):
            out.append(_compact(c.text).replace("implements", "").strip())
    return ",".join(x for x in out if x)


def _class_body(r: AutoRegion) -> Optional[JNode]:
    return _node(r).child_by_field("body")


def _slot_init_fields(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    names: List[str] = []
    for m in body.children:
        if m.kind != "constructor_declaration":
            continue
        cb = m.child_by_field("body")
        if cb is None:
            continue
        for s in cb.children:
            if s.kind != "expression_statement" or not s.children:
                continue
            a = s.children[0]
            if a.kind != "assignment_expression":
                continue
            left = a.child_by_field("left")
            if left is not None and left.kind == "field_access" \
                    and left.text.startswith("this."):
                names.append(left.text[len("this."):])
    return ",".join(names)


def _slot_class_vars(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    names: List[str] = []
    for m in body.children:
        if m.kind == "field_declaration":
            for c in m.children:
                if c.kind == "variable_declarator":
                    nm = c.child_by_field("name")
                    if nm is not None:
                        names.append(nm.text)
    return ",".join(names)


def _slot_method_sigs(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    sigs: List[str] = []
    for m in body.children:
        if m.kind != "method_declaration":
            continue
        nm = m.child_by_field("name")
        tp = m.child_by_field("type")
        pr = m.child_by_field("parameters")
        sigs.append(((_compact(tp.text) + " ") if tp is not None else "")
                    + (nm.text if nm is not None else "?")
                    + (_compact(pr.text) if pr is not None else "()"))
    return ";".join(sigs)


def _slot_call_target(r: AutoRegion) -> str:
    return _callee_name(_node(r))


def _call_args(r: AutoRegion) -> List[JNode]:
    args = _node(r).child_by_field("arguments")
    return list(args.children) if args is not None else []


def _slot_call_args_combined(r: AutoRegion) -> str:
    return ",".join(_compact(a.text) for a in _call_args(r))


def _slot_first_arg(r: AutoRegion) -> str:
    args = _call_args(r)
    return _compact(args[0].text) if args else ""


def _slot_rest_args(r: AutoRegion) -> str:
    args = _call_args(r)
    return ",".join(_compact(a.text) for a in args[1:])


def _slot_var(r: AutoRegion) -> str:
    n = _node(r).child_by_field("left")
    return _compact(n.text) if n is not None else ""


def _slot_op_sym(r: AutoRegion) -> str:
    return _node(r).op


def _slot_lit(r: AutoRegion) -> str:
    n = _node(r).child_by_field("right")
    return _compact(n.text) if n is not None else ""


SLOT_VOCAB_JAVA: Dict[str, Dict[str, SlotFn]] = {
    "function": {
        "{src}": _slot_src,
        "{func_name}": _slot_func_name,
        "{ret_type}": _slot_ret_type,
        "{sig}": _slot_sig,
        "{body_chain}": _slot_body_chain,
        "{body_blocks}": _slot_body_blocks,
    },
    "class": {
        "{src}": _slot_src,
        "{class_name}": _slot_class_name,
        "{bases}": _slot_bases,
        "{init_fields}": _slot_init_fields,
        "{class_vars}": _slot_class_vars,
        "{method_sigs}": _slot_method_sigs,
    },
    "control_flow": {
        "{src}": _slot_src,
        "{cond}": _slot_cond,
        "{body_chain}": _slot_body_chain,
        "{stmt_chain}": _slot_stmt_chain,
    },
    "data_flow": {
        "{src}": _slot_src,
        "{stmt_chain}": _slot_stmt_chain,
    },
    "api_idiom": {
        "{call_target}": _slot_call_target,
        "{call_args_combined}": _slot_call_args_combined,
        "{first_arg}": _slot_first_arg,
        "{rest_args}": _slot_rest_args,
    },
    "expression": {
        "{var}": _slot_var,
        "{op_sym}": _slot_op_sym,
        "{lit}": _slot_lit,
    },
    "module": {},
}

_MULTI_HOLE_SLOTS = {"{body_blocks}"}


def validate_template_java(template: str, region_type: str) -> Optional[str]:
    """Return a template validation error, or None for a valid template."""
    if not template.strip():
        return "empty template"
    vocab = SLOT_VOCAB_JAVA.get(region_type, {})
    slots = set(_SLOT_RE.findall(template))
    unknown = [s for s in slots if s not in vocab]
    if unknown:
        return f"unknown slot(s) {unknown} for {region_type}"
    if not slots:
        return "no slots — template is a verbatim sample, not a schema"
    bare = _SLOT_RE.sub(" ", template)
    bad = sorted({c for c in bare if ord(c) > 127})
    if bad:
        return f"non-ASCII markup {bad}"
    for marker in ("=>", "->"):
        if marker in bare:
            return f"chain marker {marker!r} not allowed in Java skeleton"
    return None


@dataclass
class RenderedJava:
    region: AutoRegion
    template: str
    vocab: Dict[str, SlotFn]
    skeleton: str = ""
    marks: List[str] = field(default_factory=list)
    inner_nodes: List[JNode] = field(default_factory=list)
    _default: str = ""

    def materialize_default(self) -> str:
        return self._default


@dataclass
class JavaTemplateTransform:
    template: str
    region_type: str
    name: str

    def matches(self, _skeleton, region: AutoRegion) -> bool:
        if region.region_type != self.region_type:
            return False
        if not isinstance(region.node, JNode):
            return False
        vocab = SLOT_VOCAB_JAVA.get(self.region_type, {})
        for slot in set(_SLOT_RE.findall(self.template)):
            fn = vocab.get(slot)
            if fn is None:
                return False
            try:
                if not fn(region).strip() and slot not in ("{rest_args}",
                                                           "{call_args_combined}",
                                                           "{bases}"):
                    return False
            except Exception:
                return False
        return True

    def render(self, region: AutoRegion) -> RenderedJava:
        vocab = SLOT_VOCAB_JAVA.get(self.region_type, {})
        out = self.template
        marks: List[str] = []
        inner: List[JNode] = []
        if "{body_blocks}" in out and isinstance(region.node, JNode):
            stmts = _stmts_of_body(region.node)
            hole_marks: List[str] = []
            for i, s in enumerate(stmts):
                m = _HOLE_MARK.format(i)
                hole_marks.append(m)
                marks.append(m)
                inner.append(s)
            out = out.replace("{body_blocks}", " ".join(hole_marks), 1)
        for slot, fn in vocab.items():
            if slot in out:
                try:
                    out = out.replace(slot, fn(region))
                except Exception:
                    out = out.replace(slot, "")
        default = out
        for m, node in zip(marks, inner):
            default = default.replace(m, _compact(node.text), 1)
        return RenderedJava(region=region, template=self.template,
                            vocab=vocab, skeleton=out, marks=marks,
                            inner_nodes=inner, _default=default)
