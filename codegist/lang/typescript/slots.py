"""Extract slot values and render TypeScript templates."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...auto_mining.extract import AutoRegion
from .render import compact_render_ts
from .shadow import TSNode

_SLOT_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_KEEP_BUDGET = 70
_HOLE_MARK = "\x02TH{}\x02"
_NAME_KINDS = {"identifier", "property_identifier", "type_identifier"}
_MODIFIER_KINDS = ("accessibility_modifier",)        # public/private/protected
_MODIFIER_WORDS = ("export", "async", "static", "readonly", "abstract",
                   "public", "private", "protected", "get", "set")


def _compact(text: str) -> str:
    return compact_render_ts(text)


def _term(text: str) -> str:
    """Terminate a TypeScript statement before joining it into a chain."""
    t = text.rstrip()
    return t if t.endswith((";", "}", "{")) else t + ";"


def _stmts_of_body(node: TSNode) -> List[TSNode]:
    body = node.child_by_field("body")
    if body is None or body.kind != "statement_block":
        return []
    return list(body.children)


# ---------------------------------------------------------------- callees

def _callee_name(call: TSNode) -> str:
    if call.kind == "new_expression":
        c = call.child_by_field("constructor")
        return ("new " + _compact(c.text)) if c is not None else "new"
    fn = call.child_by_field("function")
    return _compact(fn.text) if fn is not None else ""


def _ordered_callees(node: TSNode) -> List[str]:
    out: List[str] = []
    for n in node.walk():
        if n.kind in ("call_expression", "new_expression"):
            nm = _callee_name(n)
            if nm and len(nm) < 45 and (not out or out[-1] != nm):
                out.append(nm)
    return out


def _stmt_ops_ts(stmt: TSNode, depth: int = 0) -> List[str]:
    if depth > 4:
        return [c + "()" for c in _ordered_callees(stmt)]
    text = _compact(stmt.text)
    if stmt.kind in ("return_statement", "throw_statement"):
        return [text if text.endswith(";") else text + ";"]
    if stmt.kind == "export_statement":
        # Exports carry the public dependency signature; reducing them to a bare
        # callee chain yields invalid ``export callee();``. Keep verbatim.
        return [_term(text)]
    if stmt.kind in ("lexical_declaration", "variable_declaration"):
        if len(text) <= _KEEP_BUDGET:
            return [_term(text)]
        callees = _ordered_callees(stmt)
        if not callees:
            return [_term(text)]
        decls = [c for c in stmt.children if c.kind == "variable_declarator"]
        if len(decls) == 1:
            val = decls[0].child_by_field("value")
            if val is not None:
                lhs = _compact(stmt._src[stmt.start:val.start])
                return [lhs + "+".join(c + "()" for c in callees) + ";"]
        return ["+".join(c + "()" for c in callees) + ";"]
    if stmt.kind == "expression_statement":
        if len(text) <= _KEEP_BUDGET:
            return [_term(text)]
        callees = _ordered_callees(stmt)
        return ["+".join(c + "()" for c in callees) + ";"] if callees else [_term(text)]
    if stmt.kind == "if_statement":
        cond = stmt.child_by_field("condition")
        out = ["if" + (_compact(cond.text) if cond is not None else "(?)") + "{"]
        cons = stmt.child_by_field("consequence")
        for s in (cons.children if cons is not None and cons.kind == "statement_block"
                  else ([cons] if cons is not None else [])):
            out += _stmt_ops_ts(s, depth + 1)
        out.append("}")
        alt = stmt.child_by_field("alternative")
        if alt is not None:
            # Unwrap the else clause before rendering its block or nested if.
            inner = alt
            if alt.kind == "else_clause":
                kids = list(alt.children)
                inner = kids[0] if kids else None
            if inner is not None and inner.kind == "if_statement":
                out.append("else")
                out += _stmt_ops_ts(inner, depth + 1)
            elif inner is not None:
                out.append("else{")
                for s in (inner.children if inner.kind == "statement_block"
                          else [inner]):
                    out += _stmt_ops_ts(s, depth + 1)
                out.append("}")
        return out
    if stmt.kind in ("for_statement", "for_in_statement", "while_statement",
                     "do_statement"):
        body = stmt.child_by_field("body")
        head_end = body.start if body is not None else stmt.end
        out = [_compact(stmt._src[stmt.start:head_end]) + "{"]
        for s in (body.children if body is not None and body.kind == "statement_block"
                  else ([body] if body is not None else [])):
            out += _stmt_ops_ts(s, depth + 1)
        out.append("}")
        return out
    if stmt.kind == "try_statement":
        out = ["try{"]
        body = stmt.child_by_field("body")
        if body is not None:
            for s in body.children:
                out += _stmt_ops_ts(s, depth + 1)
        for c in stmt.children:
            if c.kind == "catch_clause":
                param = c.child_by_field("parameter")
                out.append("}catch(" + (_compact(param.text) if param else "") + "){")
                cb = c.child_by_field("body")
                if cb is not None:
                    for s in cb.children:
                        out += _stmt_ops_ts(s, depth + 1)
            elif c.kind == "finally_clause":
                out.append("}finally{")
                fb = next((x for x in c.children if x.kind == "statement_block"), None)
                if fb is not None:
                    for s in fb.children:
                        out += _stmt_ops_ts(s, depth + 1)
        out.append("}")
        return out
    if len(text) <= _KEEP_BUDGET:
        return [_term(text)]
    callees = _ordered_callees(stmt)
    return [c + "();" for c in callees] if callees else [_term(text)]


def _body_chain_ts(stmts: List[TSNode], max_ops: int = 32) -> str:
    parts: List[str] = []
    for s in stmts:
        parts.extend(_stmt_ops_ts(s))
    if len(parts) > max_ops:
        callees = []
        for s in stmts:
            callees.extend(_ordered_callees(s))
        return " ".join(c + "();" for c in callees) or " ".join(parts[:max_ops])
    out: List[str] = []
    for p in parts:
        if out and out[-1].endswith("{"):
            out.append(p)
        elif p.startswith("}"):
            out.append(p)
        else:
            out.append((" " if out else "") + p)
    return "".join(out)


# ---------------------------------------------------------------- slots

SlotFn = Callable[[AutoRegion], str]


def _node(r: AutoRegion) -> TSNode:
    return r.node  # type: ignore[return-value]


def _slot_src(r: AutoRegion) -> str:
    return r.source


def _slot_modifiers(r: AutoRegion) -> str:
    node = _node(r)
    mods: List[str] = []
    # accessibility/keyword modifiers are unnamed children before the name
    for c in node.children:
        if c.kind in _MODIFIER_KINDS:
            mods.append(c.text)
    # keyword modifiers ('async'/'static'/'export') often appear in text head
    head = node.text[:40]
    for w in _MODIFIER_WORDS:
        if re.match(rf"(.*\s)?{w}\b", head) and w not in " ".join(mods):
            if re.match(rf"^\s*({'|'.join(_MODIFIER_WORDS)})\b", node.text) and w in node.text.split("{")[0]:
                if w not in mods:
                    mods.append(w)
    # de-dup, preserve order
    seen = []
    for m in mods:
        if m not in seen:
            seen.append(m)
    return " ".join(seen)


def _slot_func_name(r: AutoRegion) -> str:
    node = _node(r)
    n = node.child_by_field("name")
    if n is not None:
        return n.text
    # arrow bound to a const: name lives in the parent variable_declarator
    p = node.parent
    if p is not None and p.kind == "variable_declarator":
        nm = p.child_by_field("name")
        if nm is not None:
            return nm.text
    return ""


def _slot_ret_type(r: AutoRegion) -> str:
    n = _node(r).child_by_field("return_type")
    if n is None:
        return ""
    t = _compact(n.text)
    return t[1:].strip() if t.startswith(":") else t


def _slot_sig(r: AutoRegion) -> str:
    node = _node(r)
    params = node.child_by_field("parameters")
    sig = _compact(params.text) if params is not None else "()"
    rt = node.child_by_field("return_type")
    if rt is not None:
        sig += _compact(rt.text)        # type_annotation text starts with ':'
    return sig


def _slot_body_chain(r: AutoRegion) -> str:
    return _body_chain_ts(_stmts_of_body(_node(r)))


def _slot_body_blocks(r: AutoRegion) -> str:
    return " ".join(_term(_compact(s.text)) for s in _stmts_of_body(_node(r)))


def _slot_stmt_chain(r: AutoRegion) -> str:
    node = _node(r)
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        stmts = list(r.parent_node.children)[r.first_idx:r.last_idx + 1]
        return _body_chain_ts(stmts)
    return _body_chain_ts([node])


def _slot_cond(r: AutoRegion) -> str:
    n = _node(r).child_by_field("condition")
    return _compact(n.text) if n is not None else ""


def _slot_class_name(r: AutoRegion) -> str:
    n = _node(r).child_by_field("name")
    return n.text if n is not None else ""


def _slot_bases(r: AutoRegion) -> str:
    node = _node(r)
    out: List[str] = []
    for c in node.children:
        if c.kind in ("class_heritage", "extends_clause", "implements_clause"):
            out.append(_compact(c.text).replace("extends", "")
                       .replace("implements", "").strip())
    return ",".join(x for x in out if x)


def _class_body(r: AutoRegion) -> Optional[TSNode]:
    return _node(r).child_by_field("body")


def _slot_init_fields(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    names: List[str] = []
    for m in body.children:
        if m.kind != "method_definition":
            continue
        nm = m.child_by_field("name")
        if nm is None or nm.text != "constructor":
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
            if left is not None and left.text.startswith("this."):
                names.append(left.text[len("this."):])
    return ",".join(names)


def _slot_class_vars(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    names: List[str] = []
    for m in body.children:
        if m.kind in ("public_field_definition", "property_signature"):
            nm = m.child_by_field("name")
            if nm is not None:
                names.append(nm.text)
    return ",".join(names)


def _slot_method_sigs(r: AutoRegion) -> str:
    body = _class_body(r)
    if body is None:
        return ""
    sigs: List[str] = []
    for m in body.children:
        # Keep all interface and type members in the compact sketch.
        txt = _compact(m.text).rstrip(";,").strip()
        if txt:
            sigs.append(txt)
    return ";".join(sigs)


def _slot_call_target(r: AutoRegion) -> str:
    return _callee_name(_node(r))


def _call_args(r: AutoRegion) -> List[TSNode]:
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


SLOT_VOCAB_TS: Dict[str, Dict[str, SlotFn]] = {
    "function": {
        "{src}": _slot_src,
        "{modifiers}": _slot_modifiers,
        "{func_name}": _slot_func_name,
        "{ret_type}": _slot_ret_type,
        "{sig}": _slot_sig,
        "{body_chain}": _slot_body_chain,
        "{body_blocks}": _slot_body_blocks,
    },
    "class": {
        "{src}": _slot_src,
        "{modifiers}": _slot_modifiers,
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
    "data_flow": {"{src}": _slot_src, "{stmt_chain}": _slot_stmt_chain},
    "api_idiom": {
        "{call_target}": _slot_call_target,
        "{call_args_combined}": _slot_call_args_combined,
        "{first_arg}": _slot_first_arg,
        "{rest_args}": _slot_rest_args,
    },
    "expression": {
        "{var}": _slot_var, "{op_sym}": _slot_op_sym, "{lit}": _slot_lit,
    },
    "module": {},
}

_MULTI_HOLE_SLOTS = {"{body_blocks}"}


def validate_template_ts(template: str, region_type: str) -> Optional[str]:
    if not template.strip():
        return "empty template"
    vocab = SLOT_VOCAB_TS.get(region_type, {})
    slots = set(_SLOT_RE.findall(template))
    unknown = [s for s in slots if s not in vocab]
    if unknown:
        return f"unknown slot(s) {unknown} for {region_type}"
    if not slots:
        return "no slots — template is a verbatim sample, not a schema"
    bare = _SLOT_RE.sub(" ", template)
    badc = sorted({c for c in bare if ord(c) > 127})
    if badc:
        return f"non-ASCII markup {badc}"
    # '=>' / '->' may NOT be used as connectors in the literal skeleton.
    # (Source arrow functions live INSIDE slot values, never in the skeleton.)
    for marker in ("=>", "->"):
        if marker in bare:
            return f"chain marker {marker!r} not allowed in TS skeleton"
    return None


@dataclass
class RenderedTS:
    region: AutoRegion
    template: str
    skeleton: str = ""
    marks: List[str] = field(default_factory=list)
    inner_nodes: List[TSNode] = field(default_factory=list)
    _default: str = ""

    def materialize_default(self) -> str:
        return self._default


@dataclass
class TSTemplateTransform:
    template: str
    region_type: str
    name: str

    def matches(self, _skeleton, region: AutoRegion) -> bool:
        if region.region_type != self.region_type:
            return False
        if not isinstance(region.node, TSNode):
            return False
        vocab = SLOT_VOCAB_TS.get(self.region_type, {})
        optional = {"{rest_args}", "{call_args_combined}", "{bases}",
                    "{modifiers}", "{init_fields}", "{class_vars}"}
        for slot in set(_SLOT_RE.findall(self.template)):
            fn = vocab.get(slot)
            if fn is None:
                return False
            try:
                if not fn(region).strip() and slot not in optional:
                    return False
            except Exception:
                return False
        return True

    def render(self, region: AutoRegion) -> RenderedTS:
        vocab = SLOT_VOCAB_TS.get(self.region_type, {})
        out = self.template
        marks: List[str] = []
        inner: List[TSNode] = []
        if "{body_blocks}" in out and isinstance(region.node, TSNode):
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
        return RenderedTS(region=region, template=self.template, skeleton=out,
                          marks=marks, inner_nodes=inner, _default=default)
