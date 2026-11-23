"""Wrap TypeScript parse nodes with stable identities and source spans."""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple

_COMMENT_KINDS = {"comment"}

# Only true statement blocks — see module docstring + the Java lesson.
BLOCK_KINDS = {"statement_block"}


@dataclass
class TSNode:
    kind: str
    start: int                      # char offset into the decoded source
    end: int                        # char offset (exclusive)
    field: Optional[str] = None     # field name in the parent, if any
    op: str = ""                    # operator token text ("==", "+", ...)
    children: List["TSNode"] = dc_field(default_factory=list)
    parent: Optional["TSNode"] = None
    _src: str = ""                  # whole-file source (shared reference)

    @property
    def text(self) -> str:
        return self._src[self.start:self.end]

    def child_by_field(self, name: str) -> Optional["TSNode"]:
        for c in self.children:
            if c.field == name:
                return c
        return None

    def children_by_field(self, name: str) -> List["TSNode"]:
        return [c for c in self.children if c.field == name]

    def stmt_blocks(self) -> List[Tuple["TSNode", str, List["TSNode"]]]:
        out: List[Tuple["TSNode", str, List[TSNode]]] = []
        if self.kind in BLOCK_KINDS and self.children:
            out.append((self, "stmts", list(self.children)))
        return out

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def __repr__(self) -> str:
        return f"TSNode({self.kind}, {self.start}:{self.end})"


def _byte_to_char_map(data: bytes):
    """btoc[i] = char index of byte i; None for pure ASCII (identity)."""
    text = data.decode("utf-8", errors="replace")
    if len(text) == len(data):
        return None
    btoc = [0] * (len(data) + 1)
    ci = 0
    bi = 0
    for ch in text:
        o = ord(ch)
        blen = 1 if o < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4
        for k in range(blen):
            if bi + k <= len(data) - 1:
                btoc[bi + k] = ci
        bi += blen
        ci += 1
    btoc[len(data)] = ci
    return btoc


_PARSERS: dict = {}


def _parser(tsx: bool = False):
    key = "tsx" if tsx else "ts"
    if key not in _PARSERS:
        import tree_sitter_typescript as tst
        from tree_sitter import Language, Parser
        raw = tst.language_tsx() if tsx else tst.language_typescript()
        lang = Language(raw)
        try:
            _PARSERS[key] = Parser(lang)
        except TypeError:
            p = Parser()
            p.set_language(lang)
            _PARSERS[key] = p
    return _PARSERS[key]


def _convert(ts_node, src: str, btoc,
             parent: Optional[TSNode], fname: Optional[str]) -> Optional[TSNode]:
    if ts_node.type in _COMMENT_KINDS:
        return None
    start = ts_node.start_byte if btoc is None else btoc[ts_node.start_byte]
    end = ts_node.end_byte if btoc is None else btoc[ts_node.end_byte]
    node = TSNode(kind=ts_node.type, start=start, end=end,
                  field=fname, parent=parent, _src=src)
    op_child = ts_node.child_by_field_name("operator")
    if op_child is not None:
        node.op = op_child.text.decode("utf-8", errors="replace")
    for i, ch in enumerate(ts_node.children):
        if not ch.is_named:
            continue
        cf = ts_node.field_name_for_child(i)
        conv = _convert(ch, src, btoc, node, cf)
        if conv is not None:
            node.children.append(conv)
    return node


def parse_ts(src: str, tsx: bool = False) -> TSNode:
    """Parse TS/TSX source into a TSNode shadow tree (root kind ``program``)."""
    data = src.encode("utf-8", errors="replace")
    tree = _parser(tsx).parse(data)
    root = tree.root_node
    if root.type == "ERROR":
        raise SyntaxError("tree-sitter could not parse file")
    btoc = _byte_to_char_map(data)
    shadow = _convert(root, src, btoc, None, None)
    if shadow is None:
        raise SyntaxError("empty parse")
    return shadow


def has_errors(node: TSNode) -> bool:
    return any(n.kind == "ERROR" for n in node.walk())
