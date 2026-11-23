"""Extract typed regions from TypeScript source."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from ...auto_mining.extract import AutoRegion
from .shadow import TSNode, parse_ts

_CLASS_KINDS = {"class_declaration", "interface_declaration",
                "abstract_class_declaration"}
_FUNC_KINDS = {"function_declaration", "method_definition",
               "function_signature", "generator_function_declaration"}
_CF_KINDS = {"if_statement", "for_statement", "for_in_statement",
             "while_statement", "do_statement", "try_statement",
             "switch_statement"}
_CALL_KINDS = {"call_expression", "new_expression"}
_EXPR_KINDS = {"binary_expression", "unary_expression", "ternary_expression"}

_WINDOW_SIZES = (2, 3, 4, 5, 6, 7, 8)
_MAX_WINDOW_BODY = 80


def _lineno(src: str, off: int) -> int:
    return src.count("\n", 0, off) + 1


def _region(rt: str, node: TSNode, src: str, file: str, **kw) -> AutoRegion:
    return AutoRegion(region_type=rt, node=node, source=node.text, file=file,
                      lineno=_lineno(src, node.start), **kw)


def _is_assignment_stmt(node: TSNode) -> bool:
    if node.kind != "expression_statement" or not node.children:
        return False
    return node.children[0].kind in ("assignment_expression",
                                     "augmented_assignment_expression",
                                     "update_expression")


def _is_const_arrow(node: TSNode) -> bool:
    if node.kind != "lexical_declaration":
        return False
    for d in node.children:
        if d.kind == "variable_declarator":
            v = d.child_by_field("value")
            if v is not None and v.kind == "arrow_function":
                return True
    return False


def _data_flow_node_kind(node: TSNode) -> bool:
    return (node.kind in ("lexical_declaration", "variable_declaration",
                          "return_statement")
            or _is_assignment_stmt(node)) and not _is_const_arrow(node)


def _extract_import_prefix(root: TSNode, src: str, file: str) -> Optional[AutoRegion]:
    imports: List[TSNode] = []
    for c in root.children:
        if c.kind == "import_statement":
            imports.append(c)
        elif imports:
            break
    if len(imports) < 2:
        return None
    seg = src[imports[0].start:imports[-1].end]
    return AutoRegion(
        region_type="module", node=root, source=seg, file=file,
        lineno=_lineno(src, imports[0].start),
        span_kind="stmt_range", parent_node=root, body_attr="stmts",
        first_idx=root.children.index(imports[0]),
        last_idx=root.children.index(imports[-1]),
        extra={"kind": "import_prefix", "count": len(imports)},
    )


def _extract_windows(root: TSNode, src: str, file: str) -> Iterator[AutoRegion]:
    for node in root.walk():
        for block, attr, stmts in node.stmt_blocks():
            n = len(stmts)
            if n > _MAX_WINDOW_BODY:
                continue
            for size in _WINDOW_SIZES:
                if size > n:
                    break
                for i in range(0, n - size + 1):
                    first, last = stmts[i], stmts[i + size - 1]
                    seg = src[first.start:last.end]
                    if not seg.strip():
                        continue
                    yield AutoRegion(
                        region_type="data_flow", node=first, source=seg,
                        file=file, lineno=_lineno(src, first.start),
                        span_kind="stmt_range", parent_node=block, body_attr=attr,
                        first_idx=i, last_idx=i + size - 1,
                        extra={"window_size": size},
                    )


def extract_from_tree_ts(root: TSNode, src: str, file: str) -> List[AutoRegion]:
    regions: List[AutoRegion] = []
    regions.append(_region("module", root, src, file,
                           extra={"kind": "whole_module"}))
    mi = _extract_import_prefix(root, src, file)
    if mi is not None:
        regions.append(mi)

    for node in root.walk():
        if node is root:
            continue
        rt: Optional[str] = None
        if node.kind in _CLASS_KINDS:
            rt = "class"
        elif node.kind in _FUNC_KINDS:
            rt = "function"
        elif _is_const_arrow(node):
            rt = "function"
        elif node.kind in _CF_KINDS:
            rt = "control_flow"
        elif node.kind in _CALL_KINDS:
            rt = "api_idiom"
        elif node.kind in _EXPR_KINDS:
            rt = "expression"
        elif _data_flow_node_kind(node):
            rt = "data_flow"
        if rt is None:
            continue
        if not node.text.strip():
            continue
        regions.append(_region(rt, node, src, file))

    regions.extend(_extract_windows(root, src, file))
    return regions


def extract_from_source_ts(src: str, file: str) -> List[AutoRegion]:
    tsx = str(file).endswith(".tsx")
    try:
        root = parse_ts(src, tsx=tsx)
    except SyntaxError:
        return []
    return extract_from_tree_ts(root, src, file)


def iter_corpus_ts(corpus_dir: Path,
                   exclude: list | None = None) -> Iterator[AutoRegion]:
    import logging
    from ..._excludes import is_excluded
    log = logging.getLogger("codegist.lang.typescript.extract")
    excludes = list(exclude) if exclude else []
    n_files = n_skipped = 0
    paths = sorted(list(corpus_dir.rglob("*.ts")) + list(corpus_dir.rglob("*.tsx")))
    for path in paths:
        # skip declaration files — pure type signatures, no compressible body
        if path.name.endswith(".d.ts"):
            continue
        if excludes and is_excluded(path, excludes):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            n_skipped += 1
            continue
        try:
            regions = extract_from_source_ts(src, str(path))
        except (RecursionError, MemoryError, Exception) as e:  # noqa: BLE001
            n_skipped += 1
            log.warning("skip %s: %s: %s", path, type(e).__name__, e)
            continue
        n_files += 1
        if n_files % 5000 == 0:
            log.info("iter_corpus_ts: %d files parsed (%d skipped)",
                     n_files, n_skipped)
        yield from regions
