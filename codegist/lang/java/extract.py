"""Extract typed regions from Java source."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from ...auto_mining.extract import AutoRegion
from .shadow import JNode, parse_java

_CLASS_KINDS = {"class_declaration", "interface_declaration",
                "enum_declaration", "record_declaration"}
_FUNC_KINDS = {"method_declaration", "constructor_declaration"}
_CF_KINDS = {"if_statement", "for_statement", "enhanced_for_statement",
             "while_statement", "do_statement", "try_statement",
             "try_with_resources_statement"}
_CALL_KINDS = {"method_invocation", "object_creation_expression"}
_EXPR_KINDS = {"binary_expression", "unary_expression",
               "ternary_expression", "instanceof_expression"}

_WINDOW_SIZES = (2, 3, 4, 5, 6, 7, 8)
_MAX_WINDOW_BODY = 80


def _lineno(src: str, off: int) -> int:
    return src.count("\n", 0, off) + 1


def _region(rt: str, node: JNode, src: str, file: str, **kw) -> AutoRegion:
    return AutoRegion(
        region_type=rt, node=node, source=node.text, file=file,
        lineno=_lineno(src, node.start), **kw,
    )


def _is_assignment_stmt(node: JNode) -> bool:
    if node.kind != "expression_statement" or not node.children:
        return False
    return node.children[0].kind in ("assignment_expression",
                                     "update_expression")


def _data_flow_node_kind(node: JNode) -> bool:
    return (node.kind in ("local_variable_declaration", "return_statement")
            or _is_assignment_stmt(node))


def _extract_import_prefix(root: JNode, src: str, file: str) -> Optional[AutoRegion]:
    imports: List[JNode] = []
    for c in root.children:
        if c.kind == "package_declaration" and not imports:
            continue
        if c.kind == "import_declaration":
            imports.append(c)
        elif imports:
            break
        else:
            break
    if len(imports) < 2:
        return None
    seg = src[imports[0].start:imports[-1].end]
    region = AutoRegion(
        region_type="module", node=root, source=seg, file=file,
        lineno=_lineno(src, imports[0].start),
        span_kind="stmt_range", parent_node=root, body_attr="stmts",
        first_idx=root.children.index(imports[0]),
        last_idx=root.children.index(imports[-1]),
        extra={"kind": "import_prefix", "count": len(imports)},
    )
    return region


def _extract_windows(root: JNode, src: str, file: str) -> Iterator[AutoRegion]:
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
                        region_type="data_flow",
                        node=first, source=seg, file=file,
                        lineno=_lineno(src, first.start),
                        span_kind="stmt_range",
                        parent_node=block, body_attr=attr,
                        first_idx=i, last_idx=i + size - 1,
                        extra={"window_size": size},
                    )


def extract_from_tree_java(root: JNode, src: str, file: str) -> List[AutoRegion]:
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


def extract_from_source_java(src: str, file: str) -> List[AutoRegion]:
    try:
        root = parse_java(src)
    except SyntaxError:
        return []
    return extract_from_tree_java(root, src, file)


def iter_corpus_java(corpus_dir: Path,
                     exclude: list | None = None) -> Iterator[AutoRegion]:
    import logging
    from ..._excludes import is_excluded
    log = logging.getLogger("codegist.lang.java.extract")
    excludes = list(exclude) if exclude else []
    n_files = n_skipped = 0
    for path in sorted(corpus_dir.rglob("*.java")):
        if excludes and is_excluded(path, excludes):
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            n_skipped += 1
            continue
        # Skip files whose parse trees exceed recursion or memory limits.
        try:
            regions = extract_from_source_java(src, str(path))
        except (RecursionError, MemoryError, Exception) as e:  # noqa: BLE001
            n_skipped += 1
            log.warning("skip %s: %s: %s", path, type(e).__name__, e)
            continue
        n_files += 1
        if n_files % 5000 == 0:
            log.info("iter_corpus_java: %d files parsed (%d skipped)",
                     n_files, n_skipped)
        yield from regions
