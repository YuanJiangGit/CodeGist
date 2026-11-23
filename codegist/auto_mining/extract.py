"""Extract typed regions from Python ASTs."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
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
class AutoRegion:
    region_type: str
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


# ---- helpers ----

# Cache split lines so each region can be sliced using AST byte offsets.
try:
    _SPLITLINES = ast._splitlines_no_ff  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - private API moved/renamed
    _SPLITLINES = None


@lru_cache(maxsize=8)
def _cached_lines(src: str) -> tuple:
    return tuple(_SPLITLINES(src))  # type: ignore[misc]


def _source_segment(src: str, node: ast.AST) -> Optional[str]:
    if _SPLITLINES is None:  # fall back to stdlib if the private helper is gone
        try:
            return ast.get_source_segment(src, node)
        except Exception:
            return None
    try:
        if node.end_lineno is None or node.end_col_offset is None:
            return None
        lineno = node.lineno - 1
        end_lineno = node.end_lineno - 1
        col = node.col_offset
        end_col = node.end_col_offset
        lines = _cached_lines(src)
        if end_lineno == lineno:
            return lines[lineno].encode()[col:end_col].decode()
        first = lines[lineno].encode()[col:].decode()
        last = lines[end_lineno].encode()[:end_col].decode()
        return "".join((first, *lines[lineno + 1:end_lineno], last))
    except Exception:
        return None


def _multi_segment(src_lines: List[str], first: ast.stmt, last: ast.stmt) -> str:
    start = first.lineno - 1
    end = (last.end_lineno or last.lineno)
    return "".join(src_lines[start:end])


def _is_docstring(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


_BODY_ATTRS = ("body", "orelse", "finalbody")


# ---- per-category extractors ----

def _extract_module_imports(
    tree: ast.Module, src: str, file: str
) -> Optional[AutoRegion]:
    start = 1 if (tree.body and _is_docstring(tree.body[0])) else 0
    imports: list[ast.stmt] = []
    idx = start
    while idx < len(tree.body) and isinstance(
            tree.body[idx], (ast.Import, ast.ImportFrom)):
        imports.append(tree.body[idx])
        idx += 1
    if len(imports) < 2:
        return None
    seg = _multi_segment(src.splitlines(keepends=True), imports[0], imports[-1])
    return AutoRegion(
        region_type="module", node=tree, source=seg, file=file,
        lineno=imports[0].lineno,
        span_kind="stmt_range", parent_node=tree, body_attr="body",
        first_idx=start, last_idx=idx - 1,
        extra={"kind": "import_prefix", "count": len(imports)},
    )


# Cap the body length to limit the number of statement windows.
_WINDOW_SIZES = (2, 3, 4, 5, 6, 7, 8)
_MAX_WINDOW_BODY = 80


def _extract_data_flow_windows(
    tree: ast.Module, src: str, file: str
) -> Iterator[AutoRegion]:
    """Extract contiguous windows of two to eight statements."""
    lines = src.splitlines(keepends=True)
    for parent in ast.walk(tree):
        for attr in _BODY_ATTRS:
            body = getattr(parent, attr, None)
            if not (isinstance(body, list) and body
                    and all(isinstance(s, ast.stmt) for s in body)):
                continue
            n = len(body)
            if n > _MAX_WINDOW_BODY:
                continue
            for size in _WINDOW_SIZES:
                if size > n:
                    break
                for i in range(0, n - size + 1):
                    first = body[i]
                    last = body[i + size - 1]
                    seg = _multi_segment(lines, first, last)
                    if not seg.strip():
                        continue
                    yield AutoRegion(
                        region_type="data_flow",
                        node=first, source=seg, file=file, lineno=first.lineno,
                        span_kind="stmt_range",
                        parent_node=parent, body_attr=attr,
                        first_idx=i, last_idx=i + size - 1,
                        extra={"window_size": size},
                    )


# ---- driver ----

def extract_from_tree(tree: ast.Module, src: str, file: str) -> List[AutoRegion]:
    regions: List[AutoRegion] = []

    # module — full module + import-prefix
    seg = _source_segment(src, tree)
    if seg is not None:
        regions.append(AutoRegion("module", tree, seg, file, 1,
                                  extra={"kind": "whole_module"}))
    mi = _extract_module_imports(tree, src, file)
    if mi is not None:
        regions.append(mi)

    for node in ast.walk(tree):
        rt: Optional[str] = None
        if isinstance(node, ast.ClassDef):
            rt = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rt = "function"
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            rt = "control_flow"
        elif isinstance(node, ast.Call):
            rt = "api_idiom"
        elif isinstance(node, (ast.Compare, ast.BinOp, ast.BoolOp,
                               ast.UnaryOp, ast.IfExp)):
            rt = "expression"
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                               ast.Return)):
            # Expose standalone assignments and returns as function-body holes.
            rt = "data_flow"
        if rt is None:
            continue
        if node is tree:
            continue
        seg = _source_segment(src, node)
        if seg is None or not seg.strip():
            continue
        regions.append(AutoRegion(
            region_type=rt, node=node, source=seg, file=file,
            lineno=getattr(node, "lineno", 0) or 0,
        ))

    regions.extend(_extract_data_flow_windows(tree, src, file))
    return regions


def extract_from_source(src: str, file: str) -> List[AutoRegion]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return extract_from_tree(tree, src, file)


def iter_corpus(corpus_dir: Path, transform=None,
                exclude: list[str] | None = None) -> Iterator[AutoRegion]:
    """Yield regions from Python files, with optional filtering and preprocessing."""
    from .._excludes import is_excluded
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
    """Return (kept, skipped) ``.py`` file counts — handy for logging."""
    from .._excludes import is_excluded
    excludes = list(exclude) if exclude else []
    kept = skipped = 0
    for path in corpus_dir.rglob("*.py"):
        if excludes and is_excluded(path, excludes):
            skipped += 1
        else:
            kept += 1
    return kept, skipped
