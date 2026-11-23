"""Generate and score candidates for the baseline compressor."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken

from .grouping import Bucket
from .regions import Region, extract_from_tree
from .write_infer import RenderedTemplate, WriteTemplate, synthesize


def _enc():
    return tiktoken.get_encoding("o200k_base")


_ENC = _enc()


@lru_cache(maxsize=1 << 17)
def count_tokens(s: str) -> int:
    return len(_ENC.encode_ordinary(s))


# ---- schema set loading ----

@dataclass(frozen=True)
class SchemaRef:
    schema_id: str
    region_type: str
    sub_pattern: str
    write_name: str


def load_schemas(path: Path) -> List[SchemaRef]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[SchemaRef] = []
    for s in data["schemas"]:
        out.append(SchemaRef(
            schema_id=s["schema_id"],
            region_type=s["region_type"],
            sub_pattern=s["sub_pattern"],
            write_name=s["write"]["name"],
        ))
    return out


def _index_schemas(schemas: List[SchemaRef]) -> Dict[Tuple[str, str], List[SchemaRef]]:
    idx: Dict[Tuple[str, str], List[SchemaRef]] = {}
    for s in schemas:
        idx.setdefault((s.region_type, s.sub_pattern), []).append(s)
    return idx


# ---- candidate dataclass ----

@dataclass
class Candidate:
    schema_id: str
    region_type: str
    sub_pattern: str
    write_name: str

    # Where the rewrite would land in the AST hierarchy.
    span_kind: str                 # "node" | "stmt_range"
    node: ast.AST | None           # for "node" spans
    parent_node: ast.AST | None    # for "stmt_range" spans
    body_attr: str                 # for "stmt_range" spans
    first_idx: int                 # for "stmt_range" spans
    last_idx: int                  # for "stmt_range" spans

    # Where the rewrite would land in the source text.
    start_offset: int
    end_offset: int

    original_source: str
    rewritten: str  # Sketch with holes filled by their default rendering.
    gain: int  # Estimated savings before nested rewrites.
    gain_compact: int | None = None
                          # Gain relative to the default rendering of this node as a hole.

    # Skeleton + inner-node info, needed by Algorithm 2 (uncovered children)
    # and Algorithm 3 (recursive materialisation).
    skeleton: str = ""
    marks: List[str] = field(default_factory=list)
    inner_nodes: List[ast.AST] = field(default_factory=list)


# ---- offset helpers ----

def line_starts(src: str) -> List[int]:
    """Return cumulative char offset at the start of each 1-based line."""
    starts = [0]
    for i, ch in enumerate(src):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def pos_to_offset(starts: List[int], lineno: int, col: int) -> int:
    return starts[lineno - 1] + col


# ---- candidate generation ----

def _single_member_bucket(r: Region) -> Bucket:
    return Bucket(
        key=f"online::{r.region_type}::{r.sub_pattern}",
        region_type=r.region_type, sub_pattern=r.sub_pattern, members=[r],
    )


def _write_for(region: Region, want_name: str) -> WriteTemplate | None:
    for w in synthesize(_single_member_bucket(region)):
        if w.name == want_name:
            return w
    return None


def _node_span(node: ast.AST, starts: List[int]) -> Tuple[int, int]:
    end_line = getattr(node, "end_lineno", None) or node.lineno
    end_col = getattr(node, "end_col_offset", None)
    if end_col is None:
        end_col = max(0, (starts[end_line] - starts[end_line - 1] - 1)
                      if end_line < len(starts) else 0)
    return (pos_to_offset(starts, node.lineno, node.col_offset),
            pos_to_offset(starts, end_line, end_col))


def _range_span(region: Region, starts: List[int]) -> Tuple[int, int]:
    parent = region.parent_node
    assert parent is not None
    body = getattr(parent, region.body_attr)
    first_stmt = body[region.first_idx]
    last_stmt = body[region.last_idx]
    end_line = getattr(last_stmt, "end_lineno", None) or last_stmt.lineno
    end_col = getattr(last_stmt, "end_col_offset", None) or 0
    return (pos_to_offset(starts, first_stmt.lineno, first_stmt.col_offset),
            pos_to_offset(starts, end_line, end_col))


def generate(
    src: str,
    file: str,
    schemas: List[SchemaRef],
) -> Tuple[ast.Module, List[Candidate]]:
    """Parse ``src`` and return (AST, list of candidate rewrites)."""
    tree = ast.parse(src)
    starts = line_starts(src)
    sch_idx = _index_schemas(schemas)
    # Reuse the same tree we'll hand to selection so candidate.node ids
    # match the ids visited by the postorder DP.
    regions = extract_from_tree(tree, src, file)

    out: List[Candidate] = []
    for r in regions:
        sch_list = sch_idx.get((r.region_type, r.sub_pattern), [])
        if not sch_list:
            continue
        for sref in sch_list:
            tmpl = _write_for(r, sref.write_name)
            if tmpl is None:
                continue
            try:
                rendered: RenderedTemplate = tmpl.render(r)
                default_text = rendered.materialize_default()
            except Exception:
                continue
            gain = count_tokens(r.source) - count_tokens(default_text)
            if gain <= 0:
                continue
            # Exclude reformatting already credited to the parent candidate.
            gain_compact = None
            if r.span_kind == "node" and r.node is not None:
                try:
                    gain_compact = gain - (
                        count_tokens(r.source) - count_tokens(ast.unparse(r.node)))
                except Exception:
                    gain_compact = None
            if r.span_kind == "stmt_range":
                s, e = _range_span(r, starts)
            else:
                s, e = _node_span(r.node, starts)
            out.append(Candidate(
                schema_id=sref.schema_id,
                region_type=r.region_type,
                sub_pattern=r.sub_pattern,
                write_name=sref.write_name,
                span_kind=r.span_kind,
                node=r.node if r.span_kind == "node" else None,
                parent_node=r.parent_node,
                body_attr=r.body_attr,
                first_idx=r.first_idx,
                last_idx=r.last_idx,
                start_offset=s, end_offset=e,
                original_source=r.source,
                rewritten=default_text,
                gain=gain,
                gain_compact=gain_compact,
                skeleton=rendered.skeleton,
                marks=list(rendered.marks),
                inner_nodes=list(rendered.inner_nodes),
            ))
    return tree, out
