"""Apply selected rewrites in descending source order."""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from .candidate_gen import (
    Candidate,
    SchemaRef,
    count_tokens,
    generate,
    load_schemas,
)
from .compact_render import compact_render
from .selection import SelectionResult, select

log = logging.getLogger("codegist.compress")

SchemaSource = Union[List[SchemaRef], Path, str]


@dataclass
class CompressResult:
    original: str
    compressed: str
    selection: SelectionResult
    all_candidates: List[Candidate]
    orig_tokens: int            # tokens of the source fed into compress
    comp_tokens: int            # tokens after CodeGist
    raw_tokens: int | None = None
    preprocessed_tokens: int | None = None

    @property
    def compression_ratio(self) -> float:
        """Measure compression relative to the input before any preprocessing."""
        baseline = self.raw_tokens if self.raw_tokens is not None else self.orig_tokens
        if not baseline:
            return 0.0
        return 1.0 - self.comp_tokens / baseline

    @property
    def codegist_ratio(self) -> float:
        """CR of CodeGist alone (post-preprocess → compressed)."""
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - self.comp_tokens / self.orig_tokens

    @property
    def preprocess_ratio(self) -> float:
        if self.raw_tokens is None or not self.raw_tokens:
            return 0.0
        return 1.0 - (self.preprocessed_tokens or 0) / self.raw_tokens


def _inner_default_text(node: ast.AST, src: str) -> str:
    """Render a hole from a Python AST node or a shadow node's stored text."""
    if not isinstance(node, ast.AST):
        return getattr(node, "text", "") or ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _apply(src: str, chosen: List[Candidate], *, render_variant: str = "default",
           lang: str = "python") -> str:
    if not chosen:
        return src
    if render_variant not in {"default", "compact"}:
        raise ValueError(f"unknown render_variant: {render_variant}")
    if lang == "java":
        # Lazy import keeps tree-sitter out of every Python-only run.
        from .lang.java.render import compact_render_java as _compact
    elif lang == "typescript":
        from .lang.typescript.render import compact_render_ts as _compact
    else:
        _compact = compact_render

    by_anchor: Dict[int, Candidate] = {}
    for c in chosen:
        if c.span_kind == "node" and c.node is not None:
            by_anchor[id(c.node)] = c

    inner_anchor_ids: set[int] = set()
    for c in chosen:
        for n in c.inner_nodes:
            inner_anchor_ids.add(id(n))

    def render_chain(c: Candidate) -> str:
        if not c.inner_nodes:
            text = c.rewritten
            return _compact(text) if render_variant == "compact" else text
        out = c.skeleton
        for mark, inner in zip(c.marks, c.inner_nodes):
            sub = by_anchor.get(id(inner))
            if sub is not None:
                inner_text = render_chain(sub)
            else:
                inner_text = _inner_default_text(inner, src)
            out = out.replace(mark, inner_text, 1)
        return _compact(out) if render_variant == "compact" else out

    # Replace only outermost spans. Nested rewrites are rendered into their
    # parent's holes so source replacements cannot overlap.
    def _contained(c: Candidate) -> bool:
        for o in chosen:
            if o is c:
                continue
            if (o.start_offset <= c.start_offset
                    and c.end_offset <= o.end_offset
                    and (o.start_offset, o.end_offset)
                    != (c.start_offset, c.end_offset)):
                return True
        return False

    top_level: List[Candidate] = []
    for c in chosen:
        if c.span_kind == "node" and c.node is not None \
                and id(c.node) in inner_anchor_ids:
            continue
        if _contained(c):
            continue
        top_level.append(c)

    top_level.sort(key=lambda c: c.start_offset, reverse=True)
    out = src
    for c in top_level:
        rendered = render_chain(c)
        # Indent continuation lines after compaction to keep them within the function sketch.
        if (render_variant == "compact" and "\n" in rendered
                and rendered.startswith(("def ", "async def "))):
            head, *rest = rendered.split("\n")
            rendered = head + "".join("\n " + ln for ln in rest)
        out = out[: c.start_offset] + rendered + out[c.end_offset:]
    return out


def _resolve_schemas(schemas: SchemaSource) -> List[SchemaRef]:
    if isinstance(schemas, list):
        return schemas
    return load_schemas(Path(schemas))


# ---- public API ----

def compress_code(
    src: str,
    schemas: SchemaSource,
    file_label: str = "<memory>",
    preprocess: bool = False,
) -> CompressResult:
    """Compress a Python string with optional preprocessing."""
    schemas = _resolve_schemas(schemas)
    raw_src = src
    raw_tokens = None
    pre_tokens = None
    if preprocess:
        from .preprocess_python import preprocess as _pp
        raw_tokens = count_tokens(raw_src)
        try:
            src = _pp(src)
        except SyntaxError:
            src = raw_src  # leave it alone if it's not parseable
        pre_tokens = count_tokens(src)
    tree, cands = generate(src, file_label, schemas)
    sel = select(tree, cands)   # token-optimal tree-DP (>= greedy on additive gain)
    out = _apply(src, sel.chosen)
    return CompressResult(
        original=raw_src if preprocess else src,
        compressed=out,
        selection=sel,
        all_candidates=cands,
        orig_tokens=count_tokens(src),
        comp_tokens=count_tokens(out),
        raw_tokens=raw_tokens,
        preprocessed_tokens=pre_tokens,
    )


def compress_file(
    schemas: SchemaSource,
    source_path: Union[Path, str],
    preprocess: bool = False,
) -> CompressResult:
    """Compress a single Python file on disk."""
    schemas = _resolve_schemas(schemas)
    src = Path(source_path).read_text(encoding="utf-8")
    return compress_code(src, schemas, file_label=str(source_path),
                         preprocess=preprocess)


@dataclass
class RepoCompressResult:
    files: Dict[str, CompressResult]
    orig_tokens: int           # raw (before preprocess) when preprocess on
    comp_tokens: int           # final
    skipped: List[str]
    preprocessed_tokens: int | None = None

    @property
    def compression_ratio(self) -> float:
        if not self.orig_tokens:
            return 0.0
        return 1.0 - self.comp_tokens / self.orig_tokens

    @property
    def codegist_ratio(self) -> float:
        denom = self.preprocessed_tokens or self.orig_tokens
        if not denom:
            return 0.0
        return 1.0 - self.comp_tokens / denom

    @property
    def preprocess_ratio(self) -> float:
        if (self.preprocessed_tokens is None) or not self.orig_tokens:
            return 0.0
        return 1.0 - self.preprocessed_tokens / self.orig_tokens


def compress_repo(
    repo_dir: Union[Path, str],
    schemas: SchemaSource,
    out_dir: Union[Path, str, None] = None,
    preprocess: bool = False,
) -> RepoCompressResult:
    """Compress every ``.py`` file under ``repo_dir``."""
    schemas = _resolve_schemas(schemas)
    repo = Path(repo_dir)
    out_root = Path(out_dir) if out_dir is not None else None

    files: Dict[str, CompressResult] = {}
    skipped: List[str] = []
    total_raw = 0       # input tokens (before preprocess if any)
    total_pre = 0       # input tokens (after preprocess) — used as CodeGist baseline
    total_comp = 0
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        try:
            src = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("skip %s: %s", rel, e)
            skipped.append(rel)
            continue
        try:
            res = compress_code(src, schemas, file_label=str(path),
                                preprocess=preprocess)
        except SyntaxError as e:
            log.warning("skip %s: SyntaxError %s", rel, e)
            skipped.append(rel)
            continue
        files[rel] = res
        if preprocess and res.raw_tokens is not None:
            total_raw += res.raw_tokens
            total_pre += res.orig_tokens  # post-preprocess input
        else:
            total_raw += res.orig_tokens
        total_comp += res.comp_tokens
        if out_root is not None:
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(res.compressed, encoding="utf-8")
    return RepoCompressResult(
        files=files,
        orig_tokens=total_raw,
        comp_tokens=total_comp,
        skipped=skipped,
        preprocessed_tokens=total_pre if preprocess else None,
    )
