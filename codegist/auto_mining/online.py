"""Generate and apply rewrites from mined Python schemas."""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..candidate_gen import Candidate, count_tokens, line_starts, pos_to_offset
from ..compact_render import compact_render
from ..compress import _apply
from ..selection import select, SelectionResult
from .extract import AutoRegion, extract_from_tree
from .fingerprint import fingerprint, fingerprint_for_stmt_range
from .skeleton import skeleton_from_dict, skeleton_matches
from .write_lib import ALL_TRANSFORMS, WriteTransform

log = logging.getLogger("codegist.auto_mining.online")


@dataclass(frozen=True)
class AutoSchemaRef:
    schema_id: str
    region_type: str
    fingerprint: str
    transform_name: str
    # Broad LLM templates can match within their recorded root class.
    template: Optional[str] = None
    applies_broadly: Optional[bool] = None
    root_class: Optional[str] = None
    # Stored Match skeleton for structural matching of narrow schemas.
    skeleton: Optional[dict] = None


def load_schemas(path: Union[Path, str]) -> List[AutoSchemaRef]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Lazy import keeps the openai-bearing import chain out of module load.
    from .llm_synth import _validate_template
    out: List[AutoSchemaRef] = []
    dropped = 0
    for s in data.get("schemas", []):
        transform_name = s.get("transform", "")
        template = s.get("template")
        # Reject obsolete slots before they can appear literally in the sketch.
        if transform_name.startswith("llm_synth") and template:
            err = _validate_template(template, s.get("region_type", ""))
            if err is not None:
                log.warning("load_schemas: dropping stale schema %s (%s): %r",
                            s.get("schema_id"), err, template)
                dropped += 1
                continue
        ls = s.get("llm_synth") or {}
        applies_broadly = ls.get("applies_broadly")
        if applies_broadly is None and transform_name.startswith("llm_synth"):
            # Treat schemas without a generality verdict as narrow.
            applies_broadly = False
        root_class = ls.get("root_class")
        out.append(AutoSchemaRef(
            schema_id=s["schema_id"],
            region_type=s["region_type"],
            fingerprint=s["fingerprint"],
            transform_name=transform_name,
            template=template,
            applies_broadly=applies_broadly,
            root_class=root_class,
            skeleton=s.get("skeleton"),
        ))
    if dropped:
        log.warning("load_schemas: dropped %d stale llm_synth schema(s) "
                    "referencing slots no longer in SLOT_VOCAB", dropped)
    return out


_TRANSFORMS_BY_NAME = {t.name: t for t in ALL_TRANSFORMS}


def _resolve_schemas(
    schemas: Union[List[AutoSchemaRef], Path, str],
) -> List[AutoSchemaRef]:
    if isinstance(schemas, list):
        return schemas
    return load_schemas(schemas)


def _index_by_fp(schemas: List[AutoSchemaRef]) -> Dict[Tuple[str, str], List[AutoSchemaRef]]:
    idx: Dict[Tuple[str, str], List[AutoSchemaRef]] = {}
    for s in schemas:
        idx.setdefault((s.region_type, s.fingerprint), []).append(s)
    return idx


def _index_by_region_transform(
    schemas: List[AutoSchemaRef],
) -> Dict[Tuple[str, str], List[AutoSchemaRef]]:
    idx: Dict[Tuple[str, str], List[AutoSchemaRef]] = {}
    for s in schemas:
        idx.setdefault((s.region_type, s.transform_name), []).append(s)
    return idx


def _refs_by_strict_fingerprint(
    sch_idx: Dict[Tuple[str, str], List[AutoSchemaRef]],
    regions: List[AutoRegion],
):
    """Return a callable ``region -> list[AutoSchemaRef]`` for strict mode."""
    def _lookup(r: AutoRegion) -> List[AutoSchemaRef]:
        if r.span_kind == "stmt_range" and r.parent_node is not None:
            fp = fingerprint_for_stmt_range(
                r.parent_node, r.body_attr,
                r.first_idx, r.last_idx, r.region_type)
        else:
            fp = fingerprint(r.node, r.region_type)
        return sch_idx.get((r.region_type, fp), [])
    return _lookup


def _refs_by_region_transform(
    by_rt_trans: Dict[Tuple[str, str], List[AutoSchemaRef]],
):
    """Look up built-in transforms by region type."""
    by_rt: Dict[str, List[AutoSchemaRef]] = {}
    for (rt, tname), refs in by_rt_trans.items():
        if tname.startswith("llm_synth"):
            # llm_synth refs are added selectively (per-schema) by the
            # caller — only those with applies_broadly=True belong here
            continue
        rep = min(refs, key=lambda x: x.schema_id)
        by_rt.setdefault(rt, []).append(rep)

    def _lookup(r: AutoRegion) -> List[AutoSchemaRef]:
        return by_rt.get(r.region_type, [])
    return _lookup


def _root_class_of(r: AutoRegion) -> str:
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        # Use the first statement's class as the root class of a statement range.
        body = getattr(r.parent_node, r.body_attr, None) or []
        if 0 <= r.first_idx < len(body):
            return type(body[r.first_idx]).__name__
        return ""
    return type(r.node).__name__


def _refs_by_fingerprint_for_llm_synth(
    schemas: List[AutoSchemaRef],
    *,
    only_narrow: bool = False,
):
    idx: Dict[Tuple[str, str], List[AutoSchemaRef]] = {}
    for s in schemas:
        if not s.transform_name.startswith("llm_synth"):
            continue
        if only_narrow and s.applies_broadly is True:
            continue
        idx.setdefault((s.region_type, s.fingerprint), []).append(s)

    def _lookup(r: AutoRegion) -> List[AutoSchemaRef]:
        if r.span_kind == "stmt_range" and r.parent_node is not None:
            fp = fingerprint_for_stmt_range(
                r.parent_node, r.body_attr,
                r.first_idx, r.last_idx, r.region_type)
        else:
            fp = fingerprint(r.node, r.region_type)
        candidates = idx.get((r.region_type, fp), [])
        if not candidates:
            return candidates
        # Defence-in-depth root_class gate (fingerprint already implies this)
        root_now = _root_class_of(r)
        return [s for s in candidates
                if not s.root_class or s.root_class == root_now]
    return _lookup


def _refs_by_skeleton_for_llm_synth(
    schemas: List[AutoSchemaRef],
):
    """Index narrow schemas with usable skeletons for matching node spans."""
    by_key: Dict[Tuple[str, str], List[Tuple[AutoSchemaRef, object]]] = {}
    no_skeleton = 0
    for s in schemas:
        if not s.transform_name.startswith("llm_synth"):
            continue
        if s.applies_broadly is True:
            continue
        if not s.skeleton:
            no_skeleton += 1
            continue
        try:
            sk = skeleton_from_dict(s.skeleton)
        except Exception:
            no_skeleton += 1
            continue
        key = (s.region_type, s.root_class or "")
        by_key.setdefault(key, []).append((s, sk))
    if no_skeleton:
        log.warning("narrow_match=skeleton: %d NARROW llm_synth schema(s) "
                    "lack a persisted skeleton and were skipped — re-enrich "
                    "the schema set to match them", no_skeleton)

    def _lookup(r: AutoRegion) -> List[AutoSchemaRef]:
        if r.span_kind != "node" or r.node is None:
            return []
        root_now = _root_class_of(r)
        bucket = by_key.get((r.region_type, root_now), [])
        if not bucket:
            # Also try the no-root-class bucket (schemas mined without a
            # recorded root_class still match within region_type).
            bucket = by_key.get((r.region_type, ""), [])
            if not bucket:
                return []
        out: List[AutoSchemaRef] = []
        for s, sk in bucket:
            try:
                if skeleton_matches(sk, r.node):
                    out.append(s)
            except Exception:
                continue
        return out
    return _lookup


def _llm_synth_broad_by_region_type(
    schemas: List[AutoSchemaRef],
):
    by_rt: Dict[str, List[AutoSchemaRef]] = {}
    for s in schemas:
        if (s.transform_name.startswith("llm_synth")
                and s.applies_broadly is True):
            by_rt.setdefault(s.region_type, []).append(s)

    def _lookup(r: AutoRegion) -> List[AutoSchemaRef]:
        candidates = by_rt.get(r.region_type, [])
        if not candidates:
            return candidates
        root_now = _root_class_of(r)
        return [s for s in candidates
                if not s.root_class or s.root_class == root_now]
    return _lookup


def _node_span(node: ast.AST, starts: List[int]) -> Tuple[int, int]:
    end_line = getattr(node, "end_lineno", None) or node.lineno
    end_col = getattr(node, "end_col_offset", None)
    if end_col is None:
        end_col = max(0, (starts[end_line] - starts[end_line - 1] - 1)
                      if end_line < len(starts) else 0)
    return (pos_to_offset(starts, node.lineno, node.col_offset),
            pos_to_offset(starts, end_line, end_col))


def _range_span(r: AutoRegion, starts: List[int]) -> Tuple[int, int]:
    body = getattr(r.parent_node, r.body_attr)
    first = body[r.first_idx]
    last = body[r.last_idx]
    end_line = getattr(last, "end_lineno", None) or last.lineno
    end_col = getattr(last, "end_col_offset", None) or 0
    return (pos_to_offset(starts, first.lineno, first.col_offset),
            pos_to_offset(starts, end_line, end_col))


def _expr_required_leaves(node: ast.AST) -> set:
    leaves: set = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            leaves.add(n.id)
        elif isinstance(n, ast.Attribute):
            leaves.add(n.attr)
        elif isinstance(n, ast.Constant):
            leaves.add(repr(n.value))
    return leaves


def _expr_drops_leaf(node: ast.AST, rendered: str) -> bool:
    for leaf in _expr_required_leaves(node):
        if not leaf:
            continue
        if leaf.isidentifier():
            if not re.search(r"\b" + re.escape(leaf) + r"\b", rendered):
                return True
        elif leaf not in rendered:
            return True
    return False


def _call_required_leaves(node: ast.AST) -> Tuple[set, set]:
    """Collect callee leaves and keyword names required in a call sketch."""
    targets: set = set()
    keys: set = set()
    if isinstance(node, ast.Call):
        for n in ast.walk(node.func):
            if isinstance(n, ast.Name):
                targets.add(n.id)
            elif isinstance(n, ast.Attribute):
                targets.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                targets.add(n.value)
        for kw in node.keywords:
            if kw.arg:
                keys.add(kw.arg)
    return targets, keys


def _call_drops_leaf(node: ast.AST, rendered: str) -> bool:
    targets, keys = _call_required_leaves(node)
    for leaf in targets:
        if not leaf:
            continue
        if leaf.isidentifier():
            if not re.search(r"\b" + re.escape(leaf) + r"\b", rendered):
                return True
        elif leaf not in rendered:
            return True
    for key in keys:
        if not re.search(r"\b" + re.escape(key) + r"\b", rendered):
            return True
    return False


# Checks for values omitted by call-chain templates

def _stmt_value_exprs(stmt: ast.stmt) -> list:
    """The value expression(s) a statement evaluates (RHS / return / bare)."""
    if isinstance(stmt, ast.Return):
        return [stmt.value] if stmt.value is not None else []
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return [stmt.value] if stmt.value is not None else []
    if isinstance(stmt, ast.Expr):
        return [stmt.value]
    return []


def _chain_required_leaves(node: ast.AST) -> set:
    """Collect values from computations that a call chain might omit."""
    leaves: set = set()
    body = getattr(node, "body", None)
    if not isinstance(body, list):
        return leaves
    try:
        from .write_lib import _skip_docstring
        stmts = _skip_docstring(body)
    except Exception:
        stmts = body
    for stmt in stmts:
        for value in _stmt_value_exprs(stmt):
            if value is None:
                continue
            if any(isinstance(n, ast.Call) for n in ast.walk(value)):
                continue  # call-bearing → captured by the chain
            if isinstance(value, (ast.Name, ast.Constant)):
                continue  # trivial bare value → not gist-bearing glue
            for n in ast.walk(value):
                if isinstance(n, ast.Name):
                    leaves.add(n.id)
                elif isinstance(n, ast.Attribute):
                    leaves.add(n.attr)
                elif isinstance(n, ast.Constant):
                    leaves.add(repr(n.value))
    return leaves


def _chain_loses_gist(node: ast.AST, rendered: str) -> bool:
    # Search only the body so matching names in the signature cannot mask omissions.
    body_part = rendered.split(": ", 1)[1] if ": " in rendered else rendered
    for leaf in _chain_required_leaves(node):
        if not leaf:
            continue
        if leaf.isidentifier():
            if not re.search(r"\b" + re.escape(leaf) + r"\b", body_part):
                return True
        elif leaf not in body_part:
            return True
    return False


def _ternary_safe(node: ast.AST, template: str) -> bool:
    """Require single-statement branches with matching assignment targets."""
    if not isinstance(node, ast.If):
        return False
    if len(node.body) != 1 or len(node.orelse) != 1:
        return False
    b, e = node.body[0], node.orelse[0]
    if "{tern_target}" in template:
        if not (isinstance(b, ast.Assign) and isinstance(e, ast.Assign)):
            return False
        if len(b.targets) != 1 or len(e.targets) != 1:
            return False
        return ast.dump(b.targets[0]) == ast.dump(e.targets[0])
    return (isinstance(b, ast.Return) and b.value is not None
            and isinstance(e, ast.Return) and e.value is not None)


def _subscript_assign_rewrite_drops_target(node: ast.AST, rendered: str) -> bool:
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Subscript):
        return False
    try:
        base = ast.unparse(target.value)
        key = ast.unparse(target.slice)
    except Exception:
        return False
    return bool(base and base not in rendered) or bool(key and key not in rendered)


def generate_candidates(
    src: str, file: str, schemas: List[AutoSchemaRef],
    match_mode: str = "auto",
    narrow_match: str = "fingerprint",
    score_variant: str = "default",
) -> Tuple[ast.Module, List[Candidate]]:
    """Generate candidate rewrites with token savings estimated per region."""
    if score_variant not in {"default", "compact"}:
        raise ValueError(f"unknown score_variant: {score_variant}")
    tree = ast.parse(src)
    starts = line_starts(src)
    regions = extract_from_tree(tree, src, file)

    if match_mode == "auto":
        by_rt_trans = _index_by_region_transform(schemas)
        write_lib_refs_for = _refs_by_region_transform(by_rt_trans)
        llm_synth_broad_for = _llm_synth_broad_by_region_type(schemas)
        if narrow_match == "skeleton":
            llm_synth_narrow_for = _refs_by_skeleton_for_llm_synth(schemas)
        elif narrow_match == "fingerprint":
            llm_synth_narrow_for = _refs_by_fingerprint_for_llm_synth(
                schemas, only_narrow=True)
        else:
            raise ValueError(f"unknown narrow_match: {narrow_match}")
        def candidate_refs_for(r):
            return (write_lib_refs_for(r)
                    + llm_synth_broad_for(r)
                    + llm_synth_narrow_for(r))
    elif match_mode == "fingerprint":
        sch_idx = _index_by_fp(schemas)
        candidate_refs_for = _refs_by_strict_fingerprint(sch_idx, regions)
    elif match_mode == "transform":
        by_rt_trans = _index_by_region_transform(schemas)
        # Force legacy behavior: include llm_synth in transform-mode too
        # (the broadcast that caused the over-generalization bug).
        all_refs: Dict[str, List[AutoSchemaRef]] = {}
        for (rt, tname), refs in by_rt_trans.items():
            for ref in refs:
                all_refs.setdefault(rt, []).append(ref)
        for s in schemas:
            if s.transform_name.startswith("llm_synth"):
                all_refs.setdefault(s.region_type, []).append(s)
        def candidate_refs_for(r):
            return all_refs.get(r.region_type, [])
    else:
        raise ValueError(f"unknown match_mode: {match_mode}")

    # Per-ref cache for LLM-synth transforms: rebuild once per template.
    _llm_synth_xform_cache: Dict[str, WriteTransform] = {}

    def _resolve_transform(ref: AutoSchemaRef) -> Optional[WriteTransform]:
        if ref.transform_name.startswith("llm_synth") and ref.template:
            cached = _llm_synth_xform_cache.get(ref.schema_id)
            if cached is None:
                from .pipeline import _build_llm_synth_transform
                cached = _build_llm_synth_transform(
                    ref.template, ref.region_type, ref.transform_name)
                _llm_synth_xform_cache[ref.schema_id] = cached
            return cached
        return _TRANSFORMS_BY_NAME.get(ref.transform_name)

    out: List[Candidate] = []
    for r in regions:
        refs = candidate_refs_for(r)
        if not refs:
            continue
        # The original region source tokenizes to the same count for every
        # ref matched against this region — compute it once instead of per ref.
        src_tokens = count_tokens(r.source)
        for ref in refs:
            t = _resolve_transform(ref)
            if t is None:
                continue
            # `matches` may mutate r.extra (e.g. accumulator stores acc/op);
            # we tolerate any exception there to keep the online phase robust.
            try:
                if not t.matches(None, r):  # type: ignore[arg-type]
                    continue
            except Exception:
                continue
            try:
                rendered = t.render(r)
                default_text = rendered.materialize_default()
            except Exception:
                continue
            if (r.region_type == "expression" and r.span_kind == "node"
                    and r.node is not None
                    and _expr_drops_leaf(r.node, default_text)):
                continue
            if (r.region_type == "api_idiom" and r.span_kind == "node"
                    and r.node is not None
                    and _call_drops_leaf(r.node, default_text)):
                continue
            if (r.region_type == "function" and r.node is not None
                    and ref.template
                    and ("{body_chain}" in ref.template
                         or "{call_chain}" in ref.template)
                    and _chain_loses_gist(r.node, default_text)):
                continue
            if (ref.template and "{tern_" in ref.template
                    and not _ternary_safe(r.node, ref.template)):
                continue
            # An empty dict-build rendering means the pattern did not match.
            if (ref.template and "{dict_build}" in ref.template
                    and not default_text.strip()):
                continue
            if (r.region_type == "data_flow" and r.span_kind == "node"
                    and r.node is not None
                    and _subscript_assign_rewrite_drops_target(
                        r.node, default_text)):
                continue
            scored_text = (
                compact_render(default_text)
                if score_variant == "compact"
                else default_text
            )
            gain = src_tokens - count_tokens(scored_text)
            if gain <= 0:
                continue
            # Score nested rewrites against the hole as rendered by the parent.
            gain_compact = None
            if r.span_kind == "node" and r.node is not None:
                try:
                    verbatim = ast.unparse(r.node)
                    verbatim_scored = (compact_render(verbatim)
                                       if score_variant == "compact" else verbatim)
                    gain_compact = gain - (src_tokens
                                           - count_tokens(verbatim_scored))
                except Exception:
                    gain_compact = None
            if r.span_kind == "stmt_range":
                s, e = _range_span(r, starts)
            else:
                s, e = _node_span(r.node, starts)
            out.append(Candidate(
                schema_id=ref.schema_id,
                region_type=r.region_type,
                sub_pattern="<auto>",
                write_name=ref.transform_name,
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


# ---- public API mirroring codegist.compress ----

@dataclass
class AutoCompressResult:
    original: str
    compressed: str
    selection: SelectionResult
    all_candidates: List[Candidate]
    orig_tokens: int
    comp_tokens: int
    raw_tokens: int | None = None
    preprocessed_tokens: int | None = None

    @property
    def compression_ratio(self) -> float:
        baseline = self.raw_tokens if self.raw_tokens is not None else self.orig_tokens
        if not baseline:
            return 0.0
        return 1.0 - self.comp_tokens / baseline

    @property
    def codegist_ratio(self) -> float:
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - self.comp_tokens / self.orig_tokens

    @property
    def preprocess_ratio(self) -> float:
        if self.raw_tokens is None or not self.raw_tokens:
            return 0.0
        return 1.0 - (self.preprocessed_tokens or 0) / self.raw_tokens


def auto_compress_code(
    src: str,
    schemas: Union[List[AutoSchemaRef], Path, str],
    file_label: str = "<memory>",
    preprocess: bool = False,
    match_mode: str = "auto",
    narrow_match: str = "fingerprint",
    score_variant: str = "default",
    render_variant: str = "default",
) -> AutoCompressResult:
    refs = _resolve_schemas(schemas)
    raw_src = src
    raw_tokens = None
    pre_tokens = None
    if preprocess:
        from ..preprocess_python import preprocess as _pp
        raw_tokens = count_tokens(raw_src)
        try:
            src = _pp(src)
        except SyntaxError:
            src = raw_src
        pre_tokens = count_tokens(src)
    tree, cands = generate_candidates(src, file_label, refs,
                                      match_mode=match_mode,
                                      narrow_match=narrow_match,
                                      score_variant=score_variant)
    sel = select(tree, cands)   # token-optimal tree-DP (>= greedy on additive gain)
    out = _apply(src, sel.chosen, render_variant=render_variant)
    return AutoCompressResult(
        original=raw_src if preprocess else src,
        compressed=out, selection=sel, all_candidates=cands,
        orig_tokens=count_tokens(src), comp_tokens=count_tokens(out),
        raw_tokens=raw_tokens, preprocessed_tokens=pre_tokens,
    )


def auto_compress_file(
    schemas: Union[List[AutoSchemaRef], Path, str],
    source_path: Union[Path, str],
    preprocess: bool = False,
    match_mode: str = "auto",
    narrow_match: str = "fingerprint",
    score_variant: str = "default",
    render_variant: str = "default",
) -> AutoCompressResult:
    refs = _resolve_schemas(schemas)
    src = Path(source_path).read_text(encoding="utf-8")
    return auto_compress_code(src, refs, file_label=str(source_path),
                              preprocess=preprocess, match_mode=match_mode,
                              narrow_match=narrow_match,
                              score_variant=score_variant,
                              render_variant=render_variant)


@dataclass
class AutoRepoResult:
    files: Dict[str, AutoCompressResult]
    orig_tokens: int
    comp_tokens: int
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


def auto_compress_repo(
    repo_dir: Union[Path, str],
    schemas: Union[List[AutoSchemaRef], Path, str],
    out_dir: Union[Path, str, None] = None,
    preprocess: bool = False,
    match_mode: str = "auto",
    narrow_match: str = "fingerprint",
    score_variant: str = "default",
    render_variant: str = "default",
) -> AutoRepoResult:
    refs = _resolve_schemas(schemas)
    repo = Path(repo_dir)
    out_root = Path(out_dir) if out_dir is not None else None
    files: Dict[str, AutoCompressResult] = {}
    skipped: List[str] = []
    total_raw = 0
    total_pre = 0
    total_comp = 0
    for path in sorted(repo.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            skipped.append(rel)
            continue
        try:
            res = auto_compress_code(src, refs, file_label=str(path),
                                     preprocess=preprocess,
                                     match_mode=match_mode,
                                     narrow_match=narrow_match,
                                     score_variant=score_variant,
                                     render_variant=render_variant)
        except SyntaxError:
            skipped.append(rel)
            continue
        files[rel] = res
        if preprocess and res.raw_tokens is not None:
            total_raw += res.raw_tokens
            total_pre += res.orig_tokens
        else:
            total_raw += res.orig_tokens
        total_comp += res.comp_tokens
        if out_root is not None:
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(res.compressed, encoding="utf-8")
    return AutoRepoResult(files=files, orig_tokens=total_raw,
                          comp_tokens=total_comp, skipped=skipped,
                          preprocessed_tokens=total_pre if preprocess else None)
