"""Generate and apply rewrites from TypeScript schemas."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...auto_mining.online import AutoSchemaRef, AutoCompressResult
from ...auto_mining.skeleton import skeleton_from_dict
from ...candidate_gen import Candidate, count_tokens
from ...compress import _apply
from ...selection import select
from .extract import extract_from_tree_ts
from .render import compact_render_ts
from .shadow import TSNode, parse_ts
from .skeleton import skeleton_matches_ts
from .slots import TSTemplateTransform, validate_template_ts, _callee_name

log = logging.getLogger("codegist.lang.typescript.online")


def load_schemas_ts(path: Union[Path, str]) -> List[AutoSchemaRef]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[AutoSchemaRef] = []
    dropped = 0
    for s in data.get("schemas", []):
        template = s.get("template")
        if not template:
            continue
        err = validate_template_ts(template, s.get("region_type", ""))
        if err is not None:
            log.warning("load_schemas_ts: dropping schema %s (%s)",
                        s.get("schema_id"), err)
            dropped += 1
            continue
        ls = s.get("llm_synth") or {}
        out.append(AutoSchemaRef(
            schema_id=s["schema_id"],
            region_type=s["region_type"],
            fingerprint=s["fingerprint"],
            transform_name=s.get("transform", "llm_synth.ts"),
            template=template,
            applies_broadly=bool(ls.get("applies_broadly")),
            root_class=ls.get("root_class"),
            skeleton=s.get("skeleton"),
        ))
    if dropped:
        log.warning("load_schemas_ts: dropped %d invalid schema(s)", dropped)
    return out


def _root_kind_of(region) -> str:
    if region.span_kind == "stmt_range" and region.parent_node is not None:
        stmts = list(region.parent_node.children)
        if 0 <= region.first_idx < len(stmts):
            return stmts[region.first_idx].kind
        return ""
    return region.node.kind if isinstance(region.node, TSNode) else ""


def generate_candidates_ts(
    src: str, file: str, schemas: List[AutoSchemaRef],
    score_variant: str = "compact",
) -> Tuple[TSNode, List[Candidate]]:
    root = parse_ts(src, tsx=str(file).endswith(".tsx"))
    regions = extract_from_tree_ts(root, src, file)

    broad_by_rt: Dict[str, List[AutoSchemaRef]] = {}
    narrow: List[Tuple[AutoSchemaRef, object]] = []
    for s in schemas:
        if s.applies_broadly:
            broad_by_rt.setdefault(s.region_type, []).append(s)
        elif s.skeleton:
            try:
                narrow.append((s, skeleton_from_dict(s.skeleton)))
            except Exception:
                continue

    xform_cache: Dict[str, TSTemplateTransform] = {}

    def _xform(ref: AutoSchemaRef) -> TSTemplateTransform:
        t = xform_cache.get(ref.schema_id)
        if t is None:
            t = TSTemplateTransform(ref.template or "", ref.region_type,
                                      ref.transform_name)
            xform_cache[ref.schema_id] = t
        return t

    out: List[Candidate] = []
    for r in regions:
        refs: List[AutoSchemaRef] = []
        root_now = _root_kind_of(r)
        for s in broad_by_rt.get(r.region_type, []):
            if s.root_class and s.root_class != root_now:
                continue
            refs.append(s)
        if r.span_kind == "node" and isinstance(r.node, TSNode):
            for s, sk in narrow:
                if s.region_type != r.region_type:
                    continue
                if s.root_class and s.root_class != root_now:
                    continue
                try:
                    if skeleton_matches_ts(sk, r.node):
                        refs.append(s)
                except Exception:
                    continue
        for ref in refs:
            t = _xform(ref)
            try:
                if not t.matches(None, r):
                    continue
                rendered = t.render(r)
                default_text = rendered.materialize_default()
            except Exception:
                continue
            # api_idiom guard: the callee must survive the rewrite.
            if r.region_type == "api_idiom" and isinstance(r.node, TSNode):
                callee = _callee_name(r.node)
                base = callee.split(".")[-1] if callee else ""
                if base and base not in default_text:
                    continue
            scored = (compact_render_ts(default_text)
                      if score_variant == "compact" else default_text)
            gain = count_tokens(r.source) - count_tokens(scored)
            if gain <= 0:
                continue
            if r.span_kind == "stmt_range":
                stmts = list(r.parent_node.children)
                s_off = stmts[r.first_idx].start
                e_off = stmts[r.last_idx].end
            else:
                s_off, e_off = r.node.start, r.node.end
            out.append(Candidate(
                schema_id=ref.schema_id,
                region_type=r.region_type,
                sub_pattern="<auto-ts>",
                write_name=ref.transform_name,
                span_kind=r.span_kind,
                node=r.node if r.span_kind == "node" else None,
                parent_node=r.parent_node,
                body_attr=r.body_attr,
                first_idx=r.first_idx,
                last_idx=r.last_idx,
                start_offset=s_off, end_offset=e_off,
                original_source=r.source,
                rewritten=default_text,
                gain=gain,
                skeleton=rendered.skeleton,
                marks=list(rendered.marks),
                inner_nodes=list(rendered.inner_nodes),
            ))
    return root, out


def auto_compress_code_ts(
    src: str,
    schemas: Union[List[AutoSchemaRef], Path, str],
    file_label: str = "<memory>",
    score_variant: str = "compact",
    render_variant: str = "compact",
) -> AutoCompressResult:
    refs = (schemas if isinstance(schemas, list)
            else load_schemas_ts(schemas))
    tree, cands = generate_candidates_ts(src, file_label, refs,
                                           score_variant=score_variant)
    sel = select(tree, cands)
    out = _apply(src, sel.chosen, render_variant=render_variant, lang="typescript")
    return AutoCompressResult(
        original=src, compressed=out, selection=sel, all_candidates=cands,
        orig_tokens=count_tokens(src), comp_tokens=count_tokens(out),
    )


def auto_compress_repo_ts(
    repo_dir: Union[Path, str],
    schemas: Union[List[AutoSchemaRef], Path, str],
    out_dir: Union[Path, str, None] = None,
    score_variant: str = "compact",
    render_variant: str = "compact",
):
    refs = (schemas if isinstance(schemas, list)
            else load_schemas_ts(schemas))
    repo = Path(repo_dir)
    out_root = Path(out_dir) if out_dir is not None else None
    files: Dict[str, AutoCompressResult] = {}
    skipped: List[str] = []
    total_orig = total_comp = 0
    paths = sorted(list(repo.rglob("*.ts")) + list(repo.rglob("*.tsx")))
    for path in paths:
        if path.name.endswith(".d.ts"):
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            src = path.read_text(encoding="utf-8")
            res = auto_compress_code_ts(src, refs, file_label=str(path),
                                          score_variant=score_variant,
                                          render_variant=render_variant)
        except (OSError, SyntaxError) as e:
            log.warning("skip %s: %s", rel, e)
            skipped.append(rel)
            continue
        files[rel] = res
        total_orig += res.orig_tokens
        total_comp += res.comp_tokens
        if out_root is not None:
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(res.compressed, encoding="utf-8")
    from ...auto_mining.online import AutoRepoResult
    return AutoRepoResult(files=files, orig_tokens=total_orig,
                          comp_tokens=total_comp, skipped=skipped)
