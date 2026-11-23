"""Mine and validate Python schemas from a source corpus."""
from __future__ import annotations

import ast
import json
import logging
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import tiktoken

from ..config import Config, load as load_config
from ..llm_client import LLMClient
from .extract import AutoRegion, iter_corpus
from .fingerprint import fingerprint, fingerprint_for_stmt_range
from .skeleton import Skeleton, build_skeleton, skeleton_to_dict
from .write_lib import ALL_TRANSFORMS, WriteTransform

log = logging.getLogger("codegist.auto_mining")


def _enc():
    return tiktoken.get_encoding("o200k_base")


_ENC = _enc()


def _tokens(s: str) -> int:
    return len(_ENC.encode(s, disallowed_special=()))


_BUILTINS = {
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "print", "range", "enumerate", "zip", "map", "filter", "sum", "min",
    "max", "abs", "any", "all", "sorted", "reversed", "open", "type",
    "isinstance", "issubclass", "iter", "next", "repr", "self",
}


def _free_names(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and n.id not in _BUILTINS}


def _all_param_names(fn: ast.AST) -> set:
    a = fn.args
    names = {arg.arg for arg in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


# ---- bucketing ----

@dataclass
class _AutoBucket:
    key: str
    region_type: str
    members: List[AutoRegion]
    skeleton: Skeleton | None = None


def _bucket_regions(regions: List[AutoRegion]) -> Dict[str, _AutoBucket]:
    by_key: Dict[str, _AutoBucket] = {}
    for r in regions:
        if r.span_kind == "stmt_range" and r.parent_node is not None:
            key = fingerprint_for_stmt_range(r.parent_node, r.body_attr,
                                             r.first_idx, r.last_idx,
                                             r.region_type)
        else:
            key = fingerprint(r.node, r.region_type)
        b = by_key.get(key)
        if b is None:
            b = _AutoBucket(key=key, region_type=r.region_type, members=[])
            by_key[key] = b
        b.members.append(r)
    return by_key


# ---- structural validation helpers ----

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokens_in(text: str) -> set:
    return set(_WORD_RE.findall(text))


def _structural_required(transform: WriteTransform, r: AutoRegion) -> set:
    """Set of identifiers that MUST appear in the rewritten text."""
    req: set = set()
    n = r.node
    if transform.name == "arrow_def":
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            req.add(n.name)
            req.update(_all_param_names(n))
    elif transform.name == "class_init_sketch":
        if isinstance(n, ast.ClassDef):
            req.add(n.name)
            from .write_lib import _skip_docstring
            init = _skip_docstring(n.body)[0]
            for s in _skip_docstring(init.body):
                req.add(s.targets[0].attr)
    elif transform.name == "dataclass_sketch":
        if isinstance(n, ast.ClassDef):
            req.add(n.name)
            from .write_lib import _skip_docstring
            for s in _skip_docstring(n.body):
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                    req.add(s.target.id)
    elif transform.name in ("if_return_inline", "if_return_sugar",
                            "if_raise_inline", "if_raise_sugar"):
        if isinstance(n, ast.If):
            req.update(_free_names(n.test))
            body0 = n.body[0]
            if isinstance(body0, ast.Return) and body0.value is not None:
                req.update(_free_names(body0.value))
            elif isinstance(body0, ast.Raise) and body0.exc is not None:
                req.update(_free_names(body0.exc))
    elif transform.name == "accumulator_reduce":
        if r.extra.get("acc"):
            req.add(r.extra["acc"])
    elif transform.name == "with_open_arrow":
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            req.add(n.name)
            req.update(_all_param_names(n))
    elif transform.name == "import_group_sketch":
        body = getattr(r.parent_node, r.body_attr, None) or []
        for s in body[r.first_idx:r.last_idx + 1]:
            if isinstance(s, ast.Import):
                for a in s.names:
                    req.update(a.name.split("."))
            elif isinstance(s, ast.ImportFrom):
                if s.module:
                    req.update(s.module.split("."))
                req.update(a.name for a in s.names)
    elif transform.name == "expression_idiom":
        req.update(_free_names(n))
    elif transform.name.startswith("llm_synth"):
        # Check required identifiers by region type. Data-flow and module sketches may omit names.
        rt = r.region_type
        if rt == "class" and isinstance(n, ast.ClassDef):
            # Field/method-level checks would over-constrain the LLM's design
            # choices; just enforce the class identity survives.
            req.add(n.name)
        elif rt == "function" and isinstance(n, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef)):
            req.add(n.name)
            req.update(_all_param_names(n))   # catches lossy {params}
        elif rt == "api_idiom" and isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                req.add(n.func.attr)
            req.update(_free_names(n.func))
            for arg in n.args:
                req.update(_free_names(arg))
            for kw in n.keywords:
                if kw.arg:
                    req.add(kw.arg)
                req.update(_free_names(kw.value))
        elif rt == "expression":
            # operand free names must survive — also catches the {lit}
            # empty-string trap (a Name RHS rendering as ``a==`` would drop b).
            if isinstance(n, (ast.Compare, ast.BinOp, ast.BoolOp, ast.UnaryOp)):
                req.update(_free_names(n))
        elif rt == "control_flow":
            # the controlling expression must survive; body summarisation
            # (dropping body internals) stays allowed.
            if isinstance(n, (ast.If, ast.While, ast.IfExp)):
                req.update(_free_names(n.test))
            elif isinstance(n, ast.For):
                req.update(_free_names(n.target))
                req.update(_free_names(n.iter))
            elif isinstance(n, ast.With):
                for item in n.items:
                    req.update(_free_names(item.context_expr))
            elif isinstance(n, ast.Try):
                for h in n.handlers:
                    if h.type is not None:
                        req.update(_free_names(h.type))
    # Allow templates to abbreviate dunder names.
    return {x for x in req
            if x not in _BUILTINS and not (x.startswith("__") and x.endswith("__"))}


# ---- pipeline ----

@dataclass
class _CandidateSchema:
    region_type: str
    fingerprint: str
    transform: WriteTransform
    support: int
    skeleton: Skeleton
    examples: List[dict] = field(default_factory=list)
    token_mean_gain: float = 0.0
    token_total_gain: int = 0
    structural_failures: int = 0
    llm_judged: int = 0
    llm_yes: int = 0


def _is_broad_llm_synth(c: "_CandidateSchema",
                        llm_synth_results: Dict[str, dict]) -> bool:
    if not c.transform.name.startswith("llm_synth"):
        return False
    info = llm_synth_results.get(c.fingerprint)
    return bool(info and info.get("applies_broadly"))


_INNER_ONLINE_POTENTIAL_REGION_TYPES = {
    "data_flow", "expression", "control_flow", "api_idiom",
}


def _is_inner_online_potential(
    c: "_CandidateSchema",
    llm_synth_results: Dict[str, dict],
) -> bool:
    """Check whether an admissible inner template qualifies for the gain exemption."""
    if c.region_type not in _INNER_ONLINE_POTENTIAL_REGION_TYPES:
        return False
    if not c.transform.name.startswith("llm_synth"):
        return False
    info = llm_synth_results.get(c.fingerprint)
    if not info:
        return False
    tmpl = info.get("template", "")
    from ..validators.admissibility import is_admissible
    ok, _reasons = is_admissible(tmpl, c.support, region_type=c.region_type)
    if not ok:
        return False
    slots = set(re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", tmpl))
    if c.region_type == "expression":
        return bool(slots)
    if c.region_type in ("data_flow", "control_flow"):
        return bool(slots & {"{stmt_chain}", "{body_chain}", "{body_blocks}"})
    if c.region_type == "api_idiom":
        return "{call_args_combined}" in slots
    return False


def _value_rank_narrow_llm_synth(
    survivors: List["_CandidateSchema"],
    llm_synth_results: Dict[str, dict],
    keep: int,
) -> tuple[List["_CandidateSchema"], int]:
    """Keep the top narrow schemas per region type by total token savings."""
    if not keep or keep <= 0:
        return survivors, 0
    narrow_by_rt: Dict[str, List["_CandidateSchema"]] = defaultdict(list)
    passthrough: List["_CandidateSchema"] = []
    for c in survivors:
        if (c.transform.name.startswith("llm_synth")
                and not _is_broad_llm_synth(c, llm_synth_results)
                and not _is_inner_online_potential(c, llm_synth_results)):
            narrow_by_rt[c.region_type].append(c)
        else:
            passthrough.append(c)
    kept: List["_CandidateSchema"] = list(passthrough)
    dropped = 0
    for _rt, group in narrow_by_rt.items():
        group.sort(key=lambda x: -x.token_total_gain)
        kept.extend(group[:keep])
        dropped += max(0, len(group) - keep)
    return kept, dropped


def _skeleton_repr(sk: Skeleton, depth: int = 0) -> str:
    if sk.kind == "op":
        return sk.op_category
    if sk.kind != "node":
        return f"?{sk.kind}"
    if not sk.fields:
        return sk.node_class
    if depth >= 3:
        return f"{sk.node_class}(...)"
    parts = []
    for fname, sub in sk.fields.items():
        if sub is None:
            continue
        if isinstance(sub, list):
            if not sub:
                continue
            elems = ",".join(_skeleton_repr(s, depth + 1) for s in sub[:3])
            if len(sub) > 3:
                elems += ",..."
            parts.append(f"{fname}=[{elems}]")
        elif isinstance(sub, Skeleton):
            parts.append(f"{fname}={_skeleton_repr(sub, depth + 1)}")
    return f"{sk.node_class}({', '.join(parts)})"


def _build_llm_synth_transform(template: str, rt: str, name: str) -> WriteTransform:
    from .llm_synth import render_template_structured

    def _matches(_sk, r):
        return r.region_type == rt

    def _render(r):
        return render_template_structured(template, r)

    return WriteTransform(name=name, region_types=(rt,),
                          matches=_matches, render=_render)


def _bucket_value(b: _AutoBucket, _sample: int = 8) -> int:
    """Estimate the bucket's token volume from sampled regions."""
    members = b.members
    if not members:
        return 0
    sample = members[:_sample]
    mean_tok = sum(_tokens(m.source) for m in sample) / len(sample)
    return int(len(members) * mean_tok)


def _select_buckets_for_synth(buckets: List[_AutoBucket],
                              budget: int) -> List[_AutoBucket]:
    """Apply a per-region-type synthesis budget ranked by estimated token volume."""
    from .llm_synth import supported_region_types
    from collections import defaultdict

    import os
    supported = set(supported_region_types())
    # Set CODEGIST_SKIP_REGION_TYPES to override the synthesis exclusions.
    _srt = os.environ.get("CODEGIST_SKIP_REGION_TYPES")
    if _srt is None:
        skip_rt = {"module", "api_idiom"}
    elif _srt.strip().lower() in ("", "none"):
        skip_rt = set()
    else:
        skip_rt = {t.strip() for t in _srt.split(",") if t.strip()}
    if skip_rt:
        log.info("llm-synth skipping region_types (default module,api_idiom; "
                 "override with CODEGIST_SKIP_REGION_TYPES): %s", sorted(skip_rt))
    value: Dict[str, int] = {}
    by_rt: dict = defaultdict(list)
    for b in buckets:
        if b.region_type in supported and b.region_type not in skip_rt:
            value[b.key] = _bucket_value(b)
            by_rt[b.region_type].append(b)

    selected: List[_AutoBucket] = []
    for rt, group in by_rt.items():
        group.sort(key=lambda x: -value[x.key])
        head = group[:budget]
        selected.extend(head)
        log.info("llm-synth selecting %d/%d buckets from region_type=%s "
                 "(top by token-value)", len(head), len(group), rt)
    selected.sort(key=lambda b: -value[b.key])
    log.info("llm-synth total selected: %d buckets across %d region_types "
             "(budget=%d per region_type, ranked by token-value)",
             len(selected), len(by_rt), budget)
    return selected


def _run_llm_synth(buckets: List[_AutoBucket], cfg: Config,
                   budget: int, concurrency: int = 5) -> Dict[str, dict]:
    from .llm_synth import synth_loop, _is_pure_chain_template
    import random as _random

    eligible = _select_buckets_for_synth(buckets, budget)

    # The generality check compares buckets with the same region type and root class.
    def _root_class(b: _AutoBucket) -> str:
        if b.skeleton is None or b.skeleton.kind != "node":
            return ""
        return b.skeleton.node_class

    by_rt_root: Dict[tuple, List[_AutoBucket]] = {}
    for b in buckets:
        root = _root_class(b)
        if not root:
            continue
        by_rt_root.setdefault((b.region_type, root), []).append(b)

    def _sample_neighbours(current: _AutoBucket,
                           k_top: int = 3, k_random: int = 4
                           ) -> List[AutoRegion]:
        """Sample same-root buckets with different fingerprints for the generality check."""
        root = _root_class(current)
        pool = [b for b in by_rt_root.get((current.region_type, root), [])
                if b.key != current.key and b.members]
        if not pool:
            return []
        pool_sorted = sorted(pool, key=lambda x: -len(x.members))
        top = pool_sorted[:k_top]
        chosen_keys = {b.key for b in top}
        remaining = [b for b in pool if b.key not in chosen_keys]
        if remaining:
            rnd = _random.Random(hash(current.key) & 0xFFFFFFFF)
            picks = rnd.sample(remaining, min(k_random, len(remaining)))
        else:
            picks = []
        return [b.members[0] for b in (top + picks)]

    client = LLMClient(cfg)
    out: Dict[str, dict] = {}

    def _process(b: _AutoBucket):
        neighbours = _sample_neighbours(b)
        try:
            res = synth_loop(b.members, b.region_type, client,
                             neighbours=neighbours)
        except Exception as e:
            log.warning("llm-synth crashed on %s: %s", b.key, e)
            return None
        if res is None:
            log.info("llm-synth: no template produced for %s", b.key)
            return None
        if res.critic_verdict not in ("YES", "UNKNOWN"):
            log.info("llm-synth REJECT %s (rounds=%d, reason=%s)",
                     b.key, res.rounds, res.critic_reason)
            return None
        broad = res.applies_broadly
        gen_verdict = res.generality_verdict
        # Record broad eligibility for pure function chains during mining.
        if b.region_type == "function" and _is_pure_chain_template(res.template):
            broad = True
            gen_verdict = "BROAD"
        info = {
            "template": res.template, "rationale": res.rationale,
            "rounds": res.rounds, "verdict": res.critic_verdict,
            "reason": res.critic_reason,
            "applies_broadly": broad,
            "refined": res.refined,
            "generality_verdict": gen_verdict,
            "generality_reasons": res.generality_reasons,
            "neighbours_used": len(neighbours),
        }
        log.info("llm-synth OK %s -> %r (rounds=%d, gen=%s, broad=%s%s)",
                 b.key, res.template, res.rounds,
                 res.generality_verdict, res.applies_broadly,
                 ", refined" if res.refined else "")
        return (b.key, info)

    workers = max(1, min(concurrency, len(eligible)))
    log.info("llm-synth: %d eligible buckets, %d concurrent worker(s)",
             len(eligible), workers)
    if workers <= 1:
        for b in eligible:
            r = _process(b)
            if r is not None:
                out[r[0]] = r[1]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_process, b)
                                     for b in eligible]):
                r = fut.result()
                if r is not None:
                    out[r[0]] = r[1]
    return out


def run(corpus_dir: Path, out_path: Path, cfg: Config | None = None,
        skip_llm: bool = False, preprocess: bool = False,
        exclude: list[str] | None = None,
        use_llm_synth: bool = False,
        llm_synth_budget: int = 100,
        llm_synth_keep: int = 100,
        synth_cache: Dict[str, dict] | None = None,
        layered: bool = False,
        synth_concurrency: int = 5) -> Dict[str, Any]:
    """Mine schemas, optionally reusing LLM templates recovered from a log."""
    cfg = cfg or load_config()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transform = None
    if preprocess:
        from ..preprocess_python import preprocess as _pp
        transform = _pp
        log.info("preprocess pass: ON (strip docstrings + minify)")

    if exclude:
        from .extract import count_corpus
        kept, skipped = count_corpus(corpus_dir, exclude=exclude)
        log.info("excludes active: %d patterns; %d files kept, %d skipped",
                 len(exclude), kept, skipped)

    regions = list(iter_corpus(corpus_dir, transform=transform, exclude=exclude))
    log.info("step1: extracted %d typed regions (no sub_pattern)", len(regions))

    by_key = _bucket_regions(regions)
    buckets = [b for b in by_key.values()
               if len(b.members) >= cfg.support_threshold]
    log.info("step2+3: %d buckets after fingerprint clustering (>= %d)",
             len(buckets), cfg.support_threshold)

    for b in buckets:
        b.skeleton = build_skeleton(b.members[0].node)

    transform_pool = [] if use_llm_synth else list(ALL_TRANSFORMS)

    llm_synth_results: Dict[str, dict] = {}
    llm_synth_xforms: Dict[str, WriteTransform] = {}
    if use_llm_synth:
        if synth_cache is not None:
            log.info("llm-synth: using INJECTED cache (%d templates) — "
                     "Step 4b SynthLLM↔CriticLLM SKIPPED (zero LLM)",
                     len(synth_cache))
            llm_synth_results = synth_cache
        else:
            log.info("llm-synth ON: SynthLLM↔CriticLLM up to %d buckets, "
                     "budget=%d rounds×2", llm_synth_budget, llm_synth_budget)
            llm_synth_results = _run_llm_synth(buckets, cfg, llm_synth_budget,
                                               concurrency=synth_concurrency)
        bucket_by_key = {bb.key: bb for bb in buckets}
        matched = 0
        for key, info in llm_synth_results.items():
            b = bucket_by_key.get(key)
            if b is None:
                # Cached key with no surviving bucket (e.g. support changed,
                # or a stale log) — skip it rather than crash.
                continue
            name = f"llm_synth.{key.split('::', 1)[-1][:8]}"
            llm_synth_xforms[key] = _build_llm_synth_transform(
                info["template"], b.region_type, name)
            matched += 1
        log.info("llm-synth produced %d bucket-specific transforms"
                 "%s", len(llm_synth_xforms),
                 f" ({matched}/{len(llm_synth_results)} cached keys matched a "
                 f"live bucket)" if synth_cache is not None else "")

    candidates: List[_CandidateSchema] = []
    for b in buckets:
        # Bucket-specific LLM-synth transform (if any) gets a shot first.
        if b.key in llm_synth_xforms:
            t = llm_synth_xforms[b.key]
            try:
                ok = t.matches(b.skeleton, b.members[0])
            except Exception:
                ok = False
            if ok:
                candidates.append(_CandidateSchema(
                    region_type=b.region_type, fingerprint=b.key,
                    transform=t, support=len(b.members), skeleton=b.skeleton,
                ))
        for t in transform_pool:
            if b.region_type not in t.region_types:
                continue
            # The match predicate can populate fields used by the renderer.
            try:
                ok = t.matches(b.skeleton, b.members[0])
            except Exception:
                ok = False
            if not ok:
                continue
            candidates.append(_CandidateSchema(
                region_type=b.region_type, fingerprint=b.key,
                transform=t, support=len(b.members), skeleton=b.skeleton,
            ))
    log.info("step5: %d (skeleton, Write) candidates enumerated",
             len(candidates))

    # Step 6 — token validation.
    survivors: List[_CandidateSchema] = []
    gain_exempt = 0
    for c in candidates:
        bucket = next(b for b in buckets if b.key == c.fingerprint
                      and b.region_type == c.region_type)
        total = 0
        examples: List[dict] = []
        for r in bucket.members:
            try:
                rendered = c.transform.render(r).materialize_default()
            except Exception:
                rendered = r.source
            g = _tokens(r.source) - _tokens(rendered)
            total += g
            examples.append({"file": r.file, "lineno": r.lineno,
                             "orig_tokens": _tokens(r.source),
                             "rewritten_tokens": _tokens(rendered),
                             "gain": g, "rewritten": rendered})
        n = len(bucket.members)
        mean = total / n if n else 0
        c.token_mean_gain = mean
        c.token_total_gain = total
        c.examples = examples
        # Broad templates and eligible inner templates may save more tokens
        # online than on their mining bucket.
        if (_is_broad_llm_synth(c, llm_synth_results)
                or _is_inner_online_potential(c, llm_synth_results)):
            survivors.append(c)
            if mean < cfg.gain_threshold:
                gain_exempt += 1
        elif mean >= cfg.gain_threshold:
            survivors.append(c)
    log.info("step6: %d candidates passed token validation "
             "(mean_gain >= %.2f; %d BROAD llm_synth γ-exempt)",
             len(survivors), cfg.gain_threshold, gain_exempt)

    # Step 6b — value-rank NARROW llm_synth survivors (see
    # ``_value_rank_narrow_llm_synth`` for the rationale).
    survivors, dropped6b = _value_rank_narrow_llm_synth(
        survivors, llm_synth_results, llm_synth_keep)
    if dropped6b:
        log.info("step6b: value-ranked NARROW llm_synth — kept top-%d "
                 "per region_type by token_total_gain, dropped %d "
                 "low-gain tail", llm_synth_keep, dropped6b)

    # Validate required identifiers on every bucket member and parse call
    # sketches as Python expressions.
    after_struct: List[_CandidateSchema] = []
    for c in survivors:
        bucket = next(b for b in buckets if b.key == c.fingerprint
                      and b.region_type == c.region_type)
        fails = 0
        for r in bucket.members:
            try:
                rendered = c.transform.render(r).materialize_default()
            except Exception:
                fails += 1
                continue
            req = _structural_required(c.transform, r)
            words = _tokens_in(rendered)
            if not req.issubset(words):
                fails += 1
                continue
            if c.region_type == "api_idiom":
                try:
                    ast.parse(rendered, mode="eval")
                except SyntaxError:
                    fails += 1
                    continue
        c.structural_failures = fails
        if fails == 0:
            after_struct.append(c)
        else:
            log.info("step7: drop %s on %s (struct failures=%d/%d)",
                     c.transform.name, c.fingerprint, fails, c.support)
    log.info("step7: %d candidates passed structural validation",
             len(after_struct))

    # Check template admissibility before the optional LLM validation.
    from ..validators.admissibility import is_admissible, render_lexically_valid
    after_admissible: List[_CandidateSchema] = []
    n_inadmissible = 0
    repair_requests: List[tuple[_CandidateSchema, List[str]]] = []
    for c in after_struct:
        if (c.transform.name.startswith("llm_synth")
                and c.fingerprint in llm_synth_results):
            tmpl = llm_synth_results[c.fingerprint].get("template", "")
            ok, reasons = is_admissible(tmpl, c.support, region_type=c.region_type)
            if ok:
                # Check the filled sketch for invalid tokens introduced by slot values.
                bkt = next((b for b in buckets if b.key == c.fingerprint
                            and b.region_type == c.region_type), None)
                if bkt is not None and bkt.members:
                    try:
                        rendered = c.transform.render(
                            bkt.members[0]).materialize_default()
                        rok, rreason = render_lexically_valid(rendered)
                        if not rok:
                            ok, reasons = False, ["dynI1:" + rreason]
                    except Exception:
                        pass
            if not ok:
                n_inadmissible += 1
                log.info("step7b: drop INADMISSIBLE %s on %s (%s)",
                         c.transform.name, c.fingerprint, "; ".join(reasons))
                if (use_llm_synth
                        and c.region_type in _INNER_ONLINE_POTENTIAL_REGION_TYPES
                        and any(r.startswith(("I2:", "I2b:"))
                                for r in reasons)):
                    repair_requests.append((c, reasons))
                continue
        after_admissible.append(c)
    log.info("step7b: %d/%d admissible (dropped %d inadmissible llm_synth)",
             len(after_admissible), len(after_struct), n_inadmissible)

    if repair_requests:
        from .llm_synth import synth_loop

        bucket_lookup = {(b.region_type, b.key): b for b in buckets}
        log.info("step7b-repair: attempting admissibility-guided repair for "
                 "%d inner schema(s)", len(repair_requests))

        def _repair_one(
            item: tuple[_CandidateSchema, List[str]],
        ) -> tuple[_CandidateSchema, dict] | None:
            c, reasons = item
            bkt = bucket_lookup.get((c.region_type, c.fingerprint))
            if bkt is None or not bkt.members:
                return None
            old_info = llm_synth_results.get(c.fingerprint, {})
            old_template = old_info.get("template", "")
            feedback = (
                "The previous template failed deterministic admissibility "
                "validation and must be redesigned, not merely reworded.\n"
                f"Previous template: {old_template}\n"
                f"Admissibility failure(s): {'; '.join(reasons)}\n"
                "Use a full-coverage, information-preserving template. For "
                "data_flow, avoid endpoint-only sketches such as "
                "{first_targets}={last_expr} and avoid {call_chain} as the "
                "coverage slot; prefer {stmt_chain} when it preserves the "
                "whole statement/range dependency signature. Under-compression "
                "is acceptable; dropping middle dependencies is not."
            )
            client = LLMClient(cfg)
            try:
                res = synth_loop(
                    bkt.members, c.region_type, client,
                    neighbours=[], initial_feedback=feedback,
                )
            except Exception as e:
                log.warning("step7b-repair: crashed on %s: %s",
                            c.fingerprint, e)
                return None
            if res is None or res.critic_verdict not in ("YES", "UNKNOWN"):
                return None
            info = {
                "template": res.template,
                "rationale": res.rationale,
                "rounds": res.rounds,
                "verdict": res.critic_verdict,
                "reason": res.critic_reason,
                "applies_broadly": False,
                "refined": res.refined,
                "generality_verdict": "REPAIR",
                "generality_reasons": res.generality_reasons,
                "neighbours_used": 0,
                "admissibility_repair": True,
                "admissibility_repair_reasons": reasons,
            }
            ok, new_reasons = is_admissible(
                info["template"], c.support, region_type=c.region_type)
            if not ok:
                log.info("step7b-repair: reject %s on %s (%s)",
                         info["template"], c.fingerprint,
                         "; ".join(new_reasons))
                return None
            name = f"llm_synth_repair.{c.fingerprint.split('::', 1)[-1][:8]}"
            t = _build_llm_synth_transform(info["template"], c.region_type, name)
            try:
                if not t.matches(c.skeleton, bkt.members[0]):
                    return None
            except Exception:
                return None
            rc = _CandidateSchema(
                region_type=c.region_type,
                fingerprint=c.fingerprint,
                transform=t,
                support=c.support,
                skeleton=c.skeleton,
            )

            total = 0
            examples: List[dict] = []
            for r in bkt.members:
                try:
                    rendered = rc.transform.render(r).materialize_default()
                except Exception:
                    rendered = r.source
                g = _tokens(r.source) - _tokens(rendered)
                total += g
                examples.append({"file": r.file, "lineno": r.lineno,
                                 "orig_tokens": _tokens(r.source),
                                 "rewritten_tokens": _tokens(rendered),
                                 "gain": g, "rewritten": rendered})
            n = len(bkt.members)
            rc.token_mean_gain = total / n if n else 0
            rc.token_total_gain = total
            rc.examples = examples
            if not (rc.token_mean_gain >= cfg.gain_threshold
                    or _is_inner_online_potential(rc, {rc.fingerprint: info})):
                return None

            fails = 0
            for r in bkt.members:
                try:
                    rendered = rc.transform.render(r).materialize_default()
                except Exception:
                    fails += 1
                    continue
                req = _structural_required(rc.transform, r)
                words = _tokens_in(rendered)
                if not req.issubset(words):
                    fails += 1
                    continue
                if rc.region_type == "api_idiom":
                    try:
                        ast.parse(rendered, mode="eval")
                    except SyntaxError:
                        fails += 1
                        continue
            rc.structural_failures = fails
            if fails:
                log.info("step7b-repair: structural drop %s on %s "
                         "(failures=%d/%d)",
                         rc.transform.name, rc.fingerprint, fails, rc.support)
                return None
            return rc, info

        repaired: List[tuple[_CandidateSchema, dict]] = []
        workers = max(1, min(synth_concurrency, len(repair_requests)))
        if workers <= 1:
            for item in repair_requests:
                r = _repair_one(item)
                if r is not None:
                    repaired.append(r)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(_repair_one, item): item
                    for item in repair_requests
                }
                for fut in as_completed(future_map):
                    r = fut.result()
                    if r is not None:
                        repaired.append(r)

        existing = {(c.region_type, c.fingerprint) for c in after_admissible}
        added = 0
        for rc, info in repaired:
            key = (rc.region_type, rc.fingerprint)
            if key in existing:
                continue
            llm_synth_results[rc.fingerprint] = info
            after_admissible.append(rc)
            existing.add(key)
            added += 1
        log.info("step7b-repair: added %d repaired admissible schema(s)",
                 added)
    after_struct = after_admissible

    # Step 8 — LLM semantic consistency probe.
    rng = random.Random(0)
    budget = cfg.llm_max_validations
    buckets_by_key = {(b.region_type, b.key): b for b in buckets}
    # Set CODEGIST_SKIP_STEP8=0 to enable the semantic probe.
    # A zero validation budget rejects all candidates when the probe is enabled.
    import os as _os
    _s8 = _os.environ.get("CODEGIST_SKIP_STEP8", "1").strip().lower()
    skip_step8 = skip_llm or (_s8 not in ("0", "false", "no", ""))
    if not skip_step8:
        from ..validators.llm_val import _ask

        thread_state = threading.local()
        jobs: List[tuple[_CandidateSchema, List[AutoRegion]]] = []
        for c in after_struct:
            bucket = buckets_by_key[(c.region_type, c.fingerprint)]
            sample = list(bucket.members)
            rng.shuffle(sample)
            take = min(2, len(sample), budget)
            budget = max(0, budget - take)
            jobs.append((c, sample[:take]))

        call_count = sum(len(sample) for _c, sample in jobs)
        workers = min(synth_concurrency,
                      max(1, len([1 for _c, sample in jobs if sample])))
        log.info("step8: LLM semantic validation for %d candidates "
                 "(%d calls budgeted, parallelism=%d)",
                 len(after_struct), call_count, workers)

        def _validate_one(
            c: _CandidateSchema,
            sample: List[AutoRegion],
        ) -> tuple[_CandidateSchema, int, int]:
            client = getattr(thread_state, "client", None)
            if client is None:
                client = LLMClient(cfg)
                thread_state.client = client
            judged = 0
            yes = 0
            for r in sample:
                try:
                    rewritten = c.transform.render(r).materialize_default()
                except Exception:
                    continue
                ok, _reply = _ask(client, r.source, rewritten)
                judged += 1
                if ok:
                    yes += 1
            return c, judged, yes

        active_jobs = [(c, sample) for c, sample in jobs if sample]
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(_validate_one, c, sample): c
                for c, sample in active_jobs
            }
            for fut in as_completed(future_map):
                c = future_map[fut]
                try:
                    _c, judged, yes = fut.result()
                except Exception as e:
                    judged = 0
                    yes = 0
                    log.info("step8: LLM worker failed for %s on %s: %s",
                             c.transform.name, c.fingerprint, e)
                c.llm_judged = judged
                c.llm_yes = yes
                completed += 1
                if completed % 50 == 0 or completed == len(active_jobs):
                    log.info("step8: LLM progress %d/%d candidates",
                             completed, len(active_jobs))
    final_schemas: List[Dict[str, Any]] = []
    for c in after_struct:
        if not skip_step8:
            # Both sampled instances must receive YES when Step 8 is enabled.
            is_broad_llm = _is_broad_llm_synth(c, llm_synth_results)
            min_yes_required = 2
            if c.llm_judged < 2 or c.llm_yes < min_yes_required:
                log.info("step8: drop %s on %s (llm=%d/%d, required %d/2%s)",
                         c.transform.name, c.fingerprint,
                         c.llm_yes, c.llm_judged, min_yes_required,
                         " [BROAD]" if is_broad_llm else "")
                continue
        skel_str = _skeleton_repr(c.skeleton)
        record = {
            "schema_id": f"auto.{c.region_type}.{c.fingerprint.split('::', 1)[1][:8]}.{c.transform.name}",
            "region_type": c.region_type,
            "fingerprint": c.fingerprint,
            "transform": c.transform.name,
            "support": c.support,
            "skeleton_repr": skel_str,
            "skeleton": (skeleton_to_dict(c.skeleton)
                         if c.skeleton is not None else None),
            "validation": {
                "token": {"mean_gain": c.token_mean_gain,
                          "total_gain": c.token_total_gain},
                "structural": {"failures": c.structural_failures},
                "llm": {"judged": c.llm_judged, "yes": c.llm_yes}
                        if not skip_step8 else None,
            },
            "examples": [{"file": e["file"], "lineno": e["lineno"],
                          "gain": e["gain"], "rewritten": e["rewritten"]}
                         for e in c.examples[:3]],
        }
        if c.transform.name.startswith("llm_synth") and c.fingerprint in llm_synth_results:
            info = llm_synth_results[c.fingerprint]
            root_class = ""
            if c.skeleton is not None and c.skeleton.kind == "node":
                root_class = c.skeleton.node_class
            record["template"] = info["template"]
            record["llm_synth"] = {
                "rationale": info.get("rationale", ""),
                "rounds": info.get("rounds", 0),
                "critic_verdict": info.get("verdict", ""),
                "critic_reason": info.get("reason", ""),
                # Phase B/C generality results — drive online matching
                "applies_broadly": info.get("applies_broadly", False),
                "refined": info.get("refined", False),
                "generality_verdict": info.get("generality_verdict", ""),
                "generality_reasons": info.get("generality_reasons", []),
                "neighbours_used": info.get("neighbours_used", 0),
                "root_class": root_class,
            }
        final_schemas.append(record)

    if layered:
        log.info("ALG2 layered: function body slot = {body_blocks} (block-split); "
                 "combiner + block schemas are SynthLLM-mined, none hand-injected")

    out = {
        "summary": {
            "regions": len(regions),
            "buckets": len(buckets),
            "candidates": len(candidates),
            "after_token": len(survivors),
            "after_structural": len(after_struct),
            "validated_schemas": len(final_schemas),
        },
        "schemas": final_schemas,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("step8: wrote %d validated schemas to %s",
             len(final_schemas), out_path)
    return out
