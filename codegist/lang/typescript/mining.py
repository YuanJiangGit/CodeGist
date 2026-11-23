"""Mine and validate schemas from TypeScript source."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...auto_mining.extract import AutoRegion
from ...candidate_gen import count_tokens
from ...config import Config, load as load_config
from ...llm_client import LLMClient
from ...validators.admissibility import information_preserved, is_generalizable
from ...auto_mining.skeleton import skeleton_to_dict
from .extract import iter_corpus_ts
from .fingerprint import fingerprint_ts, fingerprint_ts_range
from .shadow import TSNode
from .skeleton import build_skeleton_ts
from .slots import TSTemplateTransform, validate_template_ts
from .synth import synth_loop_ts

log = logging.getLogger("codegist.lang.typescript.mining")

_SYNTH_REGION_TYPES = ("function", "data_flow", "api_idiom",
                       "control_flow", "class", "expression", "module")


# Keep a bounded sample per bucket while counting all occurrences in support.
_MEMBER_CAP = 200


@dataclass
class _Bucket:
    region_type: str
    key: str
    members: List[AutoRegion] = field(default_factory=list)
    support: int = 0


def _fingerprint_region(r: AutoRegion) -> str:
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        stmts = list(r.parent_node.children)[r.first_idx:r.last_idx + 1]
        return fingerprint_ts_range(stmts, r.region_type)
    return fingerprint_ts(r.node, r.region_type)


def _root_kind(r: AutoRegion) -> str:
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        stmts = list(r.parent_node.children)
        if 0 <= r.first_idx < len(stmts):
            return stmts[r.first_idx].kind
        return ""
    return r.node.kind if isinstance(r.node, TSNode) else ""


def _required_names(r: AutoRegion) -> List[str]:
    node = r.node
    if not isinstance(node, TSNode):
        return []
    out: List[str] = []
    if r.region_type in ("function", "class"):
        nm = node.child_by_field("name")
        if nm is not None:
            out.append(nm.text)
    return out


def run_ts(corpus_dir: Path, out_path: Path, cfg: Config | None = None,
             skip_llm: bool = False, exclude: list | None = None,
             llm_synth_budget: int = 100, llm_synth_keep: int = 100,
             synth_concurrency: int = 3) -> dict:
    cfg = cfg or load_config()

    # Deep parse trees need extra recursion depth for fingerprinting.
    import sys as _sys
    _old_limit = _sys.getrecursionlimit()
    _sys.setrecursionlimit(max(_old_limit, 50000))

    # Step 1-3 — STREAM regions into fingerprint buckets (never hold them all).
    buckets: Dict[Tuple[str, str], _Bucket] = {}
    n_regions = 0
    try:
        for r in iter_corpus_ts(Path(corpus_dir), exclude=exclude):
            n_regions += 1
            try:
                fp = _fingerprint_region(r)
            except Exception:
                continue
            b = buckets.setdefault((r.region_type, fp),
                                   _Bucket(region_type=r.region_type, key=fp))
            b.support += 1
            if len(b.members) < _MEMBER_CAP:
                b.members.append(r)
            if n_regions % 500000 == 0:
                log.info("step1[ts]: %d regions streamed, %d buckets so far",
                         n_regions, len(buckets))
    finally:
        _sys.setrecursionlimit(_old_limit)
    log.info("step1[ts]: extracted %d typed regions", n_regions)

    min_support = 2
    kept = [b for b in buckets.values() if b.support >= min_support]
    log.info("step2+3[ts]: %d buckets after fingerprint clustering (>= %d)",
             len(kept), min_support)

    summary = {"language": "typescript", "regions": n_regions,
               "buckets": len(kept)}

    schemas: List[dict] = []
    if skip_llm:
        log.info("skip-llm: ts mining has no hand-written write_lib — "
                 "frontend stats only, schema set will be empty")
    else:
        # Step 5 prep — value-ranked budget per region_type.
        by_rt: Dict[str, List[_Bucket]] = {}
        for b in kept:
            by_rt.setdefault(b.region_type, []).append(b)

        def _value(b: _Bucket) -> float:
            mean_tokens = (sum(count_tokens(m.source) for m in b.members[:5])
                           / max(1, min(5, b.support)))
            return b.support * mean_tokens

        jobs: List[_Bucket] = []
        for rt in _SYNTH_REGION_TYPES:
            ranked = sorted(by_rt.get(rt, []), key=_value, reverse=True)
            jobs.extend(ranked[:llm_synth_budget])
        log.info("step5[ts]: %d buckets selected for synth "
                 "(budget %d per region_type)", len(jobs), llm_synth_budget)

        neighbours_pool: Dict[Tuple[str, str], List[AutoRegion]] = {}
        for b in kept:
            for m in b.members[:1]:
                neighbours_pool.setdefault(
                    (b.region_type, _root_kind(m)), []).append(m)

        def _synth_one(b: _Bucket):
            client = LLMClient(cfg)
            rep = b.members[0]
            pool = [r for r in neighbours_pool.get(
                        (b.region_type, _root_kind(rep)), [])
                    if _fingerprint_region(r) != b.key][:5]
            try:
                res = synth_loop_ts(b.members, b.region_type, client,
                                      neighbours=pool)
            except Exception as e:
                log.warning("synth crashed on %s: %s", b.key, e)
                return b, None
            return b, res

        results: List[Tuple[_Bucket, object]] = []
        workers = max(1, min(synth_concurrency, len(jobs) or 1))
        if workers <= 1:
            for b in jobs:
                results.append(_synth_one(b))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_synth_one, b): b for b in jobs}
                for i, fut in enumerate(as_completed(futs), start=1):
                    results.append(fut.result())
                    if i % 10 == 0 or i == len(futs):
                        log.info("step5[ts]: synth %d/%d", i, len(futs))

        # Steps 6/7/7b — validation gates.
        n_gain_drop = n_struct_drop = n_inadm = 0
        survivors: List[Tuple[_Bucket, object, TSTemplateTransform,
                              float, List[dict]]] = []
        for b, res in results:
            if res is None or res.critic_verdict != "YES":
                continue
            err = validate_template_ts(res.template, b.region_type)
            if err:
                continue
            name = f"llm_synth.ts.{b.key.split('::', 1)[-1][:8]}"
            t = TSTemplateTransform(res.template, b.region_type, name)
            total = 0
            examples: List[dict] = []
            render_fail = 0
            for m in b.members:
                try:
                    rendered = t.render(m).materialize_default()
                except Exception:
                    render_fail += 1
                    continue
                g = count_tokens(m.source) - count_tokens(rendered)
                total += g
                examples.append({"file": m.file, "lineno": m.lineno,
                                 "gain": g, "rewritten": rendered})
                req = _required_names(m)
                if any(nm and nm not in rendered for nm in req):
                    render_fail += 1
            if render_fail:
                n_struct_drop += 1
                continue
            mean = total / max(1, b.support)
            if mean < cfg.gain_threshold and not res.applies_broadly:
                n_gain_drop += 1
                continue
            ok2, r2 = information_preserved(res.template)
            ok3, r3 = is_generalizable(res.template, b.support)
            if not (ok2 and ok3):
                n_inadm += 1
                log.info("step7b[ts]: drop %s (%s)", res.template,
                         "; ".join(x for x in (r2, r3) if x))
                continue
            survivors.append((b, res, t, mean, examples))

        log.info("step6/7/7b[ts]: %d survivors (gain-drop %d, struct-drop "
                 "%d, inadmissible %d)", len(survivors), n_gain_drop,
                 n_struct_drop, n_inadm)

        # Step 8 — LLM semantic probe (budgeted; 0 budget == skip8 protocol).
        budget = cfg.llm_max_validations
        if budget > 0 and survivors:
            from ...validators.llm_val import _ask
            client = LLMClient(cfg)
            checked: List[Tuple[_Bucket, object, TSTemplateTransform,
                                float, List[dict]]] = []
            for b, res, t, mean, examples in survivors:
                take = min(2, len(b.members), budget)
                budget -= take
                ok_all = True
                for m in b.members[:take]:
                    try:
                        rendered = t.render(m).materialize_default()
                        ok, _reply = _ask(client, m.source, rendered)
                    except Exception:
                        ok = False
                    if not ok:
                        ok_all = False
                        break
                if ok_all:
                    checked.append((b, res, t, mean, examples))
                else:
                    log.info("step8[ts]: drop %s on %s", t.name, b.key)
            survivors = checked
            log.info("step8[ts]: %d schemas after semantic probe",
                     len(survivors))
        else:
            log.info("step8[ts]: SKIPPED (llm_max_validations=0)")

        # NARROW value cut, mirroring --llm-synth-keep.
        if llm_synth_keep > 0:
            narrow = [s for s in survivors if not s[1].applies_broadly]
            broad = [s for s in survivors if s[1].applies_broadly]
            narrow.sort(key=lambda s: s[3] * s[0].support, reverse=True)
            per_rt: Dict[str, int] = {}
            kept_narrow = []
            for s in narrow:
                c = per_rt.get(s[0].region_type, 0)
                if c < llm_synth_keep:
                    kept_narrow.append(s)
                    per_rt[s[0].region_type] = c + 1
            survivors = broad + kept_narrow

        for b, res, t, mean, examples in survivors:
            rep = b.members[0]
            try:
                sk = skeleton_to_dict(build_skeleton_ts(
                    rep.node if rep.span_kind == "node" else rep.node))
            except Exception:
                sk = None
            schemas.append({
                "schema_id": f"auto.{b.region_type}.{b.key.split('::', 1)[-1][:8]}"
                             f".{t.name}",
                "region_type": b.region_type,
                "fingerprint": b.key,
                "transform": t.name,
                "template": res.template,
                "support": b.support,
                "token_mean_gain": mean,
                "skeleton": sk,
                "examples": examples[:3],
                "llm_synth": {
                    "applies_broadly": res.applies_broadly,
                    "root_class": _root_kind(rep),
                    "rationale": res.rationale,
                    "generality_verdict": res.generality_verdict,
                },
            })

    out = {"language": "typescript", "schemas": schemas, "summary": summary}
    summary["validated_schemas"] = len(schemas)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("wrote %d ts schemas to %s", len(schemas), out_path)
    return {"summary": summary}
