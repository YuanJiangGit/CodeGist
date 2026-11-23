"""Glue: run the 8-step offline mining pipeline end-to-end."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .config import Config, load as load_config
from .grouping import Bucket, group
from .llm_client import LLMClient
from .match_infer import Match, infer_match
from .regions import iter_corpus
from .validators import llm_val, structural_val, token_val
from .write_infer import WriteTemplate, synthesize


log = logging.getLogger("codegist.pipeline")


def _schema_record(
    bucket: Bucket,
    match: Match,
    write: WriteTemplate,
    token_report: token_val.TokenReport,
    struct_report: structural_val.StructuralReport,
    llm_report: llm_val.LLMReport | None,
) -> Dict[str, Any]:
    return {
        "schema_id": f"{bucket.region_type}.{bucket.sub_pattern}.{write.name}",
        "region_type": bucket.region_type,
        "sub_pattern": bucket.sub_pattern,
        "bucket_key": bucket.key,
        "support": len(bucket.members),
        "match": asdict(match),
        "write": {"name": write.name, "template": write.template},
        "validation": {
            "token": {
                "instances": token_report.instances,
                "mean_gain": token_report.mean_gain,
                "total_gain": token_report.total_gain,
            },
            "structural": {
                "instances": struct_report.instances,
                "failures": len(struct_report.failures),
            },
            "llm": (
                {
                    "judged": llm_report.judged,
                    "yes": llm_report.yes,
                    "consistency_ratio": llm_report.consistency_ratio,
                    "samples": llm_report.samples,
                }
                if llm_report is not None
                else None
            ),
        },
        "examples": [
            {"file": s["file"], "lineno": s["lineno"],
             "rewritten": s["rewritten"], "gain": s["gain"]}
            for s in token_report.samples[:3]
        ],
    }


def run(corpus_dir: Path, out_path: Path, cfg: Config | None = None,
        skip_llm: bool = False, preprocess: bool = False,
        exclude: list[str] | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional pre-pass: aggressive whitespace/docstring stripping.
    transform = None
    if preprocess:
        from .preprocess_python import preprocess as _pp
        transform = _pp
        log.info("preprocess pass: ON (strip docstrings + minify)")

    if exclude:
        from .regions import count_corpus
        kept, skipped = count_corpus(corpus_dir, exclude=exclude)
        log.info("excludes active: %d patterns; %d files kept, %d skipped",
                 len(exclude), kept, skipped)

    # Step 1 — typed region extraction
    regions = list(iter_corpus(corpus_dir, transform=transform, exclude=exclude))
    log.info("step1: extracted %d typed regions", len(regions))

    # Step 2 — candidate grouping
    buckets = group(regions, support_threshold=cfg.support_threshold)
    log.info("step2: %d buckets (support >= %d)",
             len(buckets), cfg.support_threshold)

    # Step 3 + 4 — Match / Write inference
    candidates: List[tuple[Bucket, Match, WriteTemplate]] = []
    for b in buckets:
        m = infer_match(b)
        for w in synthesize(b):
            candidates.append((b, m, w))
    log.info("step3+4: %d (Match, Write) candidates from %d buckets",
             len(candidates), len(buckets))

    # Step 5 — token validation
    after_token: List[tuple[Bucket, Match, WriteTemplate, token_val.TokenReport]] = []
    for b, m, w in candidates:
        ok, rep = token_val.validate(b, w, cfg.gain_threshold)
        if ok:
            after_token.append((b, m, w, rep))
    log.info("step5: %d candidates passed token validation (gain >= %.2f)",
             len(after_token), cfg.gain_threshold)

    # Step 6 — structural validation
    after_struct: List[tuple[Bucket, Match, WriteTemplate,
                             token_val.TokenReport,
                             structural_val.StructuralReport]] = []
    for b, m, w, trep in after_token:
        ok, srep = structural_val.validate(b, w)
        if ok:
            after_struct.append((b, m, w, trep, srep))
        else:
            log.info("step6: drop %s.%s (struct failures=%d)",
                     b.sub_pattern, w.name, len(srep.failures))
    log.info("step6: %d candidates passed structural validation",
             len(after_struct))

    # Step 7 — LLM semantic validation
    client: LLMClient | None = None
    rng = random.Random(0)
    schemas: List[Dict[str, Any]] = []
    llm_budget = cfg.llm_max_validations
    for b, m, w, trep, srep in after_struct:
        lrep: llm_val.LLMReport | None = None
        if skip_llm:
            pass  # accept all that survived structural validation
        else:
            if client is None:
                client = LLMClient(cfg)
            sample_size = max(1, min(2, len(b.members)))
            ok, lrep, used = llm_val.validate(
                b, w, client,
                sample_size=sample_size,
                budget_remaining=llm_budget,
            )
            llm_budget = max(0, llm_budget - used)
            log.info(
                "step7: %s.%s -> %s (yes=%d/%d, budget_left=%d)",
                b.sub_pattern, w.name, "PASS" if ok else "FAIL",
                lrep.yes, lrep.judged, llm_budget,
            )
            if not ok:
                continue
        schemas.append(_schema_record(b, m, w, trep, srep, lrep))

    # Step 8 — validated schema set
    out = {
        "summary": {
            "regions": len(regions),
            "buckets": len(buckets),
            "candidates": len(candidates),
            "after_token": len(after_token),
            "after_structural": len(after_struct),
            "validated_schemas": len(schemas),
        },
        "schemas": schemas,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("step8: wrote %d validated schemas to %s",
             len(schemas), out_path)
    return out
