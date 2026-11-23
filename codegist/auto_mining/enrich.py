"""Backfill AST skeletons in an existing schema set."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .extract import AutoRegion, iter_corpus
from .fingerprint import fingerprint, fingerprint_for_stmt_range
from .skeleton import build_skeleton, skeleton_to_dict

log = logging.getLogger("codegist.auto_mining.enrich")


def _region_fingerprint(r: AutoRegion) -> str:
    """The same key the pipeline buckets on (see pipeline._bucket_regions)."""
    if r.span_kind == "stmt_range" and r.parent_node is not None:
        return fingerprint_for_stmt_range(r.parent_node, r.body_attr,
                                          r.first_idx, r.last_idx,
                                          r.region_type)
    return fingerprint(r.node, r.region_type)


def build_skeleton_index(
    corpus_dir,
    *,
    preprocess: bool = False,
    exclude: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Index the first node-span skeleton for each fingerprint."""
    transform = None
    if preprocess:
        from ..preprocess_python import preprocess as _pp
        transform = _pp
    index: Dict[str, dict] = {}
    n_regions = 0
    for r in iter_corpus(corpus_dir, transform=transform, exclude=exclude):
        n_regions += 1
        if r.span_kind != "node" or r.node is None:
            continue
        key = _region_fingerprint(r)
        if key in index:
            continue
        try:
            index[key] = skeleton_to_dict(build_skeleton(r.node))
        except Exception:
            continue
    log.info("enrich: scanned %d regions, %d distinct node-span fingerprints",
             n_regions, len(index))
    return index


def enrich_schema_set_with_skeletons(
    schema_set_path,
    corpus_dir,
    out_path,
    *,
    preprocess: bool = False,
    exclude: Optional[List[str]] = None,
    overwrite: bool = False,
) -> dict:
    """Fill missing skeletons for fingerprints present in the corpus."""
    data = json.loads(Path(schema_set_path).read_text(encoding="utf-8"))
    schemas = data.get("schemas", [])
    index = build_skeleton_index(corpus_dir, preprocess=preprocess,
                                 exclude=exclude)

    enriched = already = missing = 0
    for s in schemas:
        if s.get("skeleton") and not overwrite:
            already += 1
            continue
        sk = index.get(s.get("fingerprint"))
        if sk is None:
            missing += 1
            continue
        s["skeleton"] = sk
        enriched += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    summary = {
        "total_schemas": len(schemas),
        "enriched": enriched,
        "already_had_skeleton": already,
        "fingerprint_not_in_corpus": missing,
        "corpus_node_fingerprints": len(index),
    }
    log.info("enrich done: %s", summary)
    return summary
