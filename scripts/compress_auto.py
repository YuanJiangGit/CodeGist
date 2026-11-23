"""CLI: online compression with auto-mined schemas."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codegist.auto_mining.online import (  # noqa: E402
    auto_compress_file, auto_compress_repo, load_schemas,
)


def _resolve_filtered_schemas(args: argparse.Namespace):
    """Load LLM-synthesized schemas and optionally include built-in transforms."""
    schemas = load_schemas(args.schemas)
    n_total = len(schemas)
    if not args.include_write_lib:
        schemas = [s for s in schemas
                   if s.transform_name.startswith("llm_synth")]
        print(f"[filter] llm-synth-only (default): {len(schemas)}/{n_total} "
              f"schemas kept; pass --include-write-lib to load all")
    return schemas


def _single_java(args: argparse.Namespace) -> int:
    from codegist.lang.java.online import (auto_compress_code_java,
                                           load_schemas_java)
    refs = load_schemas_java(args.schemas)
    print(f"[lang] java schema set: {len(refs)} schemas")
    src = Path(args.input).read_text(encoding="utf-8")
    res = auto_compress_code_java(src, refs, file_label=args.input,
                                  score_variant=args.score_variant,
                                  render_variant=args.render_variant)
    print(f"candidates_generated : {len(res.all_candidates)}")
    print(f"candidates_selected  : {len(res.selection.chosen)}")
    print(f"orig_tokens          : {res.orig_tokens}")
    print(f"comp_tokens          : {res.comp_tokens}")
    print(f"compression_ratio    : {res.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- chosen rewrites ---")
        for c in res.selection.chosen:
            print(f"  [{c.span_kind:10s}] {c.schema_id} gain={c.gain}  "
                  f"-> {c.rewritten!r}")
    if args.out:
        Path(args.out).write_text(res.compressed, encoding="utf-8")
        print(f"\nwrote: {args.out}")
    else:
        print("\n=== compressed ===")
        print(res.compressed)
    return 0


def _batch_java(args: argparse.Namespace) -> int:
    if not args.out_dir:
        print("--out-dir required with --input-dir", file=sys.stderr)
        return 2
    from codegist.lang.java.online import (auto_compress_repo_java,
                                           load_schemas_java)
    refs = load_schemas_java(args.schemas)
    print(f"[lang] java schema set: {len(refs)} schemas")
    repo = auto_compress_repo_java(args.input_dir, refs,
                                   out_dir=args.out_dir,
                                   score_variant=args.score_variant,
                                   render_variant=args.render_variant)
    print(f"files_compressed     : {len(repo.files)}")
    print(f"files_skipped        : {len(repo.skipped)}")
    print(f"orig_tokens (total)  : {repo.orig_tokens}")
    print(f"comp_tokens (total)  : {repo.comp_tokens}")
    print(f"compression_ratio    : {repo.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- per-file CR ---")
        for rel, res in sorted(repo.files.items()):
            print(f"  {rel:40s} {res.compression_ratio * 100:6.2f}%  "
                  f"({res.orig_tokens} -> {res.comp_tokens})")
    return 0


def _single_ts(args: argparse.Namespace) -> int:
    from codegist.lang.typescript.online import (auto_compress_code_ts,
                                                 load_schemas_ts)
    refs = load_schemas_ts(args.schemas)
    print(f"[lang] typescript schema set: {len(refs)} schemas")
    src = Path(args.input).read_text(encoding="utf-8")
    res = auto_compress_code_ts(src, refs, file_label=args.input,
                                score_variant=args.score_variant,
                                render_variant=args.render_variant)
    print(f"candidates_generated : {len(res.all_candidates)}")
    print(f"candidates_selected  : {len(res.selection.chosen)}")
    print(f"orig_tokens          : {res.orig_tokens}")
    print(f"comp_tokens          : {res.comp_tokens}")
    print(f"compression_ratio    : {res.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- chosen rewrites ---")
        for c in res.selection.chosen:
            print(f"  [{c.span_kind:10s}] {c.schema_id} gain={c.gain}  "
                  f"-> {c.rewritten!r}")
    if args.out:
        Path(args.out).write_text(res.compressed, encoding="utf-8")
        print(f"\nwrote: {args.out}")
    else:
        print("\n=== compressed ===")
        print(res.compressed)
    return 0


def _batch_ts(args: argparse.Namespace) -> int:
    if not args.out_dir:
        print("--out-dir required with --input-dir", file=sys.stderr)
        return 2
    from codegist.lang.typescript.online import (auto_compress_repo_ts,
                                                 load_schemas_ts)
    refs = load_schemas_ts(args.schemas)
    print(f"[lang] typescript schema set: {len(refs)} schemas")
    repo = auto_compress_repo_ts(args.input_dir, refs,
                                 out_dir=args.out_dir,
                                 score_variant=args.score_variant,
                                 render_variant=args.render_variant)
    print(f"files_compressed     : {len(repo.files)}")
    print(f"files_skipped        : {len(repo.skipped)}")
    print(f"orig_tokens (total)  : {repo.orig_tokens}")
    print(f"comp_tokens (total)  : {repo.comp_tokens}")
    print(f"compression_ratio    : {repo.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- per-file CR ---")
        for rel, res in sorted(repo.files.items()):
            print(f"  {rel:40s} {res.compression_ratio * 100:6.2f}%  "
                  f"({res.orig_tokens} -> {res.comp_tokens})")
    return 0


def _single(args: argparse.Namespace) -> int:
    schemas = _resolve_filtered_schemas(args)
    res = auto_compress_file(schemas, args.input,
                             preprocess=args.preprocess,
                             match_mode=args.match_mode,
                             narrow_match=args.narrow_match,
                             score_variant=args.score_variant,
                             render_variant=args.render_variant)
    print(f"candidates_generated : {len(res.all_candidates)}")
    print(f"candidates_selected  : {len(res.selection.chosen)}")
    print(f"total_gain (DP)      : {res.selection.total_gain}")
    if args.preprocess and res.raw_tokens is not None:
        print(f"raw_tokens           : {res.raw_tokens}")
        print(f"preprocessed_tokens  : {res.orig_tokens}")
        print(f"comp_tokens          : {res.comp_tokens}")
        print(f"preprocess CR        : {res.preprocess_ratio * 100:.2f}%")
        print(f"codegist CR (marg.)  : {res.codegist_ratio * 100:.2f}%")
        print(f"stacked CR           : {res.compression_ratio * 100:.2f}%")
    else:
        print(f"orig_tokens          : {res.orig_tokens}")
        print(f"comp_tokens          : {res.comp_tokens}")
        print(f"compression_ratio    : {res.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- chosen rewrites ---")
        for c in res.selection.chosen:
            print(f"  [{c.span_kind:10s}] {c.schema_id} gain={c.gain}  "
                  f"-> {c.rewritten!r}")
    if args.out:
        Path(args.out).write_text(res.compressed, encoding="utf-8")
        print(f"\nwrote: {args.out}")
    else:
        print("\n=== compressed ===")
        print(res.compressed)
    return 0


def _batch(args: argparse.Namespace) -> int:
    if not args.out_dir:
        print("--out-dir required with --input-dir", file=sys.stderr)
        return 2
    schemas = _resolve_filtered_schemas(args)
    repo = auto_compress_repo(args.input_dir, schemas,
                              out_dir=args.out_dir,
                              preprocess=args.preprocess,
                              match_mode=args.match_mode,
                              narrow_match=args.narrow_match,
                              score_variant=args.score_variant,
                              render_variant=args.render_variant)
    print(f"files_compressed     : {len(repo.files)}")
    print(f"files_skipped        : {len(repo.skipped)}")
    if args.preprocess and repo.preprocessed_tokens is not None:
        print(f"raw_tokens (total)        : {repo.orig_tokens}")
        print(f"preprocessed_tokens (tot) : {repo.preprocessed_tokens}")
        print(f"comp_tokens (total)       : {repo.comp_tokens}")
        print(f"preprocess CR             : {repo.preprocess_ratio * 100:.2f}%")
        print(f"codegist CR (marginal)    : {repo.codegist_ratio * 100:.2f}%")
        print(f"stacked CR                : {repo.compression_ratio * 100:.2f}%")
    else:
        print(f"orig_tokens (total)  : {repo.orig_tokens}")
        print(f"comp_tokens (total)  : {repo.comp_tokens}")
        print(f"compression_ratio    : {repo.compression_ratio * 100:.2f}%")
    if args.report:
        print("\n--- per-file CR ---")
        for rel, res in sorted(repo.files.items()):
            print(f"  {rel:40s} {res.compression_ratio * 100:6.2f}%  "
                  f"({res.orig_tokens} -> {res.comp_tokens})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="auto-mined online compression")
    ap.add_argument("--schemas", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input")
    src.add_argument("--input-dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--preprocess", action="store_true",
                    help="apply preprocess pre-pass before compression")
    ap.add_argument(
        "--match-mode", choices=("auto", "transform", "fingerprint"),
        default="auto",
        help=("Python schema matching mode. In 'auto', built-in transforms use "
              "region type and their match predicates, broad LLM schemas use "
              "region type and any recorded root class, and narrow LLM schemas "
              "use --narrow-match. 'transform' looks up schemas by region type. "
              "'fingerprint' uses region type and exact fingerprint."),
    )
    ap.add_argument(
        "--narrow-match", choices=("fingerprint", "skeleton"),
        default="skeleton",
        help=("Match narrow Python LLM schemas in 'auto' mode. 'skeleton' "
              "(default) uses the schema's stored AST skeleton. "
              "'fingerprint' uses exact fingerprint matching."),
    )
    ap.add_argument(
        "--include-write-lib", action="store_true",
        help=("opt back into write_lib's hand-written transforms at "
              "compression time. Default behaviour loads ONLY the "
              "LLM-synthesised templates (transform_name starts with "
              "'llm_synth'). Useful only for paper-style ablation "
              "comparisons against the hand-rolled baseline."),
    )
    ap.add_argument(
        "--score-variant", choices=("default", "compact"), default="compact",
        help=("How the DP estimates token savings for each candidate. "
              "'compact' (default) scores each candidate's compact rendering "
              "in isolation. The compression ratio uses the rendered file."),
    )
    ap.add_argument(
        "--render-variant", choices=("default", "compact"), default="compact",
        help=("rendering style for Alg.3 replacement. 'compact' compacts only "
              "selected schema spans by default; untouched source outside "
              "selected spans is preserved."),
    )
    args = ap.parse_args()
    from codegist.lang import schema_language
    lang = schema_language(args.schemas)
    if lang == "java":
        if args.input_dir:
            return _batch_java(args)
        return _single_java(args)
    if lang == "typescript":
        if args.input_dir:
            return _batch_ts(args)
        return _single_ts(args)
    if args.input_dir:
        return _batch(args)
    return _single(args)


if __name__ == "__main__":
    raise SystemExit(main())
