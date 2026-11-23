"""Compress Python source with baseline schemas."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codegist.compress import compress_file, compress_repo  # noqa: E402


def _single(args: argparse.Namespace) -> int:
    res = compress_file(args.schemas, args.input, preprocess=args.preprocess)
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
                  f"@offset[{c.start_offset}:{c.end_offset}]  -> {c.rewritten!r}")
        print("\n--- skipped (blocked by a better choice) ---")
        for c in res.selection.skipped:
            print(f"  [{c.span_kind:10s}] {c.schema_id} gain={c.gain}  "
                  f"@offset[{c.start_offset}:{c.end_offset}]")

    if args.out:
        Path(args.out).write_text(res.compressed, encoding="utf-8")
        print(f"\nwrote: {args.out}")
    else:
        print("\n=== compressed ===")
        print(res.compressed)
    return 0


def _batch(args: argparse.Namespace) -> int:
    if not args.out_dir:
        print("--out-dir is required with --input-dir", file=sys.stderr)
        return 2
    repo = compress_repo(args.input_dir, args.schemas, out_dir=args.out_dir,
                         preprocess=args.preprocess)
    n = len(repo.files)
    print(f"files_compressed     : {n}")
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
        if repo.skipped:
            print("\n--- skipped ---")
            for rel in repo.skipped:
                print(f"  {rel}")
    print(f"\nwrote {n} files to: {args.out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CodeGist online compression — Algorithm 2 + Algorithm 3"
    )
    ap.add_argument("--schemas", required=True,
                    help="path to validated schema_set.json from `mine`")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="single .py source file to compress")
    src.add_argument("--input-dir",
                     help="directory of .py files to batch-compress")
    ap.add_argument("--out", default=None,
                    help="(single mode) output path; default: stdout")
    ap.add_argument("--out-dir", default=None,
                    help="(batch mode) output directory; mirrors input tree")
    ap.add_argument("--report", action="store_true",
                    help="print per-candidate / per-file selection report")
    ap.add_argument("--preprocess", action="store_true",
                    help="run preprocess_python.preprocess() on input "
                         "before compression (Formatting Removal pre-"
                         "pass); CR breakdown reports preprocess vs "
                         "codegist contributions separately")
    args = ap.parse_args()

    if args.input_dir:
        return _batch(args)
    return _single(args)


if __name__ == "__main__":
    raise SystemExit(main())
