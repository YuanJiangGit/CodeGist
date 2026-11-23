"""CLI for the offline schema mining pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the project root importable when invoked as `python -m scripts.mine`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codegist.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="CodeGist offline schema mining")
    ap.add_argument("--corpus", required=True,
                    help="directory of .py files to mine")
    ap.add_argument("--out", default="out/schema_set.json",
                    help="path for the validated schema set JSON")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip Step 7 (LLM semantic validation) — for "
                         "offline smoke tests without API credentials")
    ap.add_argument("--preprocess", action="store_true",
                    help="run preprocess_python.preprocess() on each "
                         "corpus file before extraction (Formatting "
                         "Removal pre-pass; OFF by default)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="PATTERN",
                    help="extra exclude glob patterns")
    ap.add_argument("--no-default-excludes", action="store_true",
                    help="disable the always-on default exclude list")
    ap.add_argument("--use-recommended-excludes", action="store_true",
                    help="also skip tests/, docs/, vendor/, build/, ...")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(name)s | %(message)s",
    )

    from codegist._excludes import RECOMMENDED_EXCLUDES, resolve
    excludes = resolve(args.exclude, use_defaults=not args.no_default_excludes)
    if args.use_recommended_excludes:
        excludes.extend(RECOMMENDED_EXCLUDES)

    summary = run(
        corpus_dir=Path(args.corpus),
        out_path=Path(args.out),
        skip_llm=args.skip_llm,
        preprocess=args.preprocess,
        exclude=excludes,
    )["summary"]
    print("\n=== pipeline summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
