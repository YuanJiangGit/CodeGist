"""Rebuild schemas from a mining log using the original corpus and options."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codegist._excludes import RECOMMENDED_EXCLUDES, resolve  # noqa: E402
from codegist.auto_mining.pipeline import run  # noqa: E402
from codegist.auto_mining.resume import parse_synth_log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a schema set from a crashed mine's log "
                    "(zero LLM — Step 4b templates recovered from the log, "
                    "deterministic steps re-run, Step 8 skipped).",
    )
    ap.add_argument("--log", required=True,
                    help="the crashed run's .err.log (contains 'llm-synth OK' "
                         "lines)")
    ap.add_argument("--corpus", required=True,
                    help="SAME corpus dir as the crashed run")
    ap.add_argument("--out", default="out/schema_resumed.json")
    ap.add_argument("--preprocess", action="store_true",
                    help="MUST match the crashed run (it was OFF for the "
                         "boiler_real mine)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="PATTERN")
    ap.add_argument("--no-default-excludes", action="store_true")
    ap.add_argument("--use-recommended-excludes", action="store_true",
                    help="match the crashed run (it used this — 22 patterns)")
    ap.add_argument("--llm-synth-keep", type=int, default=0,
                    help="top-N NARROW llm_synth per region_type by "
                         "token_total_gain. Default 0 = KEEP ALL recovered "
                         "(measure the full batch's CR ceiling). Set 100 to "
                         "reproduce the crashed run's exact gating.")
    ap.add_argument("--gain-threshold", type=float, default=None,
                    help="Step-5 min mean token gain (default Config 0.3).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(name)s | %(message)s",
    )

    cache = parse_synth_log(args.log)
    broad = sum(1 for v in cache.values() if v.get("applies_broadly"))
    print(f"recovered {len(cache)} synth templates from {args.log} "
          f"({broad} BROAD, {len(cache) - broad} NARROW)")
    if not cache:
        print("ERROR: no 'llm-synth OK' lines parsed — wrong log file?",
              file=sys.stderr)
        return 2

    excludes = resolve(args.exclude, use_defaults=not args.no_default_excludes)
    if args.use_recommended_excludes:
        excludes.extend(RECOMMENDED_EXCLUDES)

    from codegist.config import load as load_config
    cfg = load_config(gain_threshold=args.gain_threshold)

    summary = run(
        corpus_dir=Path(args.corpus),
        out_path=Path(args.out),
        cfg=cfg,
        skip_llm=True,            # Step 8 skipped — supplement it separately
        preprocess=args.preprocess,
        exclude=excludes,
        use_llm_synth=True,
        llm_synth_keep=args.llm_synth_keep,
        synth_cache=cache,        # inject recovered templates, skip Step 4b
    )["summary"]
    print("\n=== resume summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {summary.get('validated_schemas', '?')} schemas to "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
