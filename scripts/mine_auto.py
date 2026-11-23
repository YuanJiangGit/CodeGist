"""Mine schemas from a Python, Java, or TypeScript corpus."""
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CodeGist auto-mining (generic Algorithm 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exclude pattern syntax (subset):\n"
            "  **/<name>/**  skip any path whose components include <name>\n"
            "                e.g. '**/tests/**', '**/__pycache__/**'\n"
            "  **/<glob>     filename glob on basename\n"
            "                e.g. '**/test_*.py', '**/conftest.py'\n"
            "  (anything else)  fnmatch against the full relative path\n"
            "\n"
            "Defaults always include: __pycache__, .git, .tox, .venv,\n"
            "venv, node_modules, .mypy_cache, .pytest_cache, .ruff_cache.\n"
            "Disable with --no-default-excludes.\n"
            "\n"
            "--use-recommended-excludes additionally skips tests/, docs/,\n"
            "examples/, vendor/, build/, migrations/, etc. — enable when\n"
            "mining real repos so test-only patterns don't pollute the\n"
            "schema set."
        ),
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="out/schema_set_auto.json")
    ap.add_argument("--lang", choices=("python", "java", "typescript"),
                    default="python",
                    help="corpus language. 'python' (default) is the original "
                         "pipeline, byte-for-byte unchanged. 'java' / "
                         "'typescript' mine *.java / *.ts(x) via the tree-sitter "
                         "frontends (codegist/lang/<lang>/); the language is "
                         "recorded in the schema JSON header so the online side "
                         "picks the right matcher automatically.")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--layered", action="store_true",
                    help="log how layered function templates are mined (Python)")
    ap.add_argument("--preprocess", action="store_true",
                    help="preprocess each corpus file before extraction")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="PATTERN",
                    help="extra exclude glob patterns")
    ap.add_argument("--no-default-excludes", action="store_true",
                    help="disable the always-on default exclude list")
    ap.add_argument("--use-recommended-excludes", action="store_true",
                    help=f"also exclude common research-noise dirs "
                         f"({', '.join(RECOMMENDED_EXCLUDES[:4])}, ...)")
    ap.add_argument("--llm-synth", action="store_true",
                    help="enable LLM-driven Write synthesis (paper-faithful "
                         "INFERWRITES with SynthLLM ↔ CriticLLM dialogue, up "
                         "to 2 rounds; produces per-bucket templates from a "
                         "controlled slot vocabulary)")
    ap.add_argument("--llm-synth-budget", type=int, default=100,
                    help="max buckets sent through LLM synth PER region_type, "
                         "now ranked by TOKEN VALUE (support × mean source "
                         "tokens ≈ corpus-wide tokens the pattern occupies), "
                         "NOT raw frequency — so the budget buys the highest-"
                         "impact buckets first instead of tiny high-count "
                         "idioms that compress little. Total ceiling = "
                         "(#supported region_types) × budget. Default 100; "
                         "recommend ~800 for a full re-mine (value-ranking "
                         "front-loads the high-value tail, so you no longer "
                         "need huge counts). Cost scales ~linearly (≈4-6 LLM "
                         "calls per bucket).")
    ap.add_argument("--llm-synth-keep", type=int, default=100,
                    help="after token validation, keep only the top-N NARROW "
                         "llm_synth schemas PER region_type, ranked by "
                         "token_total_gain (mean_gain × bucket support ≈ "
                         "corpus-wide tokens saved). Default 100. BROAD "
                         "llm_synth and write_lib schemas are exempt. Set 0 "
                         "to disable the cut (keep every survivor). Spends the "
                         "schema budget on high-value patterns and prunes the "
                         "low-gain tail before the LLM Step-8 budget is spent "
                         "on it.")
    ap.add_argument("--llm-synth-concurrency", type=int, default=5,
                    help="number of buckets processed CONCURRENTLY through the "
                         "SynthLLM↔CriticLLM loop (and Step-8 semantic "
                         "validation). Peak in-flight LLM requests == this "
                         "value, so set it to your endpoint's max-concurrency "
                         "limit (e.g. 5). Default 5; set 1 for fully serial.")
    ap.add_argument("--llm-val-budget", type=int, default=None,
                    help="total LLM-call budget for Step 7 semantic "
                         "validation (default = Config.llm_max_validations "
                         "= 50000; each schema consumes 2 calls, so cap on "
                         "validated schemas is budget/2). Tighten if endpoint "
                         "cost matters; raise if you have very large "
                         "corpus.")
    ap.add_argument("--gain-threshold", type=float, default=None,
                    help="minimum mean token gain per rewrite for a "
                         "schema to pass Step 5 token validation "
                         "(default = Config.gain_threshold = 0.3). "
                         "Lower → more schemas pass; higher → only "
                         "templates with large per-instance savings "
                         "survive.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(name)s | %(message)s",
    )

    excludes = resolve(args.exclude, use_defaults=not args.no_default_excludes)
    if args.use_recommended_excludes:
        excludes.extend(RECOMMENDED_EXCLUDES)

    from codegist.config import load as load_config
    cfg = load_config(llm_max_validations=args.llm_val_budget,
                      gain_threshold=args.gain_threshold)

    if args.lang == "java":
        from codegist.lang.java.mining import run_java
        summary = run_java(
            corpus_dir=Path(args.corpus),
            out_path=Path(args.out),
            cfg=cfg,
            skip_llm=args.skip_llm,
            exclude=excludes,
            llm_synth_budget=args.llm_synth_budget,
            llm_synth_keep=args.llm_synth_keep,
            synth_concurrency=args.llm_synth_concurrency,
        )["summary"]
        print("\n=== auto-mining summary (java) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    if args.lang == "typescript":
        from codegist.lang.typescript.mining import run_ts
        summary = run_ts(
            corpus_dir=Path(args.corpus),
            out_path=Path(args.out),
            cfg=cfg,
            skip_llm=args.skip_llm,
            exclude=excludes,
            llm_synth_budget=args.llm_synth_budget,
            llm_synth_keep=args.llm_synth_keep,
            synth_concurrency=args.llm_synth_concurrency,
        )["summary"]
        print("\n=== auto-mining summary (typescript) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    summary = run(
        corpus_dir=Path(args.corpus),
        out_path=Path(args.out),
        cfg=cfg,
        skip_llm=args.skip_llm,
        preprocess=args.preprocess,
        exclude=excludes,
        use_llm_synth=args.llm_synth,
        llm_synth_budget=args.llm_synth_budget,
        llm_synth_keep=args.llm_synth_keep,
        layered=args.layered,
        synth_concurrency=args.llm_synth_concurrency,
    )["summary"]
    print("\n=== auto-mining summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
