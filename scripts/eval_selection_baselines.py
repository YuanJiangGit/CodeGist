"""Compare DP, greedy, and random rewrite selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codegist.auto_mining.online import generate_candidates, load_schemas  # noqa: E402
from codegist.candidate_gen import Candidate, count_tokens  # noqa: E402
from codegist.compress import _apply  # noqa: E402
from codegist.selection import _greedy_select, select  # noqa: E402


_SCHEMAS = None
_ARGS: dict[str, Any] = {}


def _load_filtered_schemas(schema_path: Path, include_write_lib: bool):
    schemas = load_schemas(schema_path)
    if include_write_lib:
        return schemas
    return [s for s in schemas if s.transform_name.startswith("llm_synth")]


def _init_worker(
    schema_path: str,
    include_write_lib: bool,
    match_mode: str,
    narrow_match: str,
    score_variant: str,
    render_variant: str,
    input_dir: str,
    out_dir: str | None,
    random_seed: int,
    random_seed_policy: str,
) -> None:
    global _SCHEMAS, _ARGS
    _SCHEMAS = _load_filtered_schemas(Path(schema_path), include_write_lib)
    _ARGS = {
        "match_mode": match_mode,
        "narrow_match": narrow_match,
        "score_variant": score_variant,
        "render_variant": render_variant,
        "input_dir": input_dir,
        "out_dir": out_dir,
        "random_seed": random_seed,
        "random_seed_policy": random_seed_policy,
    }


def _stable_file_seed(seed: int, rel: str) -> int:
    digest = hashlib.sha256(rel.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "big")


def _random_select(
    candidates: list[Candidate],
    rel: str,
    seed: int,
    seed_policy: str,
) -> list[Candidate]:
    if seed_policy == "constant":
        rng = random.Random(seed)
    elif seed_policy == "per-file":
        rng = random.Random(_stable_file_seed(seed, rel))
    else:
        raise ValueError(f"unknown random seed policy: {seed_policy}")
    pool = [c for c in candidates if c.gain > 0]
    rng.shuffle(pool)
    chosen: list[Candidate] = []
    taken: list[tuple[int, int]] = []
    for c in pool:
        s, e = c.start_offset, c.end_offset
        if all(e <= ts or te <= s for ts, te in taken):
            chosen.append(c)
            taken.append((s, e))
    return chosen


def _write_variant(out_dir: str | None, variant: str, rel: str, text: str) -> str | None:
    if not out_dir:
        return None
    out_path = Path(out_dir) / variant / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return str(out_path)


def _eval_one(path_s: str) -> dict[str, Any]:
    assert _SCHEMAS is not None
    input_dir = Path(_ARGS["input_dir"])
    path = Path(path_s)
    rel = path.relative_to(input_dir).as_posix()
    try:
        src = path.read_text(encoding="utf-8")
        tree, candidates = generate_candidates(
            src,
            str(path),
            _SCHEMAS,
            match_mode=_ARGS["match_mode"],
            narrow_match=_ARGS["narrow_match"],
            score_variant=_ARGS["score_variant"],
        )
    except SyntaxError as exc:
        return {"rel": rel, "skipped": f"SyntaxError: {exc}"}
    except OSError as exc:
        return {"rel": rel, "skipped": f"OSError: {exc}"}

    dp_result = select(tree, candidates)
    greedy_chosen = _greedy_select(candidates)
    random_chosen = _random_select(
        candidates,
        rel,
        int(_ARGS["random_seed"]),
        str(_ARGS["random_seed_policy"]),
    )

    render_variant = _ARGS["render_variant"]
    dp_text = _apply(src, dp_result.chosen, render_variant=render_variant)
    greedy_text = _apply(src, greedy_chosen, render_variant=render_variant)
    random_text = _apply(src, random_chosen, render_variant=render_variant)

    orig_tokens = count_tokens(src)
    out_dir = _ARGS["out_dir"]
    return {
        "rel": rel,
        "path": str(path),
        "hash": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "orig_tokens": orig_tokens,
        "dp_tokens": count_tokens(dp_text),
        "greedy_tokens": count_tokens(greedy_text),
        "random_tokens": count_tokens(random_text),
        "candidates": len(candidates),
        "dp_selected": len(dp_result.chosen),
        "greedy_selected": len(greedy_chosen),
        "random_selected": len(random_chosen),
        "dp_output": _write_variant(out_dir, "dp", rel, dp_text),
        "greedy_output": _write_variant(out_dir, "greedy", rel, greedy_text),
        "random_output": _write_variant(
            out_dir,
            f"random_seed{_ARGS['random_seed']}",
            rel,
            random_text,
        ),
    }


def _cr(orig: int, comp: int) -> float:
    return 0.0 if orig <= 0 else 1.0 - comp / orig


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "n": len(rows),
        "orig_tokens": sum(r["orig_tokens"] for r in rows),
        "dp_tokens": sum(r["dp_tokens"] for r in rows),
        "greedy_tokens": sum(r["greedy_tokens"] for r in rows),
        "random_tokens": sum(r["random_tokens"] for r in rows),
    }
    orig = totals["orig_tokens"]
    totals.update({
        "dp_cr": _cr(orig, totals["dp_tokens"]),
        "greedy_cr": _cr(orig, totals["greedy_tokens"]),
        "random_cr_seed42": _cr(orig, totals["random_tokens"]),
        "dp_minus_greedy_cr": _cr(orig, totals["dp_tokens"]) - _cr(orig, totals["greedy_tokens"]),
        "dp_minus_random_cr": _cr(orig, totals["dp_tokens"]) - _cr(orig, totals["random_tokens"]),
    })
    return totals


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="selection-strategy ablation: DP vs greedy vs random",
    )
    ap.add_argument("--schemas", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="optional directory for materialized dp/greedy/random outputs")
    ap.add_argument("--stats", default="out/selection_baselines_stats.json")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--random-seed-policy", choices=("constant", "per-file"),
                    default="constant",
                    help=("random baseline seeding. 'constant' reuses the same "
                          "seed for every file and matches the reported "
                          "RepoEval selection-ablation setup; 'per-file' "
                          "derives a stable seed from the relative path."))
    ap.add_argument("--include-write-lib", action="store_true",
                    help="include write_lib transforms; default keeps llm_synth schemas only")
    ap.add_argument("--match-mode", choices=("auto", "transform", "fingerprint"),
                    default="auto")
    ap.add_argument("--narrow-match", choices=("fingerprint", "skeleton"),
                    default="skeleton")
    ap.add_argument("--score-variant", choices=("default", "compact"),
                    default="compact")
    ap.add_argument("--render-variant", choices=("default", "compact"),
                    default="compact")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(str(p) for p in input_dir.rglob("*.py"))
    if not files:
        print(f"no Python files found under {input_dir}", file=sys.stderr)
        return 2

    t0 = time.time()
    schema_count = len(_load_filtered_schemas(Path(args.schemas), args.include_write_lib))
    print(f"[schemas] {args.schemas} loaded={schema_count}")
    print(f"[files] {len(files)} workers={args.workers} random_seed={args.random_seed}")

    init_args = (
        args.schemas,
        args.include_write_lib,
        args.match_mode,
        args.narrow_match,
        args.score_variant,
        args.render_variant,
        str(input_dir),
        args.out_dir,
        args.random_seed,
        args.random_seed_policy,
    )
    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    if args.workers <= 1:
        _init_worker(*init_args)
        for i, path_s in enumerate(files, start=1):
            row = _eval_one(path_s)
            (skipped if "skipped" in row else rows).append(row)
            if i % 100 == 0 or i == len(files):
                print(f"...{i}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=init_args,
        ) as ex:
            futs = [ex.submit(_eval_one, p) for p in files]
            for i, fut in enumerate(as_completed(futs), start=1):
                row = fut.result()
                (skipped if "skipped" in row else rows).append(row)
                if i % 100 == 0 or i == len(futs):
                    print(f"...{i}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)

    result = {
        "method": "selection baselines: CodeGist DP vs greedy vs random",
        "schema": str(Path(args.schemas)),
        "input_dir": str(input_dir),
        "output_dir": args.out_dir,
        "settings": {
            "match_mode": args.match_mode,
            "narrow_match": args.narrow_match,
            "score_variant": args.score_variant,
            "render_variant": args.render_variant,
            "include_write_lib": args.include_write_lib,
            "random_seed": args.random_seed,
            "random_seed_policy": args.random_seed_policy,
        },
        "skipped": skipped,
        "summary": _totals(rows),
        "file_stats": sorted(rows, key=lambda r: r["rel"]),
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
