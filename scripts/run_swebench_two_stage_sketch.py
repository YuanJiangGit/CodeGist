"""Two-stage SWE-bench generation: sketch file selection, then raw-file patch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CODEGIST_ROOT = os.environ.get("CODEGIST_ROOT", "")

sys.path.insert(0, str(SCRIPT_DIR))

from retrieval_context import (  # noqa: E402
    build_file_context,
    default_codegen_stats,
    doc_text,
    doc_title,
    import_codegist,
    load_json_or_jsonl,
    resolve_path,
    save_json,
)
from run_swebench_generation import (  # noqa: E402
    build_patch_instruction_block,
    clean_patch,
    count_tokens,
    env_value,
    load_cached_source,
    load_existing_json,
    patch_format_status,
    select_records,
)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
RETRYABLE_EXCEPTION_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "InternalServerError",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "Timeout",
    "TimeoutError",
}
RETRYABLE_TEXT_MARKERS = [
    "empty api response",
    "concurrency limit exceeded",
    "daily usage limit exceeded",
    "origin_response_timeout",
    "temporarily unavailable",
    "model is not supported when using codex with a chatgpt account",
    "the origin web server did not return",
    "timed out",
    "timeout occurred",
]


class EmptyAPIResponseError(RuntimeError):
    """The API returned no content and no usage metadata."""


def is_retryable_generation_error(error: str) -> bool:
    lowered = error.lower()
    if any(f"error code: {code}" in lowered for code in RETRYABLE_STATUS_CODES):
        return True
    if any(name.lower() in lowered for name in RETRYABLE_EXCEPTION_NAMES):
        return True
    return any(marker in lowered for marker in RETRYABLE_TEXT_MARKERS)


def is_retryable_exception(exc: Exception) -> bool:
    """Retry only API/network failures. Model/method failures count as failures."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    if type(exc).__name__ in RETRYABLE_EXCEPTION_NAMES:
        return True
    return is_retryable_generation_error(f"{type(exc).__name__}: {exc}")


def env_keys(primary: str, fallback: str) -> list[str]:
    return [primary, fallback] if primary != fallback else [primary]


def load_client(
    env_file: Path,
    timeout: float,
    api_key_env: str,
    base_url_env: str,
    model_env: str,
) -> tuple[OpenAI, str]:
    env = dotenv_values(env_file)
    model = env_value(env, *env_keys(model_env, "OPENAI_MODEL"), default="gpt-5.2")
    client = OpenAI(
        api_key=env_value(env, *env_keys(api_key_env, "OPENAI_API_KEY")),
        base_url=env_value(env, *env_keys(base_url_env, "OPENAI_BASE_URL")),
        timeout=timeout,
        max_retries=0,
    )
    return client, model


def available_file_paths(record: dict[str, Any], topk_docs: int) -> list[str]:
    paths: list[str] = []
    for idx, doc in enumerate((record.get("docs") or [])[:topk_docs]):
        path = str(doc.get("path") or doc.get("title") or doc.get("file") or doc_title(doc, idx))
        if path not in paths:
            paths.append(path)
    return paths


def stage1_context_note(context_mode: str) -> list[str]:
    if context_mode == "none":
        return [
            "No retrieved code context is provided.",
            "Infer the most likely source file path from the issue statement and repository conventions.",
            "Return a source file path that should exist in the repository, not a test file unless the issue specifically requires changing tests.",
        ]
    if context_mode == "sketch":
        return [
            "The following retrieved code context is a CodeGist semantic sketch version.",
            "It is only for semantic understanding of APIs, names, and control flow.",
            "It is not executable source code. Do not copy sketch text directly into a patch.",
            "Choose exactly one file from the retrieved candidate files listed in the sketch blocks.",
        ]
    if context_mode == "compressed_sketch":
        return [
            "The following retrieved code context is a precompressed semantic sketch version.",
            "It is only for semantic understanding of APIs, names, and control flow.",
            "It is not executable source code. Do not copy sketch text directly into a patch.",
            "Choose exactly one file from the retrieved candidate files listed in the sketch blocks.",
        ]
    if context_mode == "tokensugar":
        return [
            "The following retrieved code context is a TokenSugar-compressed reading view.",
            "A macro definition legend appears before the retrieved files and defines every shorthand used.",
            "Mentally expand those macros while reading the files.",
            "This is not executable source code. Do not copy TokenSugar shorthand into the patch.",
            "Choose exactly one file from the retrieved candidate files listed in the code blocks.",
        ]
    if context_mode == "raw":
        return [
            "The following retrieved code context is original source code from the BM25 top-5 files.",
            "Use it to understand APIs, names, control flow, and nearby implementation patterns.",
            "Choose exactly one file from the retrieved candidate files listed in the code blocks.",
        ]
    raise ValueError(f"Unsupported stage1 context mode: {context_mode}")


def build_stage1_prompt(
    record: dict[str, Any],
    stage1_context: str,
    topk_docs: int,
    context_mode: str,
) -> str:
    issue = record.get("problem_statement") or record.get("prompt") or ""
    repo = str(record.get("repo") or "").strip()
    paths = available_file_paths(record, topk_docs)
    if context_mode == "none":
        return "\n".join(
            [
                "You will be provided with an issue statement explaining a problem to resolve.",
                f"Repository: {repo}" if repo else "",
                "<issue>",
                str(issue).strip(),
                "</issue>",
                "",
                *stage1_context_note(context_mode),
                "",
                "Return only a strict JSON object with exactly these keys:",
                '{"file": "path/to/source_file.py", "plan": "brief implementation plan"}',
                "Do not include Markdown fences, comments, reasoning text, or extra keys.",
            ]
        ).strip()
    return "\n".join(
        [
            "You will be provided with a partial code base and an issue statement explaining a problem to resolve.",
            "<issue>",
            str(issue).strip(),
            "</issue>",
            "",
            "<code>",
            *stage1_context_note(context_mode),
            "",
            stage1_context.rstrip(),
            "</code>",
            "",
            "Candidate files:",
            "\n".join(f"- {path}" for path in paths),
            "",
            "Return only a strict JSON object with exactly these keys:",
            '{"file": "path/from/candidate/files.py", "plan": "brief implementation plan"}',
            "Do not include Markdown fences, comments, reasoning text, or extra keys.",
        ]
    ).strip()


def parse_stage1_json(text: str, candidates: list[str], require_candidate: bool = True) -> dict[str, str]:
    stripped = (text or "").strip()
    match = JSON_OBJECT_RE.search(stripped)
    if match:
        stripped = match.group(0)
    data = json.loads(stripped)
    file_path = str(data.get("file") or data.get("path") or "").strip()
    plan = str(data.get("plan") or "").strip()
    if require_candidate and file_path not in candidates:
        normalized = file_path.removeprefix("a/").removeprefix("b/")
        if normalized in candidates:
            file_path = normalized
        else:
            raise ValueError(f"stage1 file is not in retrieved candidates: {file_path}")
    else:
        file_path = file_path.removeprefix("a/").removeprefix("b/")
    if not file_path:
        raise ValueError("stage1 file is empty")
    if not plan:
        raise ValueError("stage1 plan is empty")
    return {"file": file_path, "plan": plan}


def build_stage2_prompt(record: dict[str, Any], selected_file: str, source: str, plan: str) -> str:
    issue = record.get("problem_statement") or record.get("prompt") or ""
    return "\n".join(
        [
            "You will be provided with a partial code base and an issue statement explaining a problem to resolve.",
            "<issue>",
            str(issue).strip(),
            "</issue>",
            "",
            "<selected_file>",
            f"[start of {selected_file}]",
            source.rstrip(),
            f"[end of {selected_file}]",
            "</selected_file>",
            "",
            "<plan>",
            plan.strip(),
            "</plan>",
            "",
            f"Only modify this selected file: {selected_file}",
            f"The patch must contain exactly one `diff --git a/{selected_file} b/{selected_file}` file block.",
            "Do not add or modify tests or any other files.",
            "",
            build_patch_instruction_block(),
        ]
    ).strip()


def call_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool = False,
    disable_thinking: bool = False,
) -> tuple[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if disable_thinking:
        request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if stream:
        chunks = client.chat.completions.create(
            **request_kwargs,
            stream=True,
        )
        parts: list[str] = []
        usage = None
        for chunk in chunks:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                parts.append(content)
        return "".join(parts), usage
    response = client.chat.completions.create(
        **request_kwargs,
    )
    content = response.choices[0].message.content or ""
    usage = response.usage
    if not content.strip() and usage is None:
        raise EmptyAPIResponseError("empty API response: no content and no usage metadata")
    return content, usage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--raw_output_path")
    parser.add_argument("--prompt_debug_path")
    parser.add_argument("--usage_path")
    parser.add_argument("--stats_path")
    parser.add_argument("--env_file", default=str(REPO_ROOT.parent / ".env"))
    parser.add_argument("--codegist_root", default=DEFAULT_CODEGIST_ROOT)
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument(
        "--stage1_context_mode",
        choices=["none", "sketch", "compressed_sketch", "tokensugar", "raw"],
        default="sketch",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--api_key_env", default="LLM_API_KEY")
    parser.add_argument("--base_url_env", default="LLM_MODEL_URL")
    parser.add_argument("--model_env", default="LLM_MODEL")
    parser.add_argument("--stage1_max_tokens", type=int, default=512)
    parser.add_argument("--stage2_max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help='Pass extra_body={"thinking":{"type":"disabled"}} to OpenAI-compatible APIs.',
    )
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--limit_start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance_ids", nargs="*")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--stream_stage2",
        action="store_true",
        help="Use streaming for stage2 patch generation to avoid proxy read timeouts.",
    )
    parser.add_argument(
        "--stream_stage1",
        action="store_true",
        help="Use streaming for stage1 file selection to avoid proxy read timeouts.",
    )
    parser.add_argument(
        "--retry_until_patch",
        action="store_true",
        help=(
            "Retry each instance only on API/network errors. Stage1 selection, JSON, "
            "source-loading, and patch-format failures are recorded once as failures."
        ),
    )
    parser.add_argument(
        "--allow_stage1_non_candidate",
        action="store_true",
        help=(
            "Allow the stage1 JSON file path to be outside the retrieved top-k candidates "
            "when the source can be loaded from the instance cache. This is intended for "
            "targeted empty-output backfill runs."
        ),
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=1,
        help="Maximum attempts per instance. Use 0 with --retry_until_patch for no fixed limit.",
    )
    parser.add_argument("--retry_sleep", type=float, default=120.0)
    args = parser.parse_args()

    data_file = resolve_path(args.data_file, REPO_ROOT)
    output_path = resolve_path(args.output_path, REPO_ROOT)
    raw_output_path = resolve_path(args.raw_output_path, REPO_ROOT) if args.raw_output_path else None
    prompt_debug_path = resolve_path(args.prompt_debug_path, REPO_ROOT) if args.prompt_debug_path else None
    usage_path = resolve_path(args.usage_path, REPO_ROOT) if args.usage_path else None
    stats_path = resolve_path(args.stats_path, REPO_ROOT) if args.stats_path else None
    env_file = resolve_path(args.env_file, REPO_ROOT)

    records = select_records(
        load_json_or_jsonl(data_file),
        args.limit_start,
        args.limit,
        args.instance_ids,
    )
    if not records:
        raise ValueError("No records selected")

    if args.overwrite:
        for path in (output_path, raw_output_path, prompt_debug_path, usage_path, stats_path):
            if path is not None and path.exists():
                path.unlink()

    predictions = load_existing_json(output_path)
    raw_outputs = load_existing_json(raw_output_path)
    prompt_debug = load_existing_json(prompt_debug_path)
    usage_records = load_existing_json(usage_path)

    cg_stats = default_codegen_stats()
    cg = None
    if args.stage1_context_mode == "sketch":
        cg = import_codegist(resolve_path(args.codegist_root, REPO_ROOT))
    client = None
    model = args.model or "gpt-5.2"
    if not args.dry_run:
        client, env_model = load_client(
            env_file,
            args.timeout,
            args.api_key_env,
            args.base_url_env,
            args.model_env,
        )
        model = args.model or env_model
    model_name_or_path = (
        args.model_name_or_path
        or f"{model}-bm25-{args.stage1_context_mode}-two-stage-top{args.topk_docs}"
    )

    stats: dict[str, Any] = {
        "selected_records": len(records),
        "topk_docs": args.topk_docs,
        "stage1_context_mode": args.stage1_context_mode,
        "codegist": cg_stats,
        "stage1_parse_errors": [],
        "generation_errors": [],
        "disable_thinking": args.disable_thinking,
    }

    for idx in range(len(predictions), len(records)):
        record = records[idx]
        instance_id = record["instance_id"]
        candidates = available_file_paths(record, args.topk_docs)
        if args.stage1_context_mode == "none":
            stage1_context = ""
        else:
            stage1_context = build_file_context(
                record,
                mode=args.stage1_context_mode,
                topk_docs=args.topk_docs,
                cg=cg,
                stats=cg_stats,
            )
        stage1_prompt = build_stage1_prompt(
            record,
            stage1_context,
            args.topk_docs,
            args.stage1_context_mode,
        )
        stage1_messages = [{"role": "user", "content": stage1_prompt}]
        stage1_prompt_tokens = count_tokens(stage1_messages)

        if args.dry_run:
            prompt_debug.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "stage1_messages": stage1_messages,
                    "stage1_prompt_tokens": stage1_prompt_tokens,
                    "stage1_context_mode": args.stage1_context_mode,
                    "candidate_files": candidates,
                }
            )
            print(f"dry-run two-stage prompt {idx + 1}/{len(records)} {instance_id}")
            continue

        assert client is not None
        attempt = 0
        attempt_errors: list[str] = []
        attempt_prompt_tokens_total_local = 0
        while True:
            attempt += 1
            stage1_text = ""
            stage2_text = ""
            stage1 = {"file": "", "plan": ""}
            stage1_usage = None
            stage2_usage = None
            stage2_messages = []
            stage2_prompt_tokens = None
            patch = ""
            patch_status = patch_format_status(patch)
            try:
                attempt_prompt_tokens_total_local += stage1_prompt_tokens or 0
                print(
                    f"requesting stage1 {instance_id} attempt={attempt}",
                    flush=True,
                )
                stage1_text, stage1_usage = call_chat(
                    client,
                    model,
                    stage1_messages,
                    args.stage1_max_tokens,
                    args.temperature,
                    args.top_p,
                    args.stream_stage1,
                    args.disable_thinking,
                )
                if not stage1_text.strip():
                    raise EmptyAPIResponseError("empty API response from stage1")
                stage1 = parse_stage1_json(
                    stage1_text,
                    candidates,
                    require_candidate=(
                        args.stage1_context_mode != "none"
                        and not args.allow_stage1_non_candidate
                    ),
                )
                source = load_cached_source(record, stage1["file"])
                if source is None:
                    raise ValueError(f"Cannot load selected source file: {stage1['file']}")

                stage2_prompt = build_stage2_prompt(record, stage1["file"], source, stage1["plan"])
                stage2_messages = [{"role": "user", "content": stage2_prompt}]
                stage2_prompt_tokens = count_tokens(stage2_messages)
                attempt_prompt_tokens_total_local += stage2_prompt_tokens or 0
                print(
                    f"requesting stage2 {instance_id} attempt={attempt} "
                    f"file={stage1['file']}",
                    flush=True,
                )
                stage2_text, stage2_usage = call_chat(
                    client,
                    model,
                    stage2_messages,
                    args.stage2_max_tokens,
                    args.temperature,
                    args.top_p,
                    args.stream_stage2,
                    args.disable_thinking,
                )
                if not stage2_text.strip():
                    raise EmptyAPIResponseError("empty API response from stage2")
                patch = clean_patch(stage2_text, record)
                patch_status = patch_format_status(patch)
                break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                attempt_errors.append(error)
                print(
                    f"two-stage attempt {attempt} failed for {instance_id}: {error}",
                    flush=True,
                )
                reached_max = args.max_attempts > 0 and attempt >= args.max_attempts
                retryable = is_retryable_exception(exc)
                if (
                    isinstance(exc, json.JSONDecodeError)
                    and stage1_text.strip().startswith("{")
                    and stage1_usage is None
                    and not stage2_text.strip()
                ):
                    retryable = True
                if not args.retry_until_patch or reached_max or not retryable:
                    if not args.continue_on_error:
                        raise
                    stats["generation_errors"].append(
                        {
                            "instance_id": instance_id,
                            "error": error,
                            "attempt": attempt,
                            "retryable": retryable,
                        }
                    )
                    break
                print(
                    f"retrying {instance_id} after {args.retry_sleep:g}s "
                    f"(attempt {attempt + 1})",
                    flush=True,
                )
                time.sleep(args.retry_sleep)

        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": patch,
                "patch_format_status": patch_status,
            }
        )
        if raw_output_path:
            raw_outputs.append(
                {
                    "instance_id": instance_id,
                    "stage1_raw_output": stage1_text,
                    "stage2_raw_output": stage2_text,
                    "stage1_selection": stage1,
                    "attempt": attempt,
                    "attempt_errors": attempt_errors,
                }
            )
        if prompt_debug_path:
            prompt_debug.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "candidate_files": candidates,
                    "stage1_context_mode": args.stage1_context_mode,
                    "stage1_messages": stage1_messages,
                    "stage1_raw_output": stage1_text,
                    "stage1_selection": stage1,
                    "attempt": attempt,
                    "attempt_errors": attempt_errors,
                    "stage2_messages": stage2_messages,
                    "stage2_raw_output": stage2_text,
                    "clean_patch": patch,
                    "patch_format_status": patch_status,
                    "stage1_prompt_tokens": stage1_prompt_tokens,
                    "stage2_prompt_tokens": stage2_prompt_tokens,
                }
            )
        if usage_path:
            usage_records.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "stage1_selection": stage1,
                    "stage1_api_usage": stage1_usage.model_dump() if stage1_usage is not None else None,
                    "stage2_api_usage": stage2_usage.model_dump() if stage2_usage is not None else None,
                    "stage1_prompt_tokens": stage1_prompt_tokens,
                    "stage2_prompt_tokens": stage2_prompt_tokens,
                    "attempt": attempt,
                    "attempt_errors": attempt_errors,
                    "attempt_prompt_tokens_total_local": attempt_prompt_tokens_total_local,
                    "stage1_raw_chars": len(stage1_text),
                    "stage2_raw_chars": len(stage2_text),
                    "patch_chars": len(patch),
                    "patch_format_status": patch_status,
                    "stage1_context_mode": args.stage1_context_mode,
                }
            )

        print(
            f"generated {len(predictions)}/{len(records)} {instance_id} "
            f"file={stage1.get('file')} "
            f"attempt={attempt} "
            f"stage1_tokens={stage1_prompt_tokens} stage2_tokens={stage2_prompt_tokens} "
            f"patch_shape={patch_status['shape']} "
            f"likely_harness_compatible={patch_status['likely_harness_compatible']}",
            flush=True,
        )

        if len(predictions) % args.save_every == 0:
            save_json(output_path, predictions)
            save_json(raw_output_path, raw_outputs)
            save_json(prompt_debug_path, prompt_debug)
            save_json(usage_path, usage_records)
            save_json(stats_path, stats)

    if args.dry_run:
        save_json(prompt_debug_path or output_path, prompt_debug)
    else:
        save_json(output_path, predictions)
        save_json(raw_output_path, raw_outputs)
        save_json(usage_path, usage_records)
    save_json(prompt_debug_path, prompt_debug)
    save_json(stats_path, stats)
    print(f"done: selected={len(records)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
