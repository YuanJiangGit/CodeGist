"""Generate SWE-bench Lite predictions through an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import difflib
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
    context_note,
    default_codegen_stats,
    import_codegist,
    load_json_or_jsonl,
    resolve_path,
    save_json,
)


PATCH_TAG_RE = re.compile(r"<patch>\s*(.*?)\s*</patch>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:diff|patch|text)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
APPLY_PATCH_HEADER_RE = re.compile(r"^\*\*\* (Update|Add|Delete) File: (.+)$")
DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

PATCH_EXAMPLE = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points"""


def env_value(values: dict[str, str | None], *keys: str, default: str | None = None) -> str:
    for key in keys:
        value = values.get(key) or os.environ.get(key)
        if value:
            return value.strip().strip('"').strip("'")
    if default is not None:
        return default
    raise ValueError(f"Missing one of {', '.join(keys)}")


def env_keys(primary: str, fallback: str) -> list[str]:
    return [primary, fallback] if primary != fallback else [primary]


def load_existing_json(path: Path | None) -> list[Any]:
    if path is None or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_patch(text: str) -> bool:
    stripped = text.lstrip()
    return (
        stripped.startswith("diff --git ")
        or stripped.startswith("--- a/")
        or "\ndiff --git " in text
        or "\n--- a/" in text
    )


def patch_format_status(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        shape = "empty"
    elif stripped.startswith("*** Begin Patch"):
        shape = "apply_patch"
    elif stripped.startswith("diff --git "):
        shape = "git_diff"
    elif stripped.startswith("--- a/"):
        shape = "unified_no_git_header"
    elif "diff --git " in stripped or "\n--- a/" in stripped:
        shape = "embedded_diff"
    else:
        shape = "other"

    hunk_headers = [
        line for line in stripped.splitlines() if line.startswith("@@")
    ]
    malformed_hunks = [
        line for line in hunk_headers if not HUNK_HEADER_RE.match(line)
    ]
    return {
        "shape": shape,
        "has_git_diff_header": stripped.startswith("diff --git "),
        "has_apply_patch_markers": "*** Begin Patch" in stripped or "*** End Patch" in stripped,
        "has_markdown_fence": "```" in stripped,
        "hunk_count": len(hunk_headers),
        "malformed_hunk_count": len(malformed_hunks),
        "likely_harness_compatible": (
            shape == "git_diff"
            and not malformed_hunks
            and "*** Begin Patch" not in stripped
            and "```" not in stripped
        ),
    }


def build_patch_instruction_block() -> str:
    example_explanation = (
        "Here is an example of a patch file. It consists of changes to the code base. "
        "It specifies the file names, the line numbers of each change, and the removed and added lines. "
        "A single patch file can contain changes to multiple files. "
        "The patch must be a git unified diff that starts each changed file with "
        "`diff --git a/path b/path`, followed by `--- a/path` and `+++ b/path` headers."
    )
    final_instruction = (
        "I need you to solve the provided issue by generating a single patch file that I can apply "
        "directly to this repository using git apply. Output only the patch file in the exact "
        "git unified diff format shown above. Do not output Markdown fences, explanations, "
        "or apply_patch/Codex patch blocks such as `*** Begin Patch`, `*** Update File`, "
        "or `*** End Patch`."
    )
    return "\n".join(
        [
            example_explanation,
            "<patch>",
            PATCH_EXAMPLE,
            "</patch>",
            "",
            final_instruction,
            "Respond below:",
        ]
    )


def normalize_patch_path(path: str) -> str:
    path = path.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def load_cached_source(record: dict[str, Any], file_path: str) -> str | None:
    file_path = normalize_patch_path(file_path)
    for doc in record.get("docs") or []:
        doc_path = doc.get("path") or doc.get("title") or doc.get("file")
        if normalize_patch_path(str(doc_path or "")) == file_path:
            text = doc.get("text")
            if text is None:
                text = doc.get("contents")
            return "" if text is None else str(text)

    cache_path = record.get("retrieval_metadata", {}).get("cache_path")
    if not cache_path:
        return None
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if normalize_patch_path(str(item.get("path", ""))) == file_path:
                return str(item.get("text", ""))
    return None


def parse_apply_patch(text: str) -> list[dict[str, Any]]:
    lines = text.strip().splitlines()
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        if line in {"*** Begin Patch", "*** End Patch"}:
            continue
        header = APPLY_PATCH_HEADER_RE.match(line)
        if header:
            if current is not None:
                blocks.append(current)
            current = {
                "action": header.group(1).lower(),
                "path": normalize_patch_path(header.group(2)),
                "lines": [],
            }
            continue
        if current is not None:
            if line.startswith("*** "):
                continue
            current["lines"].append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def split_apply_patch_hunks(lines: list[str]) -> list[list[str]]:
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        if line.startswith((" ", "+", "-")):
            current.append(line)
    if current:
        hunks.append(current)
    return hunks


def has_hunk_change(hunk: list[str]) -> bool:
    return any(line.startswith(("+", "-")) for line in hunk)


def apply_hunk(lines: list[str], hunk: list[str]) -> list[str]:
    if not has_hunk_change(hunk):
        return lines

    old = [line[1:] for line in hunk if line.startswith((" ", "-"))]
    new = [line[1:] for line in hunk if line.startswith((" ", "+"))]
    if not old:
        return lines + new

    for start in range(0, len(lines) - len(old) + 1):
        if lines[start : start + len(old)] == old:
            return lines[:start] + new + lines[start + len(old) :]

    compact_old = "\n".join(old).strip()
    for start in range(0, len(lines) - len(old) + 1):
        if "\n".join(lines[start : start + len(old)]).strip() == compact_old:
            return lines[:start] + new + lines[start + len(old) :]

    deleted = [line[1:] for line in hunk if line.startswith("-")]
    added = [line[1:] for line in hunk if line.startswith("+")]
    if deleted:
        matches = []
        for start in range(0, len(lines) - len(deleted) + 1):
            if lines[start : start + len(deleted)] == deleted:
                matches.append(start)
        if len(matches) == 1:
            start = matches[0]
            return lines[:start] + added + lines[start + len(deleted) :]

    raise ValueError("Could not locate apply_patch hunk in source file")


def apply_update_patch(original: str, patch_lines: list[str]) -> str:
    current = original.splitlines()
    for hunk in split_apply_patch_hunks(patch_lines):
        current = apply_hunk(current, hunk)
    trailing_newline = "\n" if original.endswith("\n") else ""
    return "\n".join(current) + trailing_newline


def apply_patch_to_unified_diff(text: str, record: dict[str, Any]) -> str:
    blocks = parse_apply_patch(text)
    if not blocks:
        return text.strip()

    diff_parts: list[str] = []
    for block in blocks:
        action = block["action"]
        path = block["path"]
        if action == "add":
            original = ""
            patched = "\n".join(line[1:] for line in block["lines"] if line.startswith("+")) + "\n"
        elif action == "delete":
            original = load_cached_source(record, path)
            if original is None:
                raise ValueError(f"Cannot find source for deleted file {path}")
            patched = ""
        else:
            original = load_cached_source(record, path)
            if original is None:
                raise ValueError(f"Cannot find source for updated file {path}")
            patched = apply_update_patch(original, block["lines"])

        from_lines = original.splitlines()
        to_lines = patched.splitlines()
        diff = list(
            difflib.unified_diff(
                from_lines,
                to_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        if diff:
            diff_parts.append(f"diff --git a/{path} b/{path}\n" + "\n".join(diff))
    return "\n".join(part.rstrip() for part in diff_parts).strip()


def split_diff_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = DIFF_HEADER_RE.match(line)
        if match:
            if current_path is not None:
                blocks.append((current_path, "\n".join(current_lines)))
            current_path = normalize_patch_path(match.group(2))
            current_lines = [line]
            continue
        if current_path is not None:
            current_lines.append(line)
    if current_path is not None:
        blocks.append((current_path, "\n".join(current_lines)))
    return blocks


def recover_hunks_from_unified_block(block: str) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in block.splitlines():
        header = HUNK_HEADER_RE.match(line)
        if header:
            if current is not None:
                hunks.append(current)
            current = {
                "old_start": int(header.group(1)),
                "old_count": int(header.group(2) or "1"),
                "lines": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("\\ No newline"):
            continue
        if line.startswith((" ", "+", "-")):
            current["lines"].append(line)
        elif line == "":
            current["lines"].append(" ")
        else:
            current["lines"].append(" " + line)
    if current is not None:
        hunks.append(current)
    return hunks


def rebuild_file_from_recovered_hunks(original: str, hunks: list[dict[str, Any]]) -> str:
    current = original.splitlines()
    for hunk in hunks:
        # Match hunks by context because model-generated offsets can be wrong.
        current = apply_hunk(current, hunk["lines"])
    trailing_newline = "\n" if original.endswith("\n") else ""
    return "\n".join(current) + trailing_newline


def regenerate_unified_diff(original: str, patched: str, path: str) -> str:
    diff = list(
        difflib.unified_diff(
            original.splitlines(),
            patched.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff:
        return ""
    return f"diff --git a/{path} b/{path}\n" + "\n".join(diff)


def repair_unified_diff_against_sources(text: str, record: dict[str, Any]) -> str:
    blocks = split_diff_blocks(text)
    if not blocks:
        return text.strip()

    repaired_parts: list[str] = []
    repaired_any = False
    for path, block in blocks:
        original = load_cached_source(record, path)
        if original is None:
            repaired_parts.append(block.strip())
            continue
        hunks = recover_hunks_from_unified_block(block)
        if not hunks:
            repaired_parts.append(block.strip())
            continue
        patched = rebuild_file_from_recovered_hunks(original, hunks)
        repaired = regenerate_unified_diff(original, patched, path)
        repaired_parts.append(repaired or block.strip())
        repaired_any = True
    return "\n".join(part for part in repaired_parts if part).strip() if repaired_any else text.strip()


def unified_diff_hunk_counts_valid(text: str) -> bool:
    """Return True when a git diff has well-formed hunks and matching line counts."""
    lines = text.splitlines()
    saw_hunk = False
    idx = 0
    while idx < len(lines):
        header = HUNK_HEADER_RE.match(lines[idx])
        if not header:
            idx += 1
            continue

        saw_hunk = True
        old_expected = int(header.group(2) or "1")
        new_expected = int(header.group(4) or "1")
        old_seen = 0
        new_seen = 0
        idx += 1

        while idx < len(lines):
            line = lines[idx]
            if HUNK_HEADER_RE.match(line) or DIFF_HEADER_RE.match(line):
                break
            if line.startswith("\\ No newline"):
                idx += 1
                continue
            if not line:
                return False
            prefix = line[:1]
            if prefix == " ":
                old_seen += 1
                new_seen += 1
            elif prefix == "-":
                old_seen += 1
            elif prefix == "+":
                new_seen += 1
            else:
                return False
            idx += 1

        if old_seen != old_expected or new_seen != new_expected:
            return False

    return saw_hunk


def clean_patch(text: str, record: dict[str, Any] | None = None) -> str:
    text = (text or "").strip()
    tag_match = PATCH_TAG_RE.search(text)
    if tag_match:
        text = tag_match.group(1).strip()

    fence_matches = FENCE_RE.findall(text)
    for fenced in fence_matches:
        if looks_like_patch(fenced):
            text = fenced.strip()
            break

    starts = []
    for marker in ("diff --git ", "--- a/"):
        idx = text.find(marker)
        if idx >= 0:
            starts.append(idx)
    if starts:
        text = text[min(starts) :].strip()

    if text.lower().startswith("patch:"):
        text = text.split(":", 1)[1].strip()

    if text.startswith("--- a/"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[1].startswith("+++ b/"):
            old_path = lines[0][len("--- a/") :].strip()
            new_path = lines[1][len("+++ b/") :].strip()
            if old_path and old_path == new_path:
                text = f"diff --git a/{old_path} b/{new_path}\n{text}"

    if record is not None and text.startswith("*** Begin Patch"):
        try:
            text = apply_patch_to_unified_diff(text, record)
        except Exception as exc:  # noqa: BLE001
            print(
                f"warning: failed to convert apply_patch output for "
                f"{record.get('instance_id')}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if (
        record is not None
        and text.startswith("diff --git ")
        and not unified_diff_hunk_counts_valid(text)
    ):
        try:
            text = repair_unified_diff_against_sources(text, record)
        except Exception as exc:  # noqa: BLE001
            print(
                f"warning: failed to normalize unified diff for "
                f"{record.get('instance_id')}: {type(exc).__name__}: {exc}",
                flush=True,
            )
    return text


def select_records(
    records: list[dict[str, Any]],
    limit_start: int,
    limit: int | None,
    instance_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if instance_ids:
        wanted = set(instance_ids)
        records = [record for record in records if record.get("instance_id") in wanted]
    end = None if limit is None else limit_start + limit
    return records[limit_start:end]


def build_official_prompt(record: dict[str, Any], retrieved_context: str, context_mode: str) -> str:
    issue = record.get("problem_statement") or record.get("prompt") or ""
    premise = "You will be provided with a partial code base and an issue statement explaining a problem to resolve."
    code_lines = []
    if context_mode == "sketch" and retrieved_context.strip():
        code_lines.extend(
            [
                "The following retrieved code context is a CodeGist semantic sketch version.",
                "It is only for semantic understanding of APIs, names, and control flow.",
                "Do not copy sketch text directly into the patch.",
                "",
            ]
        )
    code_lines.append(retrieved_context.rstrip())
    return "\n".join(
        [
            premise,
            "<issue>",
            str(issue).strip(),
            "</issue>",
            "",
            "<code>",
            "\n".join(code_lines).strip(),
            "</code>",
            "",
            build_patch_instruction_block(),
        ]
    ).strip()


def build_messages(
    record: dict[str, Any],
    retrieved_context: str,
    context_mode: str,
    prompt_style: str,
) -> list[dict[str, str]]:
    if prompt_style == "official":
        return [{"role": "user", "content": build_official_prompt(record, retrieved_context, context_mode)}]

    issue = record.get("problem_statement") or record.get("prompt") or ""
    parts = []
    note = context_note(context_mode)
    if retrieved_context.strip():
        parts.extend(
            [
                note,
                "",
                "```text",
                retrieved_context.rstrip(),
                "```",
                "",
            ]
        )

    parts.extend(
        [
            "Issue:",
            str(issue).strip(),
            "",
            "Generate a minimal patch that fixes the issue while preserving existing behavior.",
            "Output only a git-style unified diff patch that starts with lines like "
            "`diff --git a/path b/path` and uses `--- a/path` / `+++ b/path` file headers.",
            "Do not output Markdown fences, explanations, or apply_patch blocks such as "
            "`*** Begin Patch`.",
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a code repair model. Produce a minimal git-style unified diff patch. "
                "Output only the patch; never output apply_patch format."
            ),
        },
        {"role": "user", "content": "\n".join(parts).strip()},
    ]


def count_tokens(messages: list[dict[str, str]]) -> int | None:
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding("o200k_base")
        return sum(
            len(enc.encode(message["content"], disallowed_special=()))
            for message in messages
        )
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--raw_output_path")
    parser.add_argument("--prompt_debug_path")
    parser.add_argument("--usage_path")
    parser.add_argument("--stats_path")
    parser.add_argument("--env_file", default=str(REPO_ROOT.parent / ".env"))
    parser.add_argument("--context_mode", choices=["none", "raw", "sketch"], default="none")
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument("--codegist_root", default=DEFAULT_CODEGIST_ROOT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--api_key_env", default="LLM_API_KEY")
    parser.add_argument("--base_url_env", default="LLM_MODEL_URL")
    parser.add_argument("--model_env", default="LLM_MODEL")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help='Pass extra_body={"thinking":{"type":"disabled"}} to OpenAI-compatible APIs.',
    )
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--limit_start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance_ids", nargs="*")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt_style", choices=["coderag", "official"], default="coderag")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    data_file = resolve_path(args.data_file, REPO_ROOT)
    output_path = resolve_path(args.output_path, REPO_ROOT)
    raw_output_path = resolve_path(args.raw_output_path, REPO_ROOT) if args.raw_output_path else None
    prompt_debug_path = resolve_path(args.prompt_debug_path, REPO_ROOT) if args.prompt_debug_path else None
    usage_path = resolve_path(args.usage_path, REPO_ROOT) if args.usage_path else None
    stats_path = resolve_path(args.stats_path, REPO_ROOT) if args.stats_path else None

    records = select_records(load_json_or_jsonl(data_file), args.limit_start, args.limit, args.instance_ids)
    if not records:
        raise ValueError("No records selected")

    if args.overwrite:
        for path in (output_path, raw_output_path, prompt_debug_path, usage_path, stats_path):
            if path is not None and path.exists():
                path.unlink()

    env = dotenv_values(resolve_path(args.env_file, REPO_ROOT))
    model = args.model or env_value(env, *env_keys(args.model_env, "OPENAI_MODEL"), default="gpt-5.2")
    model_name_or_path = args.model_name_or_path or f"{model}-{args.context_mode}-top{args.topk_docs}"

    cg = None
    stats = {
        "context_mode": args.context_mode,
        "prompt_style": args.prompt_style,
        "topk_docs": args.topk_docs,
        "selected_records": len(records),
        "codegist": default_codegen_stats(),
        "records_without_docs": 0,
        "generation_errors": [],
    }
    if args.context_mode == "sketch":
        if not args.codegist_root:
            raise ValueError("--codegist_root is required for --context_mode sketch")
        cg = import_codegist(resolve_path(args.codegist_root, REPO_ROOT))

    predictions = load_existing_json(output_path)
    raw_outputs = load_existing_json(raw_output_path)
    prompt_debug = load_existing_json(prompt_debug_path)
    usage_records = load_existing_json(usage_path)

    if raw_output_path and raw_outputs and len(raw_outputs) != len(predictions):
        raise ValueError("Existing raw and cleaned output lengths differ.")
    if usage_path and usage_records and len(usage_records) != len(predictions):
        raise ValueError("Existing usage and prediction output lengths differ.")

    client = None
    if not args.dry_run:
        client = OpenAI(
            api_key=env_value(env, *env_keys(args.api_key_env, "OPENAI_API_KEY")),
            base_url=env_value(env, *env_keys(args.base_url_env, "OPENAI_BASE_URL")),
            timeout=args.timeout,
            max_retries=2,
        )

    for idx in range(len(predictions), len(records)):
        record = records[idx]
        instance_id = record["instance_id"]
        if args.context_mode != "none" and not record.get("docs"):
            stats["records_without_docs"] += 1

        retrieved_context = build_file_context(
            record,
            mode=args.context_mode,
            topk_docs=args.topk_docs,
            cg=cg,
            stats=stats["codegist"],
        )
        messages = build_messages(record, retrieved_context, args.context_mode, args.prompt_style)
        prompt_tokens = count_tokens(messages)

        if args.dry_run:
            prompt_debug.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "context_mode": args.context_mode,
                    "prompt_style": args.prompt_style,
                    "messages": messages,
                    "prompt_tokens": prompt_tokens,
                }
            )
            print(f"dry-run prompt {idx + 1}/{len(records)} {instance_id} tokens={prompt_tokens}")
            continue

        response = None
        text = ""
        request_error = None
        for attempt in range(1, args.retries + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
                if args.disable_thinking:
                    request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                response = client.chat.completions.create(**request_kwargs)
                text = response.choices[0].message.content or ""
                break
            except Exception as exc:  # noqa: BLE001
                request_error = f"{type(exc).__name__}: {exc}"
                print(f"request failed for {instance_id} attempt={attempt}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == args.retries:
                    if not args.continue_on_error:
                        raise
                    text = ""
                time.sleep(5 * attempt)
        if request_error and not text:
            stats["generation_errors"].append({"instance_id": instance_id, "error": request_error})

        patch = clean_patch(text, record)
        patch_status = patch_format_status(patch)
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": patch,
                "patch_format_status": patch_status,
            }
        )
        if raw_output_path:
            raw_outputs.append({"instance_id": instance_id, "raw_output": text})
        if prompt_debug_path:
            prompt_debug.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "context_mode": args.context_mode,
                    "prompt_style": args.prompt_style,
                    "messages": messages,
                    "raw_output": text,
                    "clean_patch": patch,
                    "patch_format_status": patch_status,
                    "prompt_tokens": prompt_tokens,
                }
            )
        if usage_path:
            usage = response.usage.model_dump() if response is not None and response.usage else None
            usage_records.append(
                {
                    "idx": args.limit_start + idx,
                    "instance_id": instance_id,
                    "context_mode": args.context_mode,
                    "prompt_style": args.prompt_style,
                    "api_usage": usage,
                    "local_prompt_tokens": prompt_tokens,
                    "raw_chars": len(text),
                    "patch_chars": len(patch),
                    "patch_format_status": patch_status,
                }
            )

        print(
            f"generated {len(predictions)}/{len(records)} {instance_id} "
            f"prompt_tokens={prompt_tokens} raw_chars={len(text)} patch_chars={len(patch)} "
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
            print(f"saved {len(predictions)} predictions to {output_path}", flush=True)

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
