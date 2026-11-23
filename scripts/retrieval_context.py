"""Shared retrieved-context rendering helpers for SWE-bench generation scripts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):
        for key in ("test", "data", "records", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise TypeError(f"Expected a JSON list or JSONL records in {path}")
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list or JSONL records in {path}")
    return data


def save_json(path: str | Path | None, payload: Any) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def comment_text(text: str) -> str:
    return "\n".join("# " + line if line else "#" for line in text.rstrip("\n").splitlines())


def context_note(mode: str) -> str:
    if mode == "raw":
        return (
            "The following retrieved repository context is relevant repository source code.\n"
            "It is provided only to help you understand APIs, style, names, control flow, "
            "and nearby implementation patterns.\n"
            "It may be incomplete and is not necessarily the file that must be edited."
        )
    if mode == "sketch":
        return (
            "The following retrieved repository context blocks marked as CodeGist semantic "
            "sketches are compressed reading views of relevant repository code, not the "
            "original executable source.\n"
            "Use these sketches only to understand APIs, style, names, control flow, and "
            "nearby implementation patterns.\n"
            "Do not copy sketch text verbatim into the final answer.\n"
            "The context may be incomplete and is not necessarily the file that must be edited."
        )
    if mode == "compressed_sketch":
        return (
            "The following retrieved repository context blocks are precompressed semantic "
            "sketches of relevant repository code, not the original executable source.\n"
            "Use these sketches only to understand APIs, style, names, control flow, and "
            "nearby implementation patterns.\n"
            "Do not copy sketch text verbatim into the final answer.\n"
            "The context may be incomplete and is not necessarily the file that must be edited."
        )
    if mode == "tokensugar":
        return (
            "The following retrieved repository context blocks are TokenSugar-compressed "
            "reading views of relevant repository code, not the original executable source.\n"
            "A macro legend appears before the files and defines every shorthand used in "
            "the compressed files.\n"
            "Mentally expand the macros while reading, use the context only to understand "
            "APIs, style, names, control flow, and nearby implementation patterns, and do "
            "not copy TokenSugar shorthand into the final patch.\n"
            "The context may be incomplete and is not necessarily the file that must be edited."
        )
    return ""


def default_codegen_stats() -> dict[str, Any]:
    return {
        "compressed": 0,
        "repaired": 0,
        "fallback_original": 0,
        "empty": 0,
        "errors": [],
    }


def import_codegist(codegist_root: str | Path):
    codegist_root = Path(codegist_root)
    if not codegist_root.exists():
        raise FileNotFoundError(f"CodeGist root does not exist: {codegist_root}")
    sys.path.insert(0, str(codegist_root))
    import codegist as cg  # noqa: PLC0415

    return cg


def repair_incomplete_python(src: str) -> str:
    lines = src.rstrip("\n").splitlines()
    if not lines:
        return src

    def add_trailing_pass(candidate_lines: list[str]) -> list[str]:
        last_nonempty = ""
        for line in reversed(candidate_lines):
            if line.strip():
                last_nonempty = line
                break
        if last_nonempty.rstrip().endswith(":"):
            indent = last_nonempty[: len(last_nonempty) - len(last_nonempty.lstrip())]
            return candidate_lines + [indent + "    pass"]
        return candidate_lines

    candidates = [lines, add_trailing_pass(lines)]
    for cut in range(len(lines) - 1, 0, -1):
        prefix = lines[:cut]
        candidates.append(prefix)
        candidates.append(add_trailing_pass(prefix))

    seen = set()
    for candidate_lines in candidates:
        candidate = "\n".join(candidate_lines).rstrip() + "\n"
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return src


def sketch_source(cg, src: str, label: str, stats: dict[str, Any]) -> str:
    if not src.strip():
        stats["empty"] += 1
        return src

    last_error = ""
    for candidate, repaired in ((src, False), (repair_incomplete_python(src), True)):
        try:
            result = cg.sketch_module(candidate)
            text = result.text.rstrip()
            stats["compressed"] += 1
            if repaired:
                stats["repaired"] += 1
            return text + "\n"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

    stats["fallback_original"] += 1
    if len(stats["errors"]) < 20:
        stats["errors"].append({"label": label, "error": last_error})
    return src


def doc_title(doc: dict[str, Any], idx: int) -> str:
    title = doc.get("title") or doc.get("path") or doc.get("file") or f"doc-{idx}"
    return str(title)


def doc_text(doc: dict[str, Any]) -> str:
    text = doc.get("text")
    if text is None:
        text = doc.get("contents", "")
    return "" if text is None else str(text)


def build_file_context(
    record: dict[str, Any],
    mode: str,
    topk_docs: int,
    cg=None,
    stats: dict[str, Any] | None = None,
) -> str:
    if mode == "none" or topk_docs <= 0:
        return ""

    docs = record.get("docs") or []
    if not docs:
        return ""

    rendered: list[str] = []
    if mode == "tokensugar":
        macro_block = str(record.get("tokensugar_macro_block") or "").strip()
        if macro_block:
            rendered.append(macro_block)

    for idx, doc in enumerate(docs[:topk_docs]):
        title = doc_title(doc, idx)
        text = doc_text(doc)
        if mode == "sketch":
            if cg is None or stats is None:
                raise ValueError("CodeGist mode requires cg and stats")
            text = sketch_source(cg, text, f"{record.get('instance_id', idx)}::{title}", stats)
        elif mode in {"compressed_sketch", "tokensugar"}:
            text = str(doc.get("compressed_text") or text)

        rendered.append(f"[start of {title}]")
        rendered.append(text.rstrip())
        rendered.append(f"[end of {title}]")

    return "\n\n".join(rendered).rstrip()
