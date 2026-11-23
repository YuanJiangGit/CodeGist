#!/usr/bin/env python3
"""Preprocess Python source for compact LLM prompts."""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "o200k_base"
_BODY_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_ENCODER_CACHE: dict[str, tuple[object | None, str]] = {}


# Token counting


def _regex_token_count(source: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", source, flags=re.UNICODE))


def _load_model_encoder(model: str) -> tuple[object | None, str]:
    if model in _ENCODER_CACHE:
        return _ENCODER_CACHE[model]
    try:
        import tiktoken
    except ImportError:
        result = (None, "regex-fallback")
        _ENCODER_CACHE[model] = result
        return result
    enc = tiktoken.get_encoding("o200k_base")
    result = (enc, "o200k_base")
    _ENCODER_CACHE[model] = result
    return result


def tokenizer_name(model: str = DEFAULT_MODEL) -> str:
    _, name = _load_model_encoder(model)
    return name


def count_model_tokens(source: str, *, model: str = DEFAULT_MODEL) -> int:
    enc, _ = _load_model_encoder(model)
    if enc is None:
        return _regex_token_count(source)
    return len(enc.encode(source, disallowed_special=()))  # type: ignore[attr-defined]


def count_python_tokens(source: str) -> int:
    n = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type not in (
                tokenize.ENCODING,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.COMMENT,
            ):
                n += 1
    except tokenize.TokenError:
        pass
    return n


# AST cleanup


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _strip_docstrings(tree: ast.AST) -> int:
    n_removed = 0
    for node in ast.walk(tree):
        if isinstance(node, _BODY_NODES):
            body = getattr(node, "body", None)
            if body and _is_docstring_stmt(body[0]):
                body.pop(0)
                n_removed += 1
                if not body:
                    body.append(ast.Pass())
    return n_removed


def _strip_type_annotations(tree: ast.AST) -> int:
    n_removed = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                node.returns = None
                n_removed += 1
            args = node.args
            all_args = (
                list(args.posonlyargs)
                + list(args.args)
                + list(args.kwonlyargs)
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            for arg in all_args:
                if arg is not None and arg.annotation is not None:
                    arg.annotation = None
                    n_removed += 1
        elif isinstance(node, ast.AnnAssign):
            node.annotation = ast.Name(id="object", ctx=ast.Load())
            n_removed += 1
    return n_removed


# Return tuple cleanup


def _stmt_ast(src: str) -> ast.stmt | None:
    try:
        return ast.parse("def __p__():\n    " + src).body[0].body[0]  # type: ignore[attr-defined]
    except SyntaxError:
        return None


def _same_stmt_ast(a: str, b: str) -> bool:
    left = _stmt_ast(a)
    right = _stmt_ast(b)
    if left is None or right is None:
        return False
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _drop_redundant_return_tuple_parens(source: str) -> str:
    """Turn `return (a, b)` back into `return a, b` when AST-equivalent."""
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        indent = body[: len(body) - len(body.lstrip(" \t"))]
        stmt = body[len(indent) :]
        if stmt.startswith("return (") and stmt.endswith(")"):
            candidate = "return " + stmt[len("return (") : -1]
            if _same_stmt_ast(stmt, candidate):
                out.append(indent + candidate + newline)
                continue
        out.append(line)
    return "".join(out)


# Whitespace minification


_NAMELIKE = {tokenize.NAME, tokenize.NUMBER}
_STRINGLIKE: set[int] = {tokenize.STRING}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(tokenize, _name):
        _STRINGLIKE.add(getattr(tokenize, _name))


def _needs_space(prev_type: int, prev_str: str, curr_type: int, curr_str: str) -> bool:
    if prev_str == "from" and curr_str == ".":
        return True
    if prev_str == "." and curr_str == "import":
        return True
    if prev_type in _NAMELIKE and curr_type in _NAMELIKE:
        return True
    if prev_type == tokenize.NUMBER and curr_str == ".":
        return True
    if prev_type in _NAMELIKE and curr_type in _STRINGLIKE:
        return True
    if prev_type in _STRINGLIKE and curr_type in _NAMELIKE:
        return True
    return False


def _source_segment(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> str:
    start_line, start_col = start
    end_line, end_col = end
    if start_line == end_line:
        return lines[start_line - 1][start_col:end_col]
    parts = [lines[start_line - 1][start_col:]]
    parts.extend(lines[i] for i in range(start_line, end_line - 1))
    parts.append(lines[end_line - 1][:end_col])
    return "".join(parts)


def _minify_tokens(source: str, *, strip_blank_lines: bool = True, indent_char: str = "\t") -> str:
    out_lines: list[str] = []
    cur_parts: list[str] = []
    indent_level = 0
    prev_type: int | None = None
    prev_str = ""
    source_lines = source.splitlines(keepends=True)

    def flush_logical_line() -> None:
        line = "".join(cur_parts)
        if line.strip() or not strip_blank_lines:
            out_lines.append((indent_char * indent_level) + line)
        cur_parts.clear()

    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    token_iter = iter(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok in token_iter:
        t_type, t_str = tok.type, tok.string
        if t_type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        if t_type == tokenize.INDENT:
            indent_level += 1
            continue
        if t_type == tokenize.DEDENT:
            indent_level -= 1
            continue
        if t_type == tokenize.NEWLINE:
            flush_logical_line()
            prev_type, prev_str = None, ""
            continue
        if fstring_start is not None and fstring_end is not None and t_type == fstring_start:
            start = tok.start
            end = tok.end
            depth = 1
            for f_tok in token_iter:
                end = f_tok.end
                if f_tok.type == fstring_start:
                    depth += 1
                elif f_tok.type == fstring_end:
                    depth -= 1
                    if depth == 0:
                        break
            t_type = tokenize.STRING
            t_str = _source_segment(source_lines, start, end)
        if prev_type is not None and _needs_space(prev_type, prev_str, t_type, t_str):
            cur_parts.append(" ")
        cur_parts.append(t_str)
        prev_type, prev_str = t_type, t_str
    flush_logical_line()
    return "\n".join(out_lines) + ("\n" if out_lines else "")


# Public API


@dataclass
class PreprocessResult:
    text: str
    original_chars: int
    processed_chars: int
    original_py_tokens: int
    processed_py_tokens: int
    original_model_tokens: int
    processed_model_tokens: int
    model: str
    tokenizer: str
    docstrings_removed: int = 0
    annotations_removed: int = 0

    @property
    def char_reduction(self) -> float:
        return 1.0 - (self.processed_chars / self.original_chars) if self.original_chars else 0.0

    @property
    def py_token_reduction(self) -> float:
        return 1.0 - (self.processed_py_tokens / self.original_py_tokens) if self.original_py_tokens else 0.0

    @property
    def model_token_reduction(self) -> float:
        if not self.original_model_tokens:
            return 0.0
        return 1.0 - (self.processed_model_tokens / self.original_model_tokens)


def preprocess(
    source: str,
    *,
    strip_docstrings: bool = True,
    strip_type_annotations: bool = False,
    strip_blank_lines: bool = True,
    indent_char: str = "\t",
) -> str:
    tree = ast.parse(source)
    if strip_docstrings:
        _strip_docstrings(tree)
    if strip_type_annotations:
        _strip_type_annotations(tree)
    ast.fix_missing_locations(tree)
    canonical = ast.unparse(tree)
    canonical = _drop_redundant_return_tuple_parens(canonical)
    return _minify_tokens(canonical, strip_blank_lines=strip_blank_lines, indent_char=indent_char)


def preprocess_with_stats(source: str, **kwargs) -> PreprocessResult:
    model = kwargs.get("model", DEFAULT_MODEL)
    tree = ast.parse(source)
    docstrings_removed = _strip_docstrings(tree) if kwargs.get("strip_docstrings", True) else 0
    annotations_removed = _strip_type_annotations(tree) if kwargs.get("strip_type_annotations", False) else 0
    ast.fix_missing_locations(tree)
    canonical = ast.unparse(tree)
    canonical = _drop_redundant_return_tuple_parens(canonical)
    processed = _minify_tokens(
        canonical,
        strip_blank_lines=kwargs.get("strip_blank_lines", True),
        indent_char=kwargs.get("indent_char", "\t"),
    )
    return PreprocessResult(
        text=processed,
        original_chars=len(source),
        processed_chars=len(processed),
        original_py_tokens=count_python_tokens(source),
        processed_py_tokens=count_python_tokens(processed),
        original_model_tokens=count_model_tokens(source, model=model),
        processed_model_tokens=count_model_tokens(processed, model=model),
        model=model,
        tokenizer=tokenizer_name(model),
        docstrings_removed=docstrings_removed,
        annotations_removed=annotations_removed,
    )


# CLI


def _process_one(path: Path, output: Path | None, **kwargs) -> PreprocessResult:
    source = path.read_text(encoding="utf-8", errors="replace")
    result = preprocess_with_stats(source, **kwargs)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.text, encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggressively preprocess Python source for LLM prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("paths", nargs="+", type=Path, help="Python source files")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output file or output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Tokenizer model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--keep-docstrings", action="store_true", help="Do NOT strip docstrings")
    ap.add_argument("--strip-type-hints", action="store_true", help="Drop type hints")
    ap.add_argument("--keep-blank-lines", action="store_true", help="Preserve blank lines")
    ap.add_argument("--indent", default="\t", help="Indent character per nesting level (default: tab)")
    ap.add_argument("--report", action="store_true", help="Print per-file and total token reduction stats")
    args = ap.parse_args(argv)

    kwargs = dict(
        model=args.model,
        strip_docstrings=not args.keep_docstrings,
        strip_type_annotations=args.strip_type_hints,
        strip_blank_lines=not args.keep_blank_lines,
        indent_char=args.indent,
    )

    multi = len(args.paths) > 1
    if multi and args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    total = {
        "orig_chars": 0,
        "proc_chars": 0,
        "orig_py_toks": 0,
        "proc_py_toks": 0,
        "orig_model_toks": 0,
        "proc_model_toks": 0,
        "files": 0,
        "failed": 0,
    }
    tokenizer = tokenizer_name(args.model)

    for path in args.paths:
        out_path = (args.output / path.name) if multi and args.output else args.output
        try:
            res = _process_one(path, out_path, **kwargs)
        except SyntaxError as exc:
            print(f"[SKIP {path}] SyntaxError: {exc}", file=sys.stderr)
            total["failed"] += 1
            continue
        except Exception as exc:
            print(f"[FAIL {path}] {type(exc).__name__}: {exc}", file=sys.stderr)
            total["failed"] += 1
            continue

        total["orig_chars"] += res.original_chars
        total["proc_chars"] += res.processed_chars
        total["orig_py_toks"] += res.original_py_tokens
        total["proc_py_toks"] += res.processed_py_tokens
        total["orig_model_toks"] += res.original_model_tokens
        total["proc_model_toks"] += res.processed_model_tokens
        total["files"] += 1

        if args.report:
            print(
                f"{path}: chars {res.original_chars}->{res.processed_chars} ({res.char_reduction:.1%}); "
                f"model-tokens {res.original_model_tokens}->{res.processed_model_tokens} "
                f"({res.model_token_reduction:.1%}); "
                f"py-tokens {res.original_py_tokens}->{res.processed_py_tokens} "
                f"({res.py_token_reduction:.1%}); "
                f"docstrings={res.docstrings_removed} anns={res.annotations_removed}"
            )
        elif out_path is None:
            sys.stdout.write(res.text)

    if args.report and total["orig_chars"]:
        char_red = 1 - total["proc_chars"] / total["orig_chars"]
        model_red = 1 - total["proc_model_toks"] / total["orig_model_toks"] if total["orig_model_toks"] else 0
        py_red = 1 - total["proc_py_toks"] / total["orig_py_toks"] if total["orig_py_toks"] else 0
        print(
            f"\nTotal ({total['files']} files, {total['failed']} failed, tokenizer={tokenizer}): "
            f"chars {total['orig_chars']}->{total['proc_chars']} ({char_red:.1%}); "
            f"model-tokens {total['orig_model_toks']}->{total['proc_model_toks']} ({model_red:.1%}); "
            f"py-tokens {total['orig_py_toks']}->{total['proc_py_toks']} ({py_red:.1%})"
        )

    return 0 if total["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
