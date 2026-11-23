"""Compact the text of selected Python rewrites."""
from __future__ import annotations

import io
import re
import tokenize
from functools import lru_cache


_NAMELIKE = {tokenize.NAME, tokenize.NUMBER}
_STRINGLIKE: set[int] = {tokenize.STRING}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(tokenize, _name):
        _STRINGLIKE.add(getattr(tokenize, _name))

_COMPOUND_KEYWORDS = {"if", "for", "while", "try", "with", "def", "class"}


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
    if prev_str in {"return", "yield", "raise", "assert", "else"} and curr_str in {"(", "["}:
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


def _token_compact(text: str) -> str:
    parts: list[str] = []
    prev_type: int | None = None
    prev_str = ""
    indent_level = 0
    pending_line_indent: int | None = None
    source_lines = text.splitlines(keepends=True)
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    token_iter = iter(tokenize.generate_tokens(io.StringIO(text).readline))

    def append_sep(sep: str) -> None:
        if not sep:
            return
        while parts and (parts[-1] in {";", "->"} or parts[-1].startswith("\n")):
            parts.pop()
        if parts:
            parts.append(sep)

    for tok in token_iter:
        t_type, t_str = tok.type, tok.string
        if t_type == tokenize.INDENT:
            indent_level += 1
            continue
        if t_type == tokenize.DEDENT:
            indent_level = max(0, indent_level - 1)
            continue
        if t_type == tokenize.NEWLINE:
            pending_line_indent = indent_level
            continue
        if t_type in (
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.NL,
            tokenize.COMMENT,
        ):
            continue
        if pending_line_indent is not None:
            if indent_level > pending_line_indent:
                if t_type == tokenize.NAME and t_str in _COMPOUND_KEYWORDS:
                    sep = "\n" + (" " * indent_level)
                else:
                    sep = ""
            elif indent_level == pending_line_indent and indent_level > 0:
                if t_type == tokenize.NAME and t_str in _COMPOUND_KEYWORDS:
                    sep = "\n" + (" " * indent_level)
                else:
                    sep = "->"
            else:
                sep = "\n" + (" " * indent_level)
            append_sep(sep)
            if sep:
                prev_type = None
                prev_str = ""
            pending_line_indent = None
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
            parts.append(" ")
        parts.append(t_str)
        prev_type, prev_str = t_type, t_str
    return "".join(parts).strip()


def _regex_compact(text: str) -> str:
    text = re.sub(r"[ \t]*#[^\r\n]*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([(){}\[\],.:;+\-*/%<>=!&|^~])\s*", r"\1", text)
    text = re.sub(r"\b(from)\.(import)\b", r"\1 . \2", text)
    return text


def _restore_compound_boundaries(text: str) -> str:
    """Keep statement boundaries when compacting across compound statements."""
    def compound_sep(match: re.Match[str]) -> str:
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix = text[line_start:match.start()]
        stripped = prefix.lstrip()
        leading = len(prefix) - len(stripped)
        if (re.match(r"(if|for|while|try|with|def|class)\b", stripped)
                and ":" in stripped):
            return "\n" + (" " * (leading + 1))
        return "\n"

    text = re.sub(
        r"->(?=(if|for|while|try|with|def|class)\b)",
        compound_sep,
        text,
    )
    text = re.sub(
        r";(?=(if|for|while|try|with|def|class)\b)",
        compound_sep,
        text,
    )
    return re.sub(r"(?m)^\s*->", "", text)


@lru_cache(maxsize=1 << 16)
def compact_render(text: str) -> str:
    """Compact a selected rewrite while preserving its token boundaries."""
    if not text:
        return text
    try:
        compact = _token_compact(text)
    except (IndentationError, tokenize.TokenError):
        compact = _regex_compact(text)
    compact = _restore_compound_boundaries(compact)
    return compact or text.strip()
