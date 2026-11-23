"""Compact the text of selected TypeScript rewrites."""
from __future__ import annotations

_PUNCT = set("(){}[];,.")


def _strip_comments_collapse_ws(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    in_str: str | None = None       # '"', "'" or "`" when inside a literal
    pending_ws = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'", "`"):
            if pending_ws and out and out[-1] not in _PUNCT:
                out.append(" ")
            pending_ws = False
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            pending_ws = True
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            pending_ws = True
            continue
        if ch.isspace():
            pending_ws = True
            i += 1
            continue
        if pending_ws:
            prev = out[-1] if out else ""
            if prev and prev not in _PUNCT and ch not in _PUNCT:
                out.append(" ")
            pending_ws = False
        out.append(ch)
        i += 1
    return "".join(out).strip()


def compact_render_ts(text: str) -> str:
    """Compact one selected rewrite span (or one filled skeleton)."""
    if not text:
        return text
    compact = _strip_comments_collapse_ws(text)
    return compact or text.strip()
