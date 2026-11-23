"""Recover synthesized templates from a mining log."""
from __future__ import annotations

import ast
import io
import re
from typing import Dict

_OK_MARK = "llm-synth OK "
# meta tail: "(rounds=1, gen=BROAD, broad=True, refined)"
_META_RE = re.compile(
    r"\(rounds=(?P<rounds>\d+),\s*gen=(?P<gen>\w+),\s*"
    r"broad=(?P<broad>True|False)(?P<refined>,\s*refined)?\s*\)"
)


def _parse_ok_line(rest: str) -> tuple[str, dict] | None:
    """Parse an accepted template from the text after 'llm-synth OK'."""
    fp, sep, after = rest.partition(" -> ")
    if not sep or not fp:
        return None
    fp = fp.strip()
    try:
        idx = after.rindex(" (rounds=")
    except ValueError:
        return None
    repr_part = after[:idx].strip()
    meta_part = after[idx + 1:].strip()
    m = _META_RE.fullmatch(meta_part)
    if not m:
        return None
    try:
        template = ast.literal_eval(repr_part)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(template, str) or not template:
        return None
    gen = m.group("gen").upper()
    broad = m.group("broad") == "True"
    info = {
        "template": template,
        "rationale": "",
        "rounds": int(m.group("rounds")),
        "verdict": "YES",            # OK lines are accepted (Phase-A safety YES)
        "reason": "",
        "applies_broadly": broad,
        "refined": bool(m.group("refined")),
        "generality_verdict": gen,   # BROAD / NARROW / SKIPPED
        "generality_reasons": [],
        "neighbours_used": 0,
    }
    return fp, info


def parse_synth_log(log_path: str) -> Dict[str, dict]:
    """Read accepted templates keyed by fingerprint."""
    results: Dict[str, dict] = {}
    with io.open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            pos = line.find(_OK_MARK)
            if pos < 0:
                continue
            parsed = _parse_ok_line(line[pos + len(_OK_MARK):].rstrip("\n"))
            if parsed is None:
                continue
            fp, info = parsed
            results[fp] = info
    return results
