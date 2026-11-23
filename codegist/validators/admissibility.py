"""Check template vocabulary, content coverage, and generality."""
from __future__ import annotations

import io
import re
import tokenize
from typing import List, Optional, Tuple

# A slot is a controlled placeholder like {body_blocks}. Everything else in a
# template is *literal skeleton*.
_SLOT_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

# Allowed markers outside template slots.
_DECLARED_MARKERS = ("->",)   # ordered dependency-chain connector


def _strip_slots(template: str) -> str:
    return _SLOT_RE.sub(" \x01 ", template)


def lexical_closure_ok(template: str) -> Tuple[bool, str]:
    """Check the literal template syntax against the allowed markers."""
    bare = _strip_slots(template)
    nonascii = sorted({c for c in bare if ord(c) > 127})
    if nonascii:
        return False, f"non-ASCII markup {nonascii}"
    if "{" in bare or "}" in bare:
        return False, "brace-DSL markup (residual { } after slot removal)"
    probe = bare
    for mk in _DECLARED_MARKERS:
        probe = probe.replace(mk, " ")
    if "=>" in probe:
        return False, "non-Python operator '=>' (not in declared markers M)"
    return True, ""


# Endpoint and count slots need another slot to cover the intervening content.
_PARTIAL_SLOTS = {
    "{first_targets}", "{last_expr}", "{n_stmts}",
    "{n_args}", "{kw_keys}", "{method_names}",
}
# Slots that cover the FULL ordered content of their range (only glue/format is
# dropped, which is in the droppable set D).
_COMPLETE_SLOTS = {
    "{stmt_chain}", "{call_chain}", "{body_chain}", "{body_blocks}",
    "{call_args_combined}", "{rest_args}", "{method_sigs}", "{sig}",
}


def information_preserved(template: str) -> Tuple[bool, str]:
    """Reject endpoint-only templates without a full-coverage slot."""
    slots = set(_SLOT_RE.findall(template))
    if (slots & _PARTIAL_SLOTS) and not (slots & _COMPLETE_SLOTS):
        return False, ("endpoint/count slot without full-coverage chain slot: "
                       f"{sorted(slots & _PARTIAL_SLOTS)}")
    return True, ""


def data_flow_call_chain_safe(template: str, region_type: Optional[str]) -> Tuple[bool, str]:
    """Reject call chains that omit data-flow details."""
    if region_type == "data_flow" and "{call_chain}" in set(_SLOT_RE.findall(template)):
        return False, "data_flow template uses lossy {call_chain} as coverage"
    return True, ""


def abstraction_ratio(template: str) -> float:
    if not template:
        return 0.0
    slot_chars = sum(len(m.group(0)) for m in _SLOT_RE.finditer(template))
    return slot_chars / len(template)


def is_generalizable(template: str, support: Optional[int],
                     theta: float = 0.5,
                     min_support: int = 2) -> Tuple[bool, str]:
    """Require sufficient bucket support and template slot coverage."""
    if support is not None and support < min_support:
        return False, f"support {support} < min_support {min_support}"
    r = abstraction_ratio(template)
    if r < theta:
        return False, f"abstraction ratio {r:.2f} < theta {theta} (baked-in literals)"
    return True, ""


def is_admissible(template: str, support: Optional[int] = None,
                  theta: float = 0.5,
                  min_support: int = 2,
                  region_type: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Returns (admissible, reasons). reasons lists every violated gate."""
    reasons: List[str] = []
    ok1, r1 = lexical_closure_ok(template)
    ok2, r2 = information_preserved(template)
    ok2b, r2b = data_flow_call_chain_safe(template, region_type)
    ok3, r3 = is_generalizable(template, support, theta, min_support)
    if not ok1:
        reasons.append("I1:" + r1)
    if not ok2:
        reasons.append("I2:" + r2)
    if not ok2b:
        reasons.append("I2b:" + r2b)
    if not ok3:
        reasons.append("I3:" + r3)
    return (ok1 and ok2 and ok2b and ok3), reasons


def render_lexically_valid(rendered: str) -> Tuple[bool, str]:
    """Check the syntax of the filled sketch at the token level."""
    if not rendered:
        return True, ""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(rendered).readline):
            if tok.type == tokenize.ERRORTOKEN and tok.string.strip():
                return False, f"non-Python lexeme {tok.string!r}"
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return True, ""
