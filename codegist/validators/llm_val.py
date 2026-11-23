"""Require a YES verdict from the LLM on every expected sample."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List

from ..config import Config
from ..grouping import Bucket
from ..llm_client import LLMClient
from ..write_infer import WriteTemplate


_SYSTEM = (
    "You judge whether a short code-like SKETCH still lets a downstream LLM "
    "understand an original Python snippet. The sketch need not be executable; "
    "judge by UNDERSTANDING. The rule: the sketch may drop recoverable GLUE but "
    "must keep the DEPENDENCY SIGNATURE. Answer YES if it kept the dependency "
    "signature — the identity (class/function name), structural names "
    "(params/fields), and, for a body, the calls it makes IN ORDER + the "
    "control-flow shape + what it returns/raises — while dropping only "
    "recoverable glue (docstrings, type hints, formatting, self.x=x init when "
    "field names are kept, intermediate-variable bookkeeping, the return keyword). "
    "A body compressed to its ORDERED CALL CHAIN (e.g. norm1->attn->dropout->"
    "return) is GOOD — answer YES. Answer NO if it destroyed the dependency "
    "signature — lost WHICH calls or their ORDER (a body collapsed to a vague "
    "'# N calls' count or prose), lost the identity, or dropped structural names. "
    "Under-compression is fine; losing the signature is not. "
    "Answer exactly one token: YES or NO, then a one-line reason."
)


def _ask(client: LLMClient, original: str, rewritten: str) -> tuple[bool, str]:
    prompt = (
        "ORIGINAL:\n```python\n" + original.strip() + "\n```\n\n"
        "COMPRESSED SKETCH (code-like, not necessarily valid Python):\n"
        "```\n" + rewritten.strip() + "\n```\n\n"
        "Did the sketch keep the DEPENDENCY SIGNATURE — the identity, structural "
        "names, and the body's calls IN ORDER + control-flow shape — while "
        "dropping only recoverable glue? A body shown as its ordered call chain "
        "(a->b->c->return) is fine. Say NO only if WHICH calls or their order was "
        "lost (body collapsed to a count/prose), or identity/names were dropped. "
        "Answer YES or NO and one short reason."
    )
    try:
        reply = client.chat(prompt, system=_SYSTEM)
    except Exception as e:
        return False, f"llm_error: {e}"
    head = reply.strip().split(None, 1)
    verdict = head[0].upper().strip(".,:") if head else ""
    yes = bool(re.match(r"^YES\b", verdict))
    return yes, reply


@dataclass
class LLMReport:
    judged: int
    yes: int
    consistency_ratio: float
    samples: List[dict]


def validate(
    bucket: Bucket,
    write: WriteTemplate,
    client: LLMClient,
    sample_size: int = 2,
    budget_remaining: int = 1_000_000,
    min_consistency: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[bool, LLMReport, int]:
    """Returns (passed, report, calls_consumed)."""
    rng = rng or random.Random(0)
    candidates = list(bucket.members)
    rng.shuffle(candidates)
    take = min(sample_size, len(candidates), budget_remaining)
    samples: List[dict] = []
    yes = 0
    for r in candidates[:take]:
        try:
            rewritten = write.render_default(r)
        except Exception as e:
            samples.append({"file": r.file, "lineno": r.lineno,
                            "verdict": "render_error", "reply": str(e)})
            continue
        ok, reply = _ask(client, r.source, rewritten)
        if ok:
            yes += 1
        samples.append({"file": r.file, "lineno": r.lineno,
                        "verdict": "YES" if ok else "NO", "reply": reply})
    judged = len(samples)
    ratio = (yes / judged) if judged else 0.0
    report = LLMReport(judged=judged, yes=yes,
                       consistency_ratio=ratio, samples=samples)
    expected = min(sample_size, len(candidates))
    passed = judged == expected and yes == judged and ratio >= min_consistency
    return passed, report, judged
