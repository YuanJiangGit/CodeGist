"""Measure average token savings across a schema bucket."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken

from ..grouping import Bucket
from ..regions import Region
from ..write_infer import WriteTemplate


def _get_encoding():
    return tiktoken.get_encoding("o200k_base")


_ENC = _get_encoding()


def _tokens(s: str) -> int:
    return len(_ENC.encode(s, disallowed_special=()))


@dataclass
class TokenReport:
    instances: int
    mean_gain: float
    total_gain: int
    samples: List[dict]


def validate(bucket: Bucket, write: WriteTemplate, gain_threshold: int) -> tuple[bool, TokenReport]:
    samples = []
    total = 0
    for r in bucket.members:
        try:
            rewritten = write.render_default(r)
        except Exception:
            rewritten = r.source
        g = _tokens(r.source) - _tokens(rewritten)
        total += g
        samples.append({
            "file": r.file,
            "lineno": r.lineno,
            "orig_tokens": _tokens(r.source),
            "rewritten_tokens": _tokens(rewritten),
            "gain": g,
            "rewritten": rewritten,
        })
    mean = total / len(bucket.members) if bucket.members else 0.0
    report = TokenReport(instances=len(bucket.members), mean_gain=mean,
                         total_gain=total, samples=samples)
    return mean >= gain_threshold, report
