"""Path filters for mining corpora."""
from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List

# These exclusions apply unless the CLI disables them.
DEFAULT_EXCLUDES: List[str] = [
    "**/__pycache__/**",
    "**/.git/**",
    "**/.tox/**",
    "**/.venv/**",
    "**/venv/**",
    "**/node_modules/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
]

# Frequently-useful additions (not on by default — surface in --help so users
# can opt in explicitly):
RECOMMENDED_EXCLUDES: List[str] = [
    "**/tests/**",
    "**/test/**",
    "**/test_*.py",
    "**/conftest.py",
    "**/_vendor/**",
    "**/vendor/**",
    "**/docs/**",
    "**/doc/**",
    "**/examples/**",
    "**/example/**",
    "**/build/**",
    "**/dist/**",
    "**/migrations/**",
]


def is_excluded(path: Path, patterns: Iterable[str]) -> bool:
    """True if *any* pattern in ``patterns`` matches ``path``."""
    name = path.name
    rel = str(path).replace("\\", "/")
    parts = set(path.parts)
    for pat in patterns:
        if not pat:
            continue
        pat_norm = pat.replace("\\", "/")
        # **/<name>/**  — any path component equals <name>
        if pat_norm.startswith("**/") and pat_norm.endswith("/**"):
            target = pat_norm[3:-3]
            if target and target in parts:
                return True
            continue
        # **/<glob>  — basename match
        if pat_norm.startswith("**/") and "/" not in pat_norm[3:]:
            if fnmatch(name, pat_norm[3:]):
                return True
            continue
        # general fall-back — fnmatch against the full relative path
        if fnmatch(rel, pat_norm):
            return True
    return False


def resolve(extra: Iterable[str] | None,
            use_defaults: bool = True) -> List[str]:
    """Build the final exclude list to pass into ``iter_corpus``."""
    out: List[str] = list(DEFAULT_EXCLUDES) if use_defaults else []
    if extra:
        out.extend(extra)
    return out
