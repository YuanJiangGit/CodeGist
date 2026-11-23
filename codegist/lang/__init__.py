"""Language adapters for CodeGist."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

SUPPORTED_LANGS = ("python", "java", "typescript")


def schema_language(path: Union[Path, str]) -> str:
    """Read the language a schema_set.json was mined for (default python)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return "python"
    lang = (data.get("language") or "python").lower()
    return lang if lang in SUPPORTED_LANGS else "python"
