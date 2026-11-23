"""Load .env and expose pipeline thresholds."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Config:
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    support_threshold: int = 2     # θ in Alg.1 — min bucket size to mine
    # Minimum mean token savings across the bucket.
    gain_threshold: float = 0.3
    # Maximum LLM calls for semantic validation.
    llm_max_validations: int = 50_000


def load(llm_max_validations: int | None = None,
         gain_threshold: float | None = None) -> Config:
    model = os.environ.get("LLM_MODEL")
    key = os.environ.get("LLM_API_KEY")
    url = os.environ.get("LLM_MODEL_URL")
    if not (model and key and url):
        raise RuntimeError(
            "Missing LLM_MODEL / LLM_API_KEY / LLM_MODEL_URL in .env"
        )
    kwargs: dict = {"llm_model": model, "llm_api_key": key, "llm_base_url": url}
    if llm_max_validations is not None:
        kwargs["llm_max_validations"] = llm_max_validations
    if gain_threshold is not None:
        kwargs["gain_threshold"] = gain_threshold
    return Config(**kwargs)
