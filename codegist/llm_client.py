"""Thin wrapper around an OpenAI-compatible chat endpoint."""
from __future__ import annotations

import logging
import random
import time

from openai import OpenAI

from .config import Config

log = logging.getLogger("codegist.llm_client")

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRIES = 4   # bounded: after this many transient retries, give up on the call


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_transient(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in _TRANSIENT_STATUS_CODES:
        return True
    text = str(exc).lower()
    return (
        "503" in text
        or "service temporarily unavailable" in text
        or "timeout" in text
        or "timed out" in text
        or "connection" in text
    )


def _is_param_error(exc: Exception) -> bool:
    """Identify API errors caused by unsupported request parameters."""
    code = _status_code(exc)
    if code in (400, 422):
        return True
    t = str(exc).lower()
    return ("unrecognized" in t or "unsupported" in t or "unexpected" in t
            or "extra fields" in t
            or ("invalid" in t and ("seed" in t or "top_p" in t)))


def _retry_sleep(attempt: int) -> None:
    delay = min(20.0, 2.0 * (2 ** min(attempt, 3)))
    delay += random.uniform(0.0, 1.0)
    time.sleep(delay)


class LLMClient:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            # Allow enough time for synthesis requests made concurrently.
            timeout=120.0,
        )

    def chat(self, prompt: str, system: str | None = None) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        last_err: Exception | None = None
        attempts = 0
        # Use deterministic decoding when supported by the endpoint.
        import os
        extra: dict = {"seed": int(os.environ.get("CODEGIST_LLM_SEED", "0")),
                       "top_p": 1.0}
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.llm_model,
                    messages=msgs,
                    temperature=0,
                    **extra,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if extra and _is_param_error(e):
                    log.warning(
                        "endpoint rejected determinism params (seed/top_p); "
                        "dropping them and retrying once: %s", e)
                    extra = {}
                    continue
                attempts += 1
                # Retry transient failures up to the configured limit.
                cap = _MAX_RETRIES if _is_transient(e) else 1
                if attempts > cap:
                    break
                if attempts in (1, 2, 3):
                    log.warning(
                        "LLM error; retry %d/%d (status=%s): %s",
                        attempts, cap, _status_code(e), e,
                    )
                _retry_sleep(attempts)
        raise RuntimeError(
            f"LLM call failed after {attempts} attempt(s): {last_err}")
