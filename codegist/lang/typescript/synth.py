"""Synthesize and check TypeScript Write templates."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ...auto_mining.extract import AutoRegion
from ...llm_client import LLMClient
from .prompts import (PROMPT_CRITIC_TS, PROMPT_GENERALITY_TS,
                      PROMPT_SYNTH_TS, slot_menu_ts)
from .slots import TSTemplateTransform, validate_template_ts

log = logging.getLogger("codegist.lang.typescript.synth")


@dataclass
class TSSynthResult:
    template: str
    rationale: str
    rounds: int
    critic_verdict: str
    critic_reason: str = ""
    applies_broadly: bool = False
    generality_verdict: str = "SKIPPED"
    generality_reasons: List[dict] = field(default_factory=list)


def _parse_json(reply: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _render_on(template: str, rt: str, region: AutoRegion) -> str:
    return TSTemplateTransform(template, rt, "probe").render(region).materialize_default()


def _safety_check(template: str, rt: str, region: AutoRegion,
                  llm: LLMClient) -> tuple[bool, str, str]:
    try:
        rendered = _render_on(template, rt, region)
    except Exception as e:
        return False, f"render failed: {e}", "use simpler slots"
    prompt = PROMPT_CRITIC_TS.format(template=template,
                                     instance=region.source.strip(),
                                     rendered=rendered)
    try:
        reply = llm.chat(prompt)
    except Exception as e:
        return False, f"llm_error: {e}", ""
    parsed = _parse_json(reply) or {}
    return (bool(parsed.get("safe")), str(parsed.get("reason", "")),
            str(parsed.get("suggestion", "")))


def synth_loop_ts(
    bucket_instances: List[AutoRegion],
    rt: str,
    llm: LLMClient,
    neighbours: Optional[List[AutoRegion]] = None,
    max_rounds: int = 2,
) -> Optional[TSSynthResult]:
    if not bucket_instances:
        return None
    sample = bucket_instances[:3]
    instances_block = "\n\n".join(
        f"=== Instance {i+1} ===\n```typescript\n{r.source.strip()}\n```"
        for i, r in enumerate(sample))

    feedback = ""
    template = rationale = ""
    safety_ok = False
    safety_reason = ""
    rounds_used = 0
    for round_no in range(1, max_rounds + 1):
        rounds_used = round_no
        feedback_block = (f"FEEDBACK FROM PREVIOUS ROUND:\n{feedback}\n\n"
                          if feedback else "")
        prompt = PROMPT_SYNTH_TS.format(
            n=len(sample), rt=rt, instances_block=instances_block,
            slot_menu=slot_menu_ts(rt), feedback_block=feedback_block)
        try:
            reply = llm.chat(prompt)
        except Exception as e:
            log.warning("ts synth LLM call failed: %s", e)
            return None
        parsed = _parse_json(reply)
        if not parsed or "template" not in parsed:
            feedback = ("Previous reply was not valid JSON. Output ONLY a JSON "
                        "object with keys 'template' and 'rationale'.")
            continue
        new_template = (parsed.get("template") or "").strip()
        err = validate_template_ts(new_template, rt)
        if err:
            feedback = (f"Previous template was invalid: {err}. Use only slots "
                        f"from the menu and plain TS tokens.")
            continue
        template = new_template
        rationale = (parsed.get("rationale") or "").strip()
        safety_ok, safety_reason, suggestion = _safety_check(template, rt, sample[0], llm)
        if safety_ok:
            break
        feedback = (f"Critic rejected previous template. Reason: {safety_reason}. "
                    f"Suggestion: {suggestion}. Revise.")

    if not template:
        return None
    if not safety_ok:
        return TSSynthResult(template=template, rationale=rationale,
                             rounds=rounds_used, critic_verdict="NO",
                             critic_reason=safety_reason or "exhausted rounds")

    if not neighbours:
        return TSSynthResult(template=template, rationale=rationale,
                             rounds=rounds_used, critic_verdict="YES",
                             critic_reason=safety_reason)

    nb_block = "\n\n".join(
        f"=== Instance {i+1} ===\n```typescript\n{r.source.strip()}\n```"
        for i, r in enumerate(neighbours[:5]))
    try:
        reply = llm.chat(PROMPT_GENERALITY_TS.format(
            template=template, instances_block=nb_block))
        parsed = _parse_json(reply) or {}
    except Exception as e:
        log.warning("ts generality call failed: %s", e)
        parsed = {}
    verdict = str(parsed.get("verdict", "NARROW")).upper()
    per_instance = parsed.get("per_instance") or []
    return TSSynthResult(
        template=template, rationale=rationale, rounds=rounds_used,
        critic_verdict="YES", critic_reason=safety_reason,
        applies_broadly=(verdict == "BROAD"),
        generality_verdict=verdict if verdict in ("BROAD", "NARROW") else "NARROW",
        generality_reasons=[x for x in per_instance if isinstance(x, dict)],
    )
