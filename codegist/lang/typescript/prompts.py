"""Prompts for synthesizing and checking TypeScript templates."""
from __future__ import annotations

from .slots import SLOT_VOCAB_TS

_SLOT_DESCRIPTIONS_TS = {
    "{modifiers}": "leading modifiers (export/async/static/public/readonly)",
    "{func_name}": "function/method name",
    "{ret_type}": "return type (without the leading ':')",
    "{sig}": "typed parameter list + return type, e.g. (a: number): string",
    "{body_chain}": ("dependency chain of the body: short statements verbatim, "
                     "long ones as ordered callee chains, control flow keeps "
                     "head + braces; statements end with ';'"),
    "{body_blocks}": ("the body's top-level statements, EACH an open hole for "
                      "block-level schemas (the combiner slot); use in "
                      "'{ret_type} {func_name}{sig}: {body_blocks}'"),
    "{stmt_chain}": "the whole single statement / statement range at chain fidelity",
    "{cond}": "the parenthesised condition of an if/while",
    "{class_name}": "class/interface name",
    "{bases}": "extends + implements names, comma-joined",
    "{init_fields}": "field names assigned this.x=x in the constructor",
    "{class_vars}": "declared field/property names",
    "{method_sigs}": "';'-joined 'name(params): ret' for every method",
    "{call_target}": "callee, e.g. logger.info or new Map",
    "{call_args_combined}": "all arguments, comma-joined, compacted",
    "{first_arg}": "first argument only",
    "{rest_args}": "arguments after the first, comma-joined",
    "{var}": "left operand", "{op_sym}": "the operator", "{lit}": "right operand",
}


def slot_menu_ts(region_type: str) -> str:
    vocab = SLOT_VOCAB_TS.get(region_type, {})
    return "\n".join(f"  {s}  — {_SLOT_DESCRIPTIONS_TS.get(s, '')}" for s in vocab)


PROMPT_SYNTH_TS = """You design ONE rewrite template for a recurring TYPESCRIPT code structure.
You are given {n} instances of region_type={rt}. The template rewrites such a
region into a SHORTER code-like sketch a downstream LLM can still read.

RULES (hard):
- Use ONLY these slots for region_type={rt}:
{slot_menu}
- Everything outside slots must be plain ASCII TS tokens. Allowed: braces,
  semicolons, parentheses, colons, TS keywords. FORBIDDEN as connectors:
  '->' and '=>' (arrow-function collision); also no invented symbols, prose,
  '# N calls'-style counts, placeholder words like <expr>.
- Keep the dependency signature: names, typed signature (incl. return type),
  WHICH calls in WHAT order, control-flow shape, return/throw. Drop only
  recoverable glue: comments, formatting, brace-only lines, getter/setter
  boilerplate, this.x=x assignments (when {{init_fields}} keeps the names).
- Under-compression is acceptable; losing the signature is not.

GOOD canonical forms:
- method combiner:  {{modifiers}} {{func_name}}{{sig}}: {{body_blocks}}
- statement chain:  {{stmt_chain}}
- call idiom:       {{call_target}}({{call_args_combined}})
- boilerplate class: class {{class_name}} extends {{bases}} fields[{{class_vars}}]

{feedback_block}INSTANCES:
{instances_block}

Reply with ONLY a JSON object: {{"template": "...", "rationale": "..."}}
"""

PROMPT_CRITIC_TS = """You judge whether a TypeScript rewrite template is SAFE on one instance.
SAFE = applying it keeps the dependency signature (identity, typed signature,
which calls in what order, control-flow shape, return/throw) and drops only
recoverable glue. UNSAFE = information a reader needs is silently lost.

TEMPLATE: {template}
INSTANCE:
```typescript
{instance}
```
REWRITTEN BY TEMPLATE:
```
{rendered}
```
Reply with ONLY JSON: {{"safe": true/false, "reason": "...", "suggestion": "..."}}
"""

PROMPT_GENERALITY_TS = """A TS rewrite template was designed on one cluster. Decide whether it is
safe to apply BROADLY to OTHER regions of the same kind. Judge each instance
SAFE/UNSAFE as a safety critic would.

TEMPLATE: {template}

INSTANCES:
{instances_block}

Reply with ONLY JSON:
{{"verdict": "BROAD" or "NARROW", "per_instance": [{{"i": 1, "safe": true/false, "reason": "..."}}]}}
"BROAD" only if EVERY instance is safe.
"""
