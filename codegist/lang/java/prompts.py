"""Prompts for synthesizing and checking Java templates."""
from __future__ import annotations

from .slots import SLOT_VOCAB_JAVA

_SLOT_DESCRIPTIONS_JAVA = {
    "{func_name}": "method/constructor name",
    "{ret_type}": "return type (empty for constructors)",
    "{sig}": "typed parameter list incl. throws, e.g. (int a,String b) throws IOException",
    "{body_chain}": ("dependency chain of the body: short statements verbatim, "
                     "long ones as ordered callee chains, control flow keeps "
                     "head + braces; statements end with ';'"),
    "{body_blocks}": ("the body's top-level statements, EACH kept as its own "
                      "open hole for block-level schemas (the combiner slot); "
                      "use in '{ret_type} {func_name}{sig}: {body_blocks}'"),
    "{stmt_chain}": "the whole single statement / statement range at chain fidelity",
    "{cond}": "the parenthesised condition of an if/while",
    "{class_name}": "class/interface name",
    "{bases}": "extends + implements names, comma-joined",
    "{init_fields}": "field names assigned this.x=x in constructors",
    "{class_vars}": "declared field names",
    "{method_sigs}": "';'-joined 'RetType name(params)' for every method",
    "{call_target}": "callee, e.g. logger.info or new ArrayList",
    "{call_args_combined}": "all arguments, comma-joined, compacted",
    "{first_arg}": "first argument only",
    "{rest_args}": "arguments after the first, comma-joined",
    "{var}": "left operand", "{op_sym}": "the operator", "{lit}": "right operand",
}


def slot_menu_java(region_type: str) -> str:
    vocab = SLOT_VOCAB_JAVA.get(region_type, {})
    lines = [f"  {s}  — {_SLOT_DESCRIPTIONS_JAVA.get(s, '')}" for s in vocab]
    return "\n".join(lines)


PROMPT_SYNTH_JAVA = """You design ONE rewrite template for a recurring JAVA code structure.
You are given {n} instances of region_type={rt}. The template rewrites such a
region into a SHORTER code-like sketch that a downstream LLM can still read.

RULES (hard):
- Use ONLY these slots for region_type={rt}:
{slot_menu}
- Everything outside slots must be plain ASCII Java tokens. Allowed: braces,
  semicolons, parentheses, Java keywords. FORBIDDEN: '->' or '=>' as
  connectors (lambda collision), invented symbols, prose, '# N calls'-style
  counts, placeholder words like <expr>.
- Keep the dependency signature: names, typed signature, WHICH calls in WHAT
  order, control-flow shape, return/throw contract. Drop only recoverable
  glue: comments, formatting, brace-only lines, getter/setter boilerplate,
  this.x=x constructor assignments (when {{init_fields}} keeps the names).
- Under-compression is acceptable; losing the signature is not.

GOOD canonical forms:
- method combiner:  {{ret_type}} {{func_name}}{{sig}}: {{body_blocks}}
- statement chain:  {{stmt_chain}}
- call idiom:       {{call_target}}({{call_args_combined}})
- pure boilerplate class: class {{class_name}} extends {{bases}} fields[{{class_vars}}]

{feedback_block}INSTANCES:
{instances_block}

Reply with ONLY a JSON object: {{"template": "...", "rationale": "..."}}
"""

PROMPT_CRITIC_JAVA = """You judge whether a Java rewrite template is SAFE on one concrete instance.
SAFE means: applying the template keeps the dependency signature (identity,
typed signature, which calls in what order, control-flow shape, return/throw
contract) and drops only recoverable glue (comments, formatting, brace-only
lines, this.x=x bodies when field names survive). UNSAFE means information a
reader needs is silently lost (dropped calls/arguments that are the content,
reordered effects, prose/invented notation).

TEMPLATE: {template}
INSTANCE:
```java
{instance}
```
REWRITTEN BY TEMPLATE:
```
{rendered}
```
Reply with ONLY JSON: {{"safe": true/false, "reason": "...", "suggestion": "..."}}
"""

PROMPT_GENERALITY_JAVA = """A Java rewrite template was designed on one cluster of similar regions.
Decide whether it is safe to apply BROADLY to OTHER regions of the same kind.
For each instance below, judge SAFE/UNSAFE exactly as a safety critic would
(dependency signature kept, only glue dropped).

TEMPLATE: {template}

INSTANCES:
{instances_block}

Reply with ONLY JSON:
{{"verdict": "BROAD" or "NARROW", "per_instance": [{{"i": 1, "safe": true/false, "reason": "..."}}]}}
"BROAD" only if EVERY instance is safe.
"""
