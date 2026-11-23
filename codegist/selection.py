"""Select compatible rewrites by tree DP and weighted interval scheduling."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .candidate_gen import Candidate


_BODY_ATTRS = ("body", "orelse", "finalbody", "handlers")


def _child_nodes(node):
    if isinstance(node, ast.AST):
        return ast.iter_child_nodes(node)
    return getattr(node, "children", ()) or ()


def _stmt_block_specs(node):
    """Yield statement blocks and Python exception handlers for selection."""
    if isinstance(node, ast.AST):
        for attr in _BODY_ATTRS:
            val = getattr(node, attr, None)
            if isinstance(val, list) and val \
                    and all(isinstance(s, ast.stmt) for s in val):
                yield ("block", node, attr, val)
            elif attr == "handlers" and isinstance(val, list) and val:
                yield ("handlers", node, attr, val)
        return
    blocks = getattr(node, "stmt_blocks", None)
    if callable(blocks):
        for parent, attr, stmts in (blocks() or ()):
            if stmts:
                yield ("block", parent, attr, stmts)


@dataclass
class _NodeIndex:
    """Candidates rooted at a specific AST node (whole-subtree rewrites)."""
    by_node: Dict[int, List[Candidate]]   # id(node) -> candidates
    # Candidates over a contiguous slice of `parent.<body_attr>`.
    by_block: Dict[Tuple[int, str], List[Candidate]]  # (id(parent), attr) -> cands


Cache = Dict[Tuple[int, bool], Tuple[int, List[Candidate]]]


def _g(c: Candidate, nested: bool) -> int:
    """Score a nested rewrite against the hole text already accounted for by its parent."""
    if not nested or c.gain_compact is None:
        return c.gain
    return c.gain_compact


def _build_index(candidates: List[Candidate]) -> _NodeIndex:
    by_node: Dict[int, List[Candidate]] = defaultdict(list)
    by_block: Dict[Tuple[int, str], List[Candidate]] = defaultdict(list)
    for c in candidates:
        if c.span_kind == "node":
            assert c.node is not None
            by_node[id(c.node)].append(c)
        else:
            assert c.parent_node is not None
            by_block[(id(c.parent_node), c.body_attr)].append(c)
    return _NodeIndex(by_node=by_node, by_block=by_block)


def _score_with_inners(
    cand: Candidate, idx: _NodeIndex, cache: Cache, nested: bool
) -> Tuple[int, List[Candidate]]:
    score = _g(cand, nested)
    choice: List[Candidate] = [cand]
    for inner in cand.inner_nodes:
        ig, ic = _solve(inner, idx, cache, True)
        score += ig
        choice.extend(ic)
    return score, choice


def _solve_block(
    parent: ast.AST,
    body_attr: str,
    body: List[ast.stmt],
    idx: _NodeIndex,
    cache: Cache,
    nested: bool,
) -> Tuple[int, List[Candidate]]:
    n = len(body)
    stmt_gain: List[int] = []
    stmt_choice: List[List[Candidate]] = []
    for s in body:
        g, c = _solve(s, idx, cache, nested)
        stmt_gain.append(g)
        stmt_choice.append(c)
    range_cands = idx.by_block.get((id(parent), body_attr), [])
    f = [0] * (n + 1)
    pick: List[tuple] = [("none",)] * (n + 1)
    for i in range(1, n + 1):
        # default: keep stmt i-1 with its own subtree solve
        f[i] = f[i - 1] + stmt_gain[i - 1]
        pick[i] = ("stmt", i - 1)
        for c in range_cands:
            if c.last_idx != i - 1:
                continue
            first = c.first_idx
            if first < 0 or first >= n:
                continue
            score, _choice = _score_with_inners(c, idx, cache, nested)
            total = f[first] + score
            if total > f[i]:
                f[i] = total
                pick[i] = ("range", c)
    # back-trace
    chosen: List[Candidate] = []
    i = n
    while i > 0:
        kind = pick[i][0]
        if kind == "stmt":
            j: int = pick[i][1]
            chosen.extend(stmt_choice[j])
            i -= 1
        elif kind == "range":
            c: Candidate = pick[i][1]
            _score, inner_choice = _score_with_inners(c, idx, cache, nested)
            chosen.extend(inner_choice)
            i = c.first_idx
        else:
            i -= 1
    return f[n], chosen


def _solve(
    node: ast.AST, idx: _NodeIndex, cache: Cache, nested: bool
) -> Tuple[int, List[Candidate]]:
    """Return the best score and chosen rewrites for this node and context."""
    cached = cache.get((id(node), nested))
    if cached is not None:
        return cached

    if nested:
        # The renderer can inline a rewrite rooted at this exact hole node.
        # Leaving the hole as default text contributes zero additional gain.
        best_gain, best_choice = 0, []
        for cand in idx.by_node.get(id(node), []):
            score, choice = _score_with_inners(cand, idx, cache, True)
            if score > best_gain:
                best_gain, best_choice = score, choice
        cache[(id(node), nested)] = (best_gain, best_choice)
        return best_gain, best_choice

    # Choice (a): no whole-node rewrite — accumulate best gains from children.
    inner_gain = 0
    inner_choice: List[Candidate] = []
    handled_children: set[int] = set()

    for kind, parent, attr, val in _stmt_block_specs(node):
        if kind == "block":
            g, c = _solve_block(parent, attr, val, idx, cache, nested)
            inner_gain += g
            inner_choice.extend(c)
            for s in val:
                handled_children.add(id(s))
        else:  # "handlers" — solved per-child, not interval-scheduled
            for h in val:
                g, c = _solve(h, idx, cache, nested)
                inner_gain += g
                inner_choice.extend(c)
                handled_children.add(id(h))

    for child in _child_nodes(node):
        if id(child) in handled_children:
            continue
        g, c = _solve(child, idx, cache, nested)
        inner_gain += g
        inner_choice.extend(c)

    # Choice (b): every candidate rooted at v as a partial-subtree rewrite
    # whose inner holes remain open for further compression.
    best_gain = inner_gain
    best_choice = inner_choice
    for cand in idx.by_node.get(id(node), []):
        score, choice = _score_with_inners(cand, idx, cache, nested)
        # Preserve disjoint rewrites on descendants outside the source span,
        # such as decorators preceding a Python function or class definition.
        outside = [c for c in inner_choice
                   if c.end_offset <= cand.start_offset
                   or c.start_offset >= cand.end_offset]
        if outside:
            score += sum(_g(c, nested) for c in outside)
            choice = choice + outside
        if score > best_gain:
            best_gain = score
            best_choice = choice

    cache[(id(node), nested)] = (best_gain, best_choice)
    return best_gain, best_choice


@dataclass
class SelectionResult:
    total_gain: int
    chosen: List[Candidate]
    skipped: List[Candidate]   # candidates not picked (for reporting)


def _greedy_select(candidates: List[Candidate]) -> List[Candidate]:
    chosen: List[Candidate] = []
    taken: List[Tuple[int, int]] = []
    for c in sorted(candidates, key=lambda c: (-c.gain, c.start_offset)):
        s, e = c.start_offset, c.end_offset
        if all(e <= ts or te <= s for ts, te in taken):
            chosen.append(c)
            taken.append((s, e))
    return chosen


def select(tree: ast.AST, candidates: List[Candidate],
           src: str | None = None) -> SelectionResult:
    """Maximize compatible candidate scores as an estimate of final token savings."""
    idx = _build_index(candidates)
    cache: Cache = {}
    total, chosen = _solve(tree, idx, cache, False)
    chosen_ids = {id(c) for c in chosen}
    skipped = [c for c in candidates if id(c) not in chosen_ids]
    return SelectionResult(total_gain=total, chosen=chosen, skipped=skipped)
