from __future__ import annotations

from typing import Callable, List, Sequence, Optional, Dict, Any, Tuple, Iterable, Protocol
import re

from .trie_builder import TrieNode, count_tail_L2R
from .trace_utils import Trace


def peel_end_tokens(
    tokens: Sequence[str],
    count_tail: Callable[[Sequence[str]], int],
    steps: int = 4,
    max_k: int = 2,
) -> List[str]:
    """
    Deterministic tail peeling by counts.

    Iteratively drop up to `max_k` final tokens (default 2) from the messy address when doing so
    "joins a larger subtree" in the canonical trie, as measured by the count
    for the last token (anchor) increasing.

    Rule per step:
      - Let base = count_tail([last_token]).
      - For k in {1..max_k}, check new_base = count_tail([token_at(-k-1)]).
      - If any k yields new_base > base, drop the last k tokens (choose the
        k with the largest strictly-positive increase). Otherwise stop.

    This removes tails like "... HERTFORDSHIRE ENGLAND" while keeping
    informative locality tails like "... KINGS LANGLEY" intact.
    """
    if not tokens:
        return []

    out = list(tokens)
    for step_idx in range(max(0, int(steps))):
        if len(out) <= 1:
            break

        anchor = out[-1]
        base = count_tail([anchor])
        best_k = 0
        best_score = base

        max_try = min(int(max_k), len(out) - 1)
        for k in range(1, max_try + 1):
            new_anchor = out[-k - 1]
            score = count_tail([new_anchor])
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 0:
            out = out[:-best_k]
        else:
            break

    return out


def peel_end_tokens_with_trie(
    tokens: Sequence[str],
    root: TrieNode,
    steps: int = 4,
    max_k: int = 2,
    trace: Optional[Trace] = None,
) -> List[str]:
    """Thin wrapper wiring peel_end_tokens to the trie count helper."""

    def _count_tail(tail: Sequence[str]) -> int:
        return count_tail_L2R(root, tail)

    original = list(tokens)
    peeled = peel_end_tokens(tokens, _count_tail, steps=steps, max_k=max_k)
    if trace is not None:
        trace.set_tokens(tokens)
        k = len(tokens) - len(peeled)
        if k > 0:
            removed = list(tokens[-k:])
            # Recompute summary counts for observability (cheap and clear)
            base_anchor = original[-1]
            base_score = _count_tail([base_anchor])
            best_anchor = original[-k - 1]
            best_score = _count_tail([best_anchor])
            trace.add(
                {
                    "action": "PEEL_TAIL",
                    "removed_tokens": removed,
                    "k": k,
                    "base_anchor": base_anchor,
                    "base_score": int(base_score),
                    "best_anchor": best_anchor,
                    "best_score": int(best_score),
                }
            )
    return peeled


def walk_exact(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    accept_terminal_if_exhausted: bool = True,
) -> Optional[int]:
    """
    Consume tokens right-to-left using exact child transitions only.

    Accept if either:
      A) node has UPRN AND count==1 AND (no next-token descent)
      B) accept_terminal_if_exhausted AND all messy tokens are consumed AND node has UPRN

    Otherwise continue consuming or reject when stuck / at non-terminal.
    """
    node = root
    t = list(reversed([str(x) for x in tokens_L2R]))

    i = 0
    n = len(t)
    while True:
        # Check acceptance at current node before attempting to consume next token
        if node.uprn is not None:
            # A) Unique & blocked (strict, unchanged)
            if node.count == 1 and (i >= n or not node.has_child(t[i])):
                return node.uprn
            # B) Exact-exhausted terminal
            if accept_terminal_if_exhausted and i >= n:
                return node.uprn

        if i >= n:
            return None

        nxt = t[i]
        child = node.child(nxt)
        if child is None:
            return None
        node = child
        i += 1


def match_stage1_exact_only(tokens_L2R: Sequence[str], root: TrieNode) -> Optional[int]:
    """
    Stage‑1 (Step‑4): Peeling + exact walk.

    1) Peel redundant tail tokens by counts (up to 2 by default).
    2) Walk exactly right‑to‑left and accept according to Step‑3 rules
       (unique & blocked, or exact‑exhausted terminal).
    """
    toks = peel_end_tokens_with_trie(tokens_L2R, root, steps=4, max_k=2)
    return walk_exact(toks, root, accept_terminal_if_exhausted=True)


_NUMERIC_RE = re.compile(r"^\d+[A-Z]?$")


def is_numeric(tok: str) -> bool:
    """Return True for numeric-ish tokens like '19', '23A'."""
    return bool(_NUMERIC_RE.fullmatch(tok))


def dl1_edit_type(a: str, b: str) -> Optional[str]:
    """
    Return the type of DL≤1 edit from a→b, or None if distance > 1.
    Types: 'substitution', 'transpose', 'insert', 'delete'.
    """
    if a == b:
        return "exact"
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return None
    # Equal length
    if la == lb:
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diffs) == 1:
            return "substitution"
        if len(diffs) == 2:
            i, j = diffs
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return "transpose"
        return None
    # Length differs by 1
    # Ensure a is longer for delete case
    if la == lb + 1:
        # delete one char from a to match b
        i = j = 0
        edits = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                edits += 1
                if edits > 1:
                    return None
                i += 1
        return "delete"
    if lb == la + 1:
        # insert one char into a to match b
        i = j = 0
        edits = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                edits += 1
                if edits > 1:
                    return None
                j += 1
        return "insert"
    return None


def match_stage1_with_skips(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    max_cost: int = 2,
    min_exact_hits: int = 2,
    require_numeric: bool = True,
    numeric_must_be_exact: bool = True,
    skip_redundant_ratio: float = 2.0,
    accept_terminal_if_exhausted: bool = True,
    trace: Optional[Trace] = None,
) -> Optional[int]:
    """Wrapper around the internal search to return just the UPRN."""
    uprn, best_cost, runner_cost, _best_state, _parents = _search_with_skips(
        tokens_L2R,
        root,
        max_cost=max_cost,
        min_exact_hits=min_exact_hits,
        require_numeric=require_numeric,
        numeric_must_be_exact=numeric_must_be_exact,
        skip_redundant_ratio=skip_redundant_ratio,
        accept_terminal_if_exhausted=accept_terminal_if_exhausted,
        allow_swap_adjacent=False,
        swap_cost=1,
        trace=trace,
    )
    return uprn


# --- High-level Stage-1 API (Step 7) ---
from dataclasses import dataclass


@dataclass
class Params:
    max_cost: int = 2
    min_exact_hits: int = 2
    require_numeric: bool = True
    numeric_must_be_exact: bool = True
    skip_redundant_ratio: float = 2.0
    accept_terminal_if_exhausted: bool = True
    max_uprns_to_return: int = 10
    allow_swap_adjacent: bool = False
    swap_cost: int = 1
    # Canonical insert (Stage-1: exact-only lookahead)
    allow_canonical_insert: bool = True
    canonical_insert_cost: int = 1
    canonical_insert_allow_fuzzy: bool = False
    canonical_insert_max_candidates: int = 3
    canonical_insert_disallow_numeric: bool = True


# --- Step 1: Light transition types (no behavior change) ---

@dataclass(frozen=True)
class _State:
    """Immutable snapshot of the search state.

    Fields mirror the tuple used in the heap/seen keys.
    """
    node: TrieNode
    i: int
    exact_hits: int
    saw_num_any: bool
    saw_num_exact: bool


@dataclass(frozen=True)
class _Move:
    """Proposed transition produced by a rule.

    - node: target trie node after applying the move
    - i_delta / exact_delta: increments to index and exact hit counters
    - saw_num_any / saw_num_exact: updated numeric flags
    - cost_delta: additional cost (0 or 1)
    - event: trace event dict to record for this move
    - last_consume_m_index / last_canon_label: optional hints to place accept star
    """
    node: TrieNode
    i_delta: int
    exact_delta: int
    saw_num_any: bool
    saw_num_exact: bool
    cost_delta: int
    event: Dict[str, Any]
    last_consume_m_index: Optional[int] = None
    last_canon_label: Optional[str] = None


class _RuleFunc(Protocol):
    def __call__(
        self, state: _State, tokens_r2l: Sequence[str], n: int, params: "Params"
    ) -> Iterable[_Move]:
        ...


def _can_finish_exact_to_terminal(
    node: TrieNode, tokens_r2l: Sequence[str], start_i: int
) -> Tuple[bool, int, Optional[int]]:
    """Probe whether the exact path from node using tokens[start_i:] reaches a terminal.

    Returns (ok, steps_consumed, uprn_if_ok). No side effects or tracing.
    """
    n = len(tokens_r2l)
    cur = node
    i = int(start_i)
    steps = 0
    while i < n:
        tok = tokens_r2l[i]
        ch = cur.child(tok)
        if ch is None:
            return (False, steps, None)
        cur = ch
        steps += 1
        i += 1
    if cur.uprn is not None:
        return (True, steps, int(cur.uprn))
    return (False, steps, None)


def _rule_exact(state: _State, tokens_r2l: Sequence[str], n: int, params: "Params") -> Iterable[_Move]:
    """Yield an exact-consume Move if the next messy token matches a child.

    Behavior preserved: cost 0, increments i and exact_hits, updates numeric flags
    based on the messy token, and records child/anchor counts in the event.
    """
    if state.i >= n:
        return
    tok = tokens_r2l[state.i]
    child = state.node.child(tok)
    if child is None:
        return
    m_index = (n - 1) - state.i
    ev = {
        "action": "EXACT_DESCEND",
        "messy": tok,
        "canon": tok,
        "m_index": m_index,
        "child_count": child.count,
        "anchor_count": int(state.node.count),
    }
    yield _Move(
        node=child,
        i_delta=1,
        exact_delta=1,
        saw_num_any=(state.saw_num_any or is_numeric(tok)),
        saw_num_exact=(state.saw_num_exact or is_numeric(tok)),
        cost_delta=0,
        event=ev,
        last_consume_m_index=m_index,
        last_canon_label=tok,
    )


def _rule_skip(state: _State, tokens_r2l: Sequence[str], n: int, params: "Params") -> Iterable[_Move]:
    """Yield a single skip Move (redundant or penalized) for the next messy token.

    Behavior preserved: numeric flags unchanged; cost determined by _calc_skip_cost;
    trace event includes anchor/child counts, ratio, and threshold.
    """
    if state.i >= n:
        return
    tok = tokens_r2l[state.i]
    s_cost = _calc_skip_cost(state.node, tok, params.skip_redundant_ratio)
    anchor_count = int(state.node.count)
    child_count = int(state.node.child_count(tok))
    ratio = anchor_count / max(1, child_count)
    m_index = (n - 1) - state.i
    ev = {
        "action": "SKIP_REDUNDANT" if s_cost == 0 else "SKIP_PENALIZED",
        "messy": tok,
        "m_index": m_index,
        "anchor_count": anchor_count,
        "child_count": child_count,
        "ratio": float(ratio),
        "threshold": float(params.skip_redundant_ratio),
    }
    yield _Move(
        node=state.node,
        i_delta=1,
        exact_delta=0,
        saw_num_any=state.saw_num_any,
        saw_num_exact=state.saw_num_exact,
        cost_delta=s_cost,
        event=ev,
    )


def _rule_fuzzy(state: _State, tokens_r2l: Sequence[str], n: int, params: "Params") -> Iterable[_Move]:
    """Yield fuzzy-consume Moves when no exact child exists and DL<=1 applies.

    Behavior preserved: cost 1; numeric flags update for saw_num_any using the
    canonical label; saw_num_exact unchanged; event includes edit_type and child_count.
    """
    if state.i >= n:
        return
    tok = tokens_r2l[state.i]
    # Only consider fuzzy when exact is absent
    if state.node.child(tok) is not None:
        return
    for lbl, ch in state.node.iter_children():
        etype = dl1_edit_type(tok, lbl)
        if etype is not None and etype != "exact":
            m_index = (n - 1) - state.i
            ev = {
                "action": "FUZZY_CONSUME",
                "messy": tok,
                "canon": lbl,
                "m_index": m_index,
                "edit_type": etype,
                "child_count": ch.count,
            }
            yield _Move(
                node=ch,
                i_delta=1,
                exact_delta=0,
                saw_num_any=(state.saw_num_any or is_numeric(lbl)),
                saw_num_exact=state.saw_num_exact,
                cost_delta=1,
                event=ev,
                last_consume_m_index=m_index,
                last_canon_label=lbl,
            )


def _rule_canonical_insert(
    state: _State, tokens_r2l: Sequence[str], n: int, params: "Params"
) -> Iterable[_Move]:
    """Insert one canonical token, then consume current messy token under it.

    Preconditions:
      - state.i < n
      - No exact child for current messy token at this node
      - Consider children X of current node such that X has a child Y whose
        label equals the current messy token. If too many candidates, abort.

    Move effects:
      - Advance i by 1 (consumed messy token exactly at Y)
      - exact_delta += 1
      - cost += canonical_insert_cost
      - Record a standard EXACT_DESCEND event with inserted_canonical metadata.
    """
    if state.i >= n:
        return
    tok = tokens_r2l[state.i]
    # New guard: if the non-insert exact path can finish to a terminal, skip insert
    ok_no_insert, _, _ = _can_finish_exact_to_terminal(state.node, tokens_r2l, state.i)
    if ok_no_insert:
        return

    # Gather viable (X_label, X_node, Y_node) candidates
    candidates: List[Tuple[str, TrieNode, TrieNode]] = []
    for x_label, x_node in state.node.iter_children():
        if params.canonical_insert_disallow_numeric and is_numeric(x_label):
            continue
        # Conservative guard: do not insert a canonical token that already
        # appears anywhere in the remaining messy/token stream (avoid reordering fixes)
        if x_label in tokens_r2l:
            continue
        y = x_node.child(tok)
        if y is not None:
            # Require that after inserting X and consuming current tok at Y, the
            # remaining tokens can finish exactly to a terminal.
            ok_tail, steps_tail, uprn_tail = _can_finish_exact_to_terminal(
                y, tokens_r2l, state.i + 1
            )
            if ok_tail:
                candidates.append((x_label, x_node, y))

    if not candidates:
        return
    if len(candidates) > int(params.canonical_insert_max_candidates):
        # Abort if too many options at this anchor
        return

    m_index = (n - 1) - state.i
    for x_label, x_node, y_node in candidates:
        ev = {
            "action": "EXACT_DESCEND",
            "messy": tok,
            "canon": tok,
            "m_index": m_index,
            "child_count": int(y_node.count),
            # anchor_count reflects the inserted step (X)
            "anchor_count": int(x_node.count),
            # non-rendered metadata for observability
            "inserted_canonical": x_label,
        }
        yield _Move(
            node=y_node,
            i_delta=1,
            exact_delta=1,
            saw_num_any=(state.saw_num_any or is_numeric(tok)),
            saw_num_exact=(state.saw_num_exact or is_numeric(tok)),
            cost_delta=int(params.canonical_insert_cost),
            event=ev,
            last_consume_m_index=m_index,
            last_canon_label=tok,
        )


def _rule_swap_adjacent(state: _State, tokens_r2l: Sequence[str], n: int, params: "Params") -> Iterable[_Move]:
    """Swap two adjacent messy tokens if trie supports reversed canonical order.

    Preconditions:
      - exact at current position does not apply
      - state.i + 1 < n
      - node.child(tok1).child(tok0) exists
    """
    if not params.allow_swap_adjacent:
        return
    if state.i >= n - 1:
        return
    tok0 = tokens_r2l[state.i]
    # Only consider swap if exact isn't available
    if state.node.child(tok0) is not None:
        return
    tok1 = tokens_r2l[state.i + 1]
    child1 = state.node.child(tok1)
    if child1 is None:
        return
    child2 = child1.child(tok0)
    if child2 is None:
        return
    m_index0 = (n - 1) - state.i
    m_index1 = (n - 1) - (state.i + 1)
    ev = {
        "action": "SWAP_ADJACENT",
        "messy_pair": [tok0, tok1],
        "canon_pair": [tok1, tok0],
        "m_index0": m_index0,
        "m_index1": m_index1,
        "anchor_count": int(state.node.count),
        "child_count_after_first": int(child1.count),
        "child_count_after_second": int(child2.count),
    }
    yield _Move(
        node=child2,
        i_delta=2,
        exact_delta=2,
        saw_num_any=(state.saw_num_any or is_numeric(tok0) or is_numeric(tok1)),
        saw_num_exact=(state.saw_num_exact or is_numeric(tok0) or is_numeric(tok1)),
        cost_delta=int(params.swap_cost),
        event=ev,
        last_consume_m_index=m_index1,
        last_canon_label=tok0,
    )


def match_stage1(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    params: Params = Params(),
    trace: Optional[Trace] = None,
) -> Dict[str, Any]:
    """
    Stage‑1 matcher: peel → exact/skip/fuzzy search with strict guards.

    Returns a structured result dict with `matched`, `uprn`, `cost`, and the
    peeled tokens used for matching.
    """
    peeled = peel_end_tokens_with_trie(tokens_L2R, root, steps=4, max_k=2, trace=trace)

    # Reuse existing search; adapt to return cost by probing heap order would
    # require refactor. For now, run the same logic and infer cost via a tiny
    # local duplicate of acceptance logic. To keep changes surgical, we call the
    # existing function and set cost=None; we can upgrade to return cost in Step 10.
    # Use a local trace handle to enable parent-map reconstruction even if caller doesn't pass one
    search_trace = trace if trace is not None else Trace(enabled=False)
    uprn, best_cost, _, best_state, parents = _search_with_skips(
        peeled,
        root,
        max_cost=params.max_cost,
        min_exact_hits=params.min_exact_hits,
        require_numeric=params.require_numeric,
        numeric_must_be_exact=params.numeric_must_be_exact,
        skip_redundant_ratio=params.skip_redundant_ratio,
        accept_terminal_if_exhausted=params.accept_terminal_if_exhausted,
        allow_swap_adjacent=params.allow_swap_adjacent,
        swap_cost=params.swap_cost,
        trace=search_trace,
    )

    # Reconstruct chosen path whenever we have a best_state (accept or partial)
    if trace is not None and best_state is not None and parents is not None:
        ordered: List[Dict[str, Any]] = []
        cur = best_state
        # Walk back through parents; stop when no entry is available
        while cur in parents:
            info = parents[cur]
            ev = info.get("event")
            if ev:
                ordered.append(ev)
            nxt = info.get("parent")
            if nxt is None or nxt == cur:
                break
            cur = nxt
        ordered.reverse()

        # Append ACCEPT if captured on the best_state (takes precedence)
        acc_info = parents.get(best_state, {})
        ev_acc = acc_info.get("accept_event")
        if ev_acc is not None:
            ordered.append(ev_acc)

        # Determine if we already have an ACCEPT_* at the end
        has_accept = bool(ordered and str(ordered[-1].get("action", "")).startswith("ACCEPT_"))

        if not has_accept:
            # Diagnose a stop condition and add a STOP_* marker
            info_best = parents.get(best_state, {})
            node = info_best.get("node")
            # Unpack state
            _, i_consumed, exact_hits, saw_any, saw_exact = best_state
            t_r2l = list(reversed(peeled))
            n = len(t_r2l)

            def last_consumed_m_index() -> int:
                return (n - 1) - (int(i_consumed) - 1) if int(i_consumed) > 0 else (n - 1 if n > 0 else 0)

            # If the last step was a SKIP_*, prefer STOP_NO_CHILD on that column
            if ordered and str(ordered[-1].get("action", "")).startswith("SKIP_"):
                last_ev = ordered[-1]
                msg_tok = last_ev.get("messy")
                cc = int(node.child_count(msg_tok)) if (node is not None and msg_tok is not None) else None
                ordered.append({
                    "action": "STOP_NO_CHILD",
                    "m_index": int(ordered[-1]["m_index"]),
                    "messy": msg_tok,
                    "anchor_count": (int(node.count) if node is not None else None),
                    "child_count": (int(cc) if cc is not None else 0),
                })
            elif int(i_consumed) < n and node is not None:
                next_tok = t_r2l[i_consumed]
                if not node.has_child(next_tok):
                    ordered.append({
                        "action": "STOP_NO_CHILD",
                        "m_index": (n - 1) - int(i_consumed),
                        "messy": next_tok,
                        "anchor_count": int(node.count),
                        "child_count": 0,
                    })
                else:
                    if params.require_numeric and not (saw_exact if params.numeric_must_be_exact else saw_any):
                        ordered.append({
                            "action": "STOP_GUARD_NUMERIC",
                            "m_index": last_consumed_m_index(),
                            "require_numeric": True,
                            "numeric_must_be_exact": params.numeric_must_be_exact,
                            "saw_num_exact": bool(saw_exact),
                            "saw_num_any": bool(saw_any),
                        })
                    elif exact_hits < params.min_exact_hits:
                        ordered.append({
                            "action": "STOP_GUARD_MIN_EXACT",
                            "m_index": last_consumed_m_index(),
                            "exact_hits": int(exact_hits),
                            "min_exact_hits": int(params.min_exact_hits),
                        })
                    elif node.uprn is not None and node.count > 1:
                        ordered.append({
                            "action": "STOP_AMBIGUOUS",
                            "m_index": last_consumed_m_index(),
                            "node_count": int(node.count),
                        })
                    else:
                        ordered.append({
                            "action": "STOP_UNKNOWN",
                            "m_index": last_consumed_m_index(),
                            "node_count": (int(node.count) if node is not None else None),
                        })
            else:
                if node is not None and node.uprn is None:
                    ordered.append({
                        "action": "STOP_INCOMPLETE",
                        "m_index": last_consumed_m_index(),
                        "node_count": int(node.count),
                    })
                else:
                    ordered.append({"action": "STOP_UNKNOWN", "m_index": last_consumed_m_index()})
        else:
            # already appended accept; ensure star sits on the last consumed token
            # Find last consume step along the ordered path
            last_consume_idx = None
            last_consume_label = None
            for ev in reversed(ordered):
                a = ev.get("action")
                if a in ("EXACT_DESCEND", "FUZZY_CONSUME"):
                    last_consume_idx = int(ev.get("m_index"))
                    last_consume_label = ev.get("canon")
                    break
            if last_consume_idx is not None and ordered:
                acc_ev = ordered[-1]
                if str(acc_ev.get("action", "")).startswith("ACCEPT_"):
                    acc_ev["at_m_index"] = last_consume_idx
                    if last_consume_label is not None:
                        acc_ev["accepted_label"] = last_consume_label

        for ev in ordered:
            trace.add(ev)

    # Step 3: reconstruct consumed path and counts (post-processing)
    from .trace_utils import (
        reconstruct_consumed_events,
        events_to_consumed_path,
        final_node_from_state,
        collect_uprns,
    )

    consumed_events = reconstruct_consumed_events(best_state, parents)
    consumed_path, consumed_counts = events_to_consumed_path(consumed_events)
    final_node = final_node_from_state(best_state, parents)
    final_node_count = int(final_node.count) if final_node is not None else None

    out: Dict[str, Any] = {
        "matched": uprn is not None,
        "uprn": uprn,
        "cost": best_cost,
        # New preferred name
        "search_tokens": peeled,
        # Back-compat alias (TODO: remove in a future cleanup)
        "peeled_tokens": peeled,
        "input_tokens": list(tokens_L2R),
        "consumed_path": consumed_path,
        "consumed_path_counts": consumed_counts,
        "final_node_count": final_node_count,
    }
    # Step 4: candidate enumeration on small no-match subtrees
    if uprn is None and final_node is not None and final_node_count is not None:
        limit = int(params.max_uprns_to_return)
        if final_node_count <= limit:
            cands = collect_uprns(final_node, limit)
            if cands:
                out["candidate_uprns"] = cands
                out["limit_used"] = limit
    return out


def match_address(
    tokens: Sequence[str],
    trie: TrieNode,
    params: Params = Params(),
) -> Dict[str, Any]:
    """
    Thin convenience wrapper around match_stage1 returning the full result dict
    for both match and no-match cases. Pass a Params instance to tune knobs like
    max_uprns_to_return (used for small-subtree candidate enumeration on no-match).
    """
    return match_stage1(tokens, trie, params)


def _search_with_skips(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    max_cost: int = 2,
    min_exact_hits: int = 2,
    require_numeric: bool = True,
    numeric_must_be_exact: bool = True,
    skip_redundant_ratio: float = 2.0,
    accept_terminal_if_exhausted: bool = True,
    allow_swap_adjacent: bool = False,
    swap_cost: int = 1,
    trace: Optional[Trace] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[Tuple[int, int, int, bool, bool]], Optional[Dict[Tuple[int, int, int, bool, bool], Dict[str, Any]]]]:
    """Internal search that returns (uprn, best_cost, runner_cost, best_state, parents)."""
    t = list(reversed([str(x) for x in tokens_L2R]))
    n = len(t)

    import heapq

    # Debug/trace helpers removed

    def accept(node: TrieNode, i: int, exact_hits: int, saw_num: bool) -> bool:
        if node.uprn is None:
            return False
        unique_blocked = node.count == 1 and (i >= n or not node.has_child(t[i]))
        exact_exhausted = accept_terminal_if_exhausted and i >= n
        if not (unique_blocked or exact_exhausted):
            return False
        if exact_hits < min_exact_hits:
            return False
        if require_numeric and not saw_num:
            return False
        return True

    # skip_cost extracted to top-level _calc_skip_cost

    # State & parents for tracing
    StateKey = Tuple[int, int, int, bool, bool]
    ParentMap = Dict[StateKey, Dict[str, Any]]

    parents: ParentMap = {}

    heap: list[tuple[int, int, TrieNode, int, int, bool, bool]] = []
    seq = 0
    heapq.heappush(heap, (0, seq, root, 0, 0, False, False))
    best_cost = float("inf")
    best_uprn: Optional[int] = None
    runner_cost = float("inf")
    seen: dict[tuple[int, int, int, bool, bool], int] = {}
    best_state: Optional[StateKey] = None

    # Track best partial even if no accept
    best_partial_key: Optional[StateKey] = None
    best_partial_cost: float = float("inf")

    def better_partial(c_cost: int, c_exact: int, c_i: int,
                       b_cost: float, b_key: Optional[StateKey]) -> bool:
        if c_cost != b_cost:
            return c_cost < b_cost
        if b_key is None:
            return True
        _, b_i, b_exact, _, _ = b_key
        if c_exact != b_exact:
            return c_exact > b_exact
        return c_i > b_i

    # Step 3: local push helper for moves
    def push_move(cur_state: _State, base_cost: int, move: _Move) -> None:
        nonlocal seq
        new_cost = base_cost + int(move.cost_delta)
        new_i = cur_state.i + int(move.i_delta)
        new_exact = cur_state.exact_hits + int(move.exact_delta)
        new_any = bool(move.saw_num_any)
        new_exact_flag = bool(move.saw_num_exact)
        seq += 1
        heapq.heappush(
            heap,
            (new_cost, seq, move.node, new_i, new_exact, new_any, new_exact_flag),
        )
        if trace is not None:
            cur_key_l: StateKey = (
                id(cur_state.node),
                cur_state.i,
                cur_state.exact_hits,
                cur_state.saw_num_any,
                cur_state.saw_num_exact,
            )
            next_key_l: StateKey = (
                id(move.node),
                new_i,
                new_exact,
                new_any,
                new_exact_flag,
            )
            prev_l = parents.get(next_key_l)
            if prev_l is None or prev_l.get("g_cost", 1e9) > new_cost:
                entry: Dict[str, Any] = {
                    "parent": cur_key_l,
                    "event": move.event,
                    "g_cost": new_cost,
                    "node": move.node,
                }
                if move.event.get("action") in ("EXACT_DESCEND", "FUZZY_CONSUME", "SWAP_ADJACENT"):
                    if move.last_consume_m_index is not None:
                        entry["last_consume_m_index"] = int(move.last_consume_m_index)
                    if move.last_canon_label is not None:
                        entry["last_canon_label"] = str(move.last_canon_label)
                parents[next_key_l] = entry

    # Build a params-like object to pass to rules
    params_like = Params(
        max_cost=max_cost,
        min_exact_hits=min_exact_hits,
        require_numeric=require_numeric,
        numeric_must_be_exact=numeric_must_be_exact,
        skip_redundant_ratio=skip_redundant_ratio,
        accept_terminal_if_exhausted=accept_terminal_if_exhausted,
        allow_swap_adjacent=allow_swap_adjacent,
        swap_cost=swap_cost,
    )

    while heap:
        cost, _, node, i, exact_hits, saw_num_any, saw_num_exact = heapq.heappop(heap)
        if cost > max_cost:
            break
        key = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
        prev = seen.get(key)
        if prev is not None and prev <= cost:
            continue
        seen[key] = cost

        # Update best-partial on pop
        key = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
        if better_partial(cost, exact_hits, i, best_partial_cost, best_partial_key):
            best_partial_cost = cost
            best_partial_key = key

        acc = accept(node, i, exact_hits, saw_num_exact if numeric_must_be_exact else saw_num_any)
        if acc:
            if trace is not None:
                # Determine accept flavor and mark accept event on this terminal state
                unique_blocked = node.count == 1 and (i >= n or not node.has_child(t[i]))
                exact_exhausted = accept_terminal_if_exhausted and i >= n
                # Prefer last consumed token index for star placement
                cur_key_acc: StateKey = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
                info_here = parents.get(cur_key_acc) or {}
                cons_idx = info_here.get("last_consume_m_index")
                acc_label = info_here.get("last_canon_label")
                if cons_idx is None:
                    # Fallback: last consumed by i if available
                    cons_idx = (n - 1) - (i - 1) if i > 0 else ((n - 1) if n > 0 else 0)
                    if acc_label is None and i > 0:
                        acc_label = t[i - 1]

                ev_acc = {
                    "action": "ACCEPT_UNIQUE" if unique_blocked else "ACCEPT_TERMINAL",
                    "uprn": int(node.uprn) if node.uprn is not None else None,
                    "at_m_index": int(cons_idx),
                    "accepted_label": acc_label,
                    "unique_blocked": bool(unique_blocked),
                    "exact_exhausted": bool(exact_exhausted),
                    "node_count": int(node.count),
                    "next_child_exists": bool(i < n and (node.has_child(t[i]) if i < n else False)),
                    "exact_hits": int(exact_hits),
                    "saw_num_exact": bool(saw_num_exact),
                    "guards": {
                        "min_exact_hits": int(min_exact_hits),
                        "require_numeric": bool(require_numeric),
                        "numeric_must_be_exact": bool(numeric_must_be_exact),
                    },
                }
                # Preserve existing transition event; attach accept info alongside
                info = parents.get(cur_key_acc)
                if info is None:
                    parents[cur_key_acc] = {"parent": None, "event": None, "accept_event": ev_acc, "node": node}
                else:
                    info["accept_event"] = ev_acc
                    info.setdefault("node", node)
            if cost < best_cost:
                runner_cost = best_cost
                best_cost = cost
                best_uprn = node.uprn
                if trace is not None:
                    best_state = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
            elif node.uprn != best_uprn and cost < runner_cost:
                runner_cost = cost
            if heap and heap[0][0] >= best_cost + 1:
                break
            continue

        if i >= n:
            continue

        tok = t[i]
        # Build state and apply rules in fixed order: exact → skip → fuzzy
        cur_state = _State(
            node=node,
            i=i,
            exact_hits=exact_hits,
            saw_num_any=saw_num_any,
            saw_num_exact=saw_num_exact,
        )
        rules: List[_RuleFunc] = [_rule_exact]
        if params_like.allow_swap_adjacent:
            rules.append(_rule_swap_adjacent)
        rules.extend([_rule_skip, _rule_fuzzy])
        if params_like.allow_canonical_insert:
            rules.append(_rule_canonical_insert)
        for rule in rules:
            for move in rule(cur_state, t, n, params_like):
                push_move(cur_state, cost, move)

    if (
        best_uprn is not None
        and best_cost <= max_cost
        and (runner_cost == float("inf") or runner_cost >= best_cost + 1)
    ):
        return (
            best_uprn,
            int(best_cost),
            (None if runner_cost == float("inf") else int(runner_cost)),
            (best_state if trace is not None else None),
            (parents if trace is not None else None),
        )
    # If no accept, still provide best partial for tracing
    if trace is not None and best_partial_key is not None:
        return (None, None, None, best_partial_key, parents)
    return (None, None, None, None, None)
# --- Step 2: Extract skip-cost helper (behavior-preserving) ---
def _calc_skip_cost(node: TrieNode, tok: str, skip_redundant_ratio: float) -> int:
    """Return 0 for redundant skip only when tok is a known child and clearly redundant.

    Otherwise return 1 (penalized).
    """
    c_anchor = int(node.count)
    c_combo = int(node.child_count(tok))
    if c_combo > 0:
        ratio = c_anchor / c_combo
        if c_anchor > c_combo and ratio >= float(skip_redundant_ratio):
            return 0
    return 1
