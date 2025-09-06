from __future__ import annotations

from typing import Callable, List, Sequence, Optional, Dict, Any, Tuple
import re

from .trie_builder import TrieNode, count_tail_L2R


def peel_end_tokens(
    tokens: Sequence[str],
    count_tail: Callable[[Sequence[str]], int],
    steps: int = 4,
    max_k: int = 2,
    debug: Optional[Callable[[str], None]] = None,
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
            if debug:
                debug(
                    f"[peel] step {step_idx + 1}: anchor='{anchor}' base={base}; consider k={k} -> '{new_anchor}' score={score}"
                )
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 0:
            if debug:
                debug(
                    f"[peel] drop last {best_k} token(s); base→best {base}→{best_score}"
                )
                debug(f"[peel] tokens now: {' '.join(out[:-best_k])}")
            out = out[:-best_k]
        else:
            break

    return out


def peel_end_tokens_with_trie(
    tokens: Sequence[str],
    root: TrieNode,
    steps: int = 4,
    max_k: int = 2,
    debug: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Thin wrapper wiring peel_end_tokens to the trie count helper."""

    def _count_tail(tail: Sequence[str]) -> int:
        return count_tail_L2R(root, tail)

    return peel_end_tokens(tokens, _count_tail, steps=steps, max_k=max_k, debug=debug)


def walk_exact(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    accept_terminal_if_exhausted: bool = True,
    debug: Optional[Callable[[str], None]] = None,
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
                if debug:
                    debug(
                        f"[exact]{'  ' * i} ACCEPT[unique_blocked] uprn={node.uprn} i={i}/{n}"
                    )
                return node.uprn
            # B) Exact-exhausted terminal
            if accept_terminal_if_exhausted and i >= n:
                if debug:
                    debug(
                        f"[exact]{'  ' * i} ACCEPT[terminal_exhausted] uprn={node.uprn} i={i}/{n}"
                    )
                return node.uprn

        if i >= n:
            return None

        nxt = t[i]
        child = node.child(nxt)
        if child is None:
            if debug:
                debug(f"[exact]{'  ' * i} BLOCK on '{nxt}' (no child)")
            return None

        if debug:
            debug(
                f"[exact]{'  ' * i} EXACT '{nxt}' -> child(count={child.count}, uprn={child.uprn})"
            )
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


def damerau_levenshtein_at_most(a: str, b: str, k: int = 1) -> bool:
    """
    Tiny Damerau–Levenshtein check for distance ≤ k (default 1).
    Optimized for k=1 with early exits; handles substitution, insertion/deletion,
    and adjacent transposition.
    """
    if a == b:
        return True
    if k <= 0:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return False
    # Equal length: allow one substitution or one adjacent transposition
    if la == lb:
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(diffs) == 1:
            return True  # one substitution
        if len(diffs) == 2:
            i, j = diffs[0], diffs[1]
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return True  # adjacent transposition
        return False
    # Length differs by 1: allow one insertion/deletion
    # Make `a` the longer
    if lb > la:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > k:
                return False
            i += 1  # skip one char in the longer string
    # If leftover chars in longer string, count as edit
    edits += la - i
    return edits <= k


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
    debug: Optional[Callable[[str], None]] = None,
) -> Optional[int]:
    """
    Stage‑1 (Step‑5): Exact + Skip search with small cost budget.

    Transitions:
      - Exact consume: cost +0, i -> i-1, descend to child
      - Skip messy token: cost +(0 or 1) depending on counts at anchor

    Acceptance: same as Step‑3 (unique & blocked OR exact‑exhausted terminal),
    AND guards: at least `min_exact_hits` exact tokens, and (if enabled) saw a
    numeric token on the accepted path.
    """
    t = list(reversed([str(x) for x in tokens_L2R]))
    n = len(t)

    import heapq

    if debug:
        debug(
            "[search] tokens L2R: "
            + " ".join(str(x) for x in tokens_L2R)
            + " | R→L: "
            + " ".join(t)
        )

    def rem_l2r(i: int) -> str:
        # Remaining tokens in L2R before consuming the next R→L token t[i]
        return " ".join(tokens_L2R[: -(i) or None])

    def rem_l2r_after(i: int) -> str:
        # Remaining tokens after consuming/skip the next token
        return " ".join(tokens_L2R[: -(i + 1) or None])

    def consumed_l2r_after(i: int) -> str:
        return " ".join(tokens_L2R[-(i + 1) :])

    def accept(
        node: TrieNode, i: int, exact_hits: int, saw_num: bool
    ) -> Optional[tuple[str, str]]:
        if node.uprn is None:
            return None
        unique_blocked = node.count == 1 and (i >= n or not node.has_child(t[i]))
        exact_exhausted = accept_terminal_if_exhausted and i >= n
        if not (unique_blocked or exact_exhausted):
            return None
        if exact_hits < min_exact_hits:
            return None
        if require_numeric and not saw_num:
            return None
        if unique_blocked:
            reason = (
                "no next token" if i >= n else f"next token '{t[i]}' cannot descend"
            )
            return ("unique_blocked", reason)
        else:
            return ("terminal_exhausted", "all tokens consumed at terminal node")

    def skip_cost(node: TrieNode, tok: str) -> int:
        c_anchor = int(node.count)
        c_combo = int(node.child_count(tok))
        ratio = c_anchor / max(1, c_combo)
        # 0-cost skip if the token appears to be redundant at this anchor
        if c_anchor > c_combo and ratio >= float(skip_redundant_ratio):
            return 0
        return 1

    # (cost, seq, node, i, exact_hits, saw_numeric_any, saw_numeric_exact)
    heap: list[tuple[int, int, TrieNode, int, int, bool, bool]] = []
    seq = 0
    heapq.heappush(heap, (0, seq, root, 0, 0, False, False))
    best_cost = float("inf")
    best_uprn: Optional[int] = None
    runner_cost = float("inf")

    # visited pruning: keep best cost per (node_id, i, exact_hits, saw_num_any, saw_num_exact)
    seen: dict[tuple[int, int, int, bool, bool], int] = {}

    while heap:
        cost, _, node, i, exact_hits, saw_num_any, saw_num_exact = heapq.heappop(heap)

        if cost > max_cost:
            break

        key = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
        prev = seen.get(key)
        if prev is not None and prev <= cost:
            continue
        seen[key] = cost

        # Acceptance check at current node
        acc = accept(
            node,
            i,
            exact_hits,
            saw_num_exact if numeric_must_be_exact else saw_num_any,
        )
        if acc:
            acc_type, reason = acc
            if debug:
                debug(
                    f"[search]{'  ' * i} ACCEPT[{acc_type}] uprn={node.uprn} cost={cost} hits={exact_hits} numeric_exact={saw_num_exact} numeric_any={saw_num_any} progress={i}/{n} | {reason}"
                )
            if cost < best_cost:
                runner_cost = best_cost
                best_cost = cost
                best_uprn = node.uprn
            elif node.uprn != best_uprn and cost < runner_cost:
                runner_cost = cost

            # Early stop if next state cost exceeds current best by at least 1
            if heap and heap[0][0] >= best_cost + 1:
                break
            continue

        if i >= n:
            continue

        tok = t[i]

        # Exact consume
        child = node.child(tok)
        if child is not None:
            if debug:
                indent = "  " * i
                debug(f"[search]{indent} EXACT consume '{tok}': child exists → descend")
                debug(
                    f"[search]{indent}   child(count={child.count}, uprn={child.uprn}); cost stays {cost}; progress {i}/{n}→{i + 1}/{n}"
                )
                debug(
                    f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'; consumed now: '{consumed_l2r_after(i)}'"
                )
            seq += 1
            heapq.heappush(
                heap,
                (
                    cost,
                    seq,
                    child,
                    i + 1,
                    exact_hits + 1,
                    saw_num_any or is_numeric(tok),
                    saw_num_exact or is_numeric(tok),
                ),
            )

        # Skip messy
        s_cost = skip_cost(node, tok)
        if debug:
            indent = "  " * i
            c_anchor = int(node.count)
            c_combo = int(node.child_count(tok))
            ratio = c_anchor / max(1, c_combo)
            reason = (
                f"redundant (ratio={ratio:.2f} ≥ {skip_redundant_ratio})"
                if c_anchor > c_combo and ratio >= float(skip_redundant_ratio)
                else f"kept (ratio={ratio:.2f} < {skip_redundant_ratio})"
            )
            debug(
                f"[search]{indent} SKIP '{tok}': cost +{s_cost} — {reason}; progress {i}/{n}→{i + 1}/{n}"
            )
            debug(
                f"[search]{indent}   counts: anchor={c_anchor}, child_count={c_combo}"
            )
            debug(
                f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'"
            )
        seq += 1
        heapq.heappush(
            heap,
            (cost + s_cost, seq, node, i + 1, exact_hits, saw_num_any, saw_num_exact),
        )

        # Fuzzy consume (only if no exact child). Try children within DL<=1
        if child is None:
            for lbl, ch in node.iter_children():
                etype = dl1_edit_type(tok, lbl)
                if etype is not None and etype != "exact":
                    if debug:
                        indent = "  " * i
                        debug(
                            f"[search]{indent} FUZZY '{tok}' ≈ '{lbl}': {etype}; cost +1; child(count={ch.count}, uprn={ch.uprn}); progress {i}/{n}→{i + 1}/{n}"
                        )
                        debug(
                            f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'"
                        )
                    seq += 1
                    heapq.heappush(
                        heap,
                        (
                            cost + 1,
                            seq,
                            ch,
                            i + 1,
                            exact_hits,  # fuzzy doesn't count toward exact hits
                            saw_num_any or is_numeric(lbl),  # any numeric
                            saw_num_exact,  # do NOT set on fuzzy
                        ),
                    )

    if (
        best_uprn is not None
        and best_cost <= max_cost
        and (runner_cost == float("inf") or runner_cost >= best_cost + 1)
    ):
        return best_uprn
    return None


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


def match_stage1(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    params: Params = Params(),
    *,
    debug: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Stage‑1 matcher: peel → exact/skip/fuzzy search with strict guards.

    Returns a structured result dict with `matched`, `uprn`, `cost`, and the
    peeled tokens used for matching.
    """
    peeled = peel_end_tokens_with_trie(tokens_L2R, root, steps=4, max_k=2, debug=debug)

    # Reuse existing search; adapt to return cost by probing heap order would
    # require refactor. For now, run the same logic and infer cost via a tiny
    # local duplicate of acceptance logic. To keep changes surgical, we call the
    # existing function and set cost=None; we can upgrade to return cost in Step 10.
    uprn, best_cost, _ = _search_with_skips(
        peeled,
        root,
        max_cost=params.max_cost,
        min_exact_hits=params.min_exact_hits,
        require_numeric=params.require_numeric,
        numeric_must_be_exact=params.numeric_must_be_exact,
        skip_redundant_ratio=params.skip_redundant_ratio,
        accept_terminal_if_exhausted=params.accept_terminal_if_exhausted,
        debug=debug,
    )

    return {
        "matched": uprn is not None,
        "uprn": uprn,
        "cost": best_cost,
        "peeled_tokens": peeled,
        "input_tokens": list(tokens_L2R),
    }


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
    debug: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Internal search that returns (uprn, best_cost, runner_cost)."""
    t = list(reversed([str(x) for x in tokens_L2R]))
    n = len(t)

    import heapq

    if debug:
        debug(
            "[search] tokens L2R: "
            + " ".join(str(x) for x in tokens_L2R)
            + " | R→L: "
            + " ".join(t)
        )

    def rem_l2r(i: int) -> str:
        return " ".join(tokens_L2R[: -(i) or None])

    def rem_l2r_after(i: int) -> str:
        return " ".join(tokens_L2R[: -(i + 1) or None])

    def consumed_l2r_after(i: int) -> str:
        return " ".join(tokens_L2R[-(i + 1) :])

    def accept(
        node: TrieNode, i: int, exact_hits: int, saw_num: bool
    ) -> Optional[tuple[str, str]]:
        if node.uprn is None:
            return None
        unique_blocked = node.count == 1 and (i >= n or not node.has_child(t[i]))
        exact_exhausted = accept_terminal_if_exhausted and i >= n
        if not (unique_blocked or exact_exhausted):
            return None
        if exact_hits < min_exact_hits:
            return None
        if require_numeric and not saw_num:
            return None
        if unique_blocked:
            reason = (
                "no next token" if i >= n else f"next token '{t[i]}' cannot descend"
            )
            return ("unique_blocked", reason)
        else:
            return ("terminal_exhausted", "all tokens consumed at terminal node")

    def skip_cost(node: TrieNode, tok: str) -> int:
        c_anchor = int(node.count)
        c_combo = int(node.child_count(tok))
        ratio = c_anchor / max(1, c_combo)
        if c_anchor > c_combo and ratio >= float(skip_redundant_ratio):
            return 0
        return 1

    heap: list[tuple[int, int, TrieNode, int, int, bool, bool]] = []
    seq = 0
    heapq.heappush(heap, (0, seq, root, 0, 0, False, False))
    best_cost = float("inf")
    best_uprn: Optional[int] = None
    runner_cost = float("inf")
    seen: dict[tuple[int, int, int, bool, bool], int] = {}

    while heap:
        cost, _, node, i, exact_hits, saw_num_any, saw_num_exact = heapq.heappop(heap)
        if cost > max_cost:
            break
        key = (id(node), i, exact_hits, saw_num_any, saw_num_exact)
        prev = seen.get(key)
        if prev is not None and prev <= cost:
            continue
        seen[key] = cost

        acc = accept(
            node,
            i,
            exact_hits,
            saw_num_exact if numeric_must_be_exact else saw_num_any,
        )
        if acc:
            acc_type, reason = acc
            if debug:
                debug(
                    f"[search]{'  ' * i} ACCEPT[{acc_type}] uprn={node.uprn} cost={cost} hits={exact_hits} numeric_exact={saw_num_exact} numeric_any={saw_num_any} progress={i}/{n} | {reason}"
                )
            if cost < best_cost:
                runner_cost = best_cost
                best_cost = cost
                best_uprn = node.uprn
            elif node.uprn != best_uprn and cost < runner_cost:
                runner_cost = cost
            if heap and heap[0][0] >= best_cost + 1:
                break
            continue

        if i >= n:
            continue

        tok = t[i]
        child = node.child(tok)
        if child is not None:
            if debug:
                indent = "  " * i
                debug(f"[search]{indent} EXACT consume '{tok}': child exists → descend")
                debug(
                    f"[search]{indent}   child(count={child.count}, uprn={child.uprn}); cost stays {cost}; progress {i}/{n}→{i + 1}/{n}"
                )
                debug(
                    f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'; consumed now: '{consumed_l2r_after(i)}'"
                )
            seq += 1
            heapq.heappush(
                heap,
                (
                    cost,
                    seq,
                    child,
                    i + 1,
                    exact_hits + 1,
                    saw_num_any or is_numeric(tok),
                    saw_num_exact or is_numeric(tok),
                ),
            )

        s_cost = skip_cost(node, tok)
        if debug:
            indent = "  " * i
            c_anchor = int(node.count)
            c_combo = int(node.child_count(tok))
            ratio = c_anchor / max(1, c_combo)
            reason = (
                f"redundant (ratio={ratio:.2f} ≥ {skip_redundant_ratio})"
                if c_anchor > c_combo and ratio >= float(skip_redundant_ratio)
                else f"kept (ratio={ratio:.2f} < {skip_redundant_ratio})"
            )
            debug(
                f"[search]{indent} SKIP '{tok}': cost +{s_cost} — {reason}; progress {i}/{n}→{i + 1}/{n}"
            )
            debug(
                f"[search]{indent}   counts: anchor={c_anchor}, child_count={c_combo}"
            )
            debug(
                f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'"
            )
        seq += 1
        heapq.heappush(
            heap,
            (cost + s_cost, seq, node, i + 1, exact_hits, saw_num_any, saw_num_exact),
        )

        if child is None:
            for lbl, ch in node.iter_children():
                etype = dl1_edit_type(tok, lbl)
                if etype is not None and etype != "exact":
                    if debug:
                        indent = "  " * i
                        debug(
                            f"[search]{indent} FUZZY '{tok}' ≈ '{lbl}': {etype}; cost +1; child(count={ch.count}, uprn={ch.uprn}); progress {i}/{n}→{i + 1}/{n}"
                        )
                        debug(
                            f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'"
                        )
                    seq += 1
                    heapq.heappush(
                        heap,
                        (
                            cost + 1,
                            seq,
                            ch,
                            i + 1,
                            exact_hits,
                            saw_num_any or is_numeric(lbl),
                            saw_num_exact,
                        ),
                    )

    if (
        best_uprn is not None
        and best_cost <= max_cost
        and (runner_cost == float("inf") or runner_cost >= best_cost + 1)
    ):
        return (
            best_uprn,
            int(best_cost),
            (None if runner_cost == float("inf") else int(runner_cost)),
        )
    return None, None, None
