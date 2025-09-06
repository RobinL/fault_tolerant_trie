from __future__ import annotations

from typing import Callable, List, Sequence, Optional
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
                    f"[peel] step {step_idx+1}: anchor='{anchor}' base={base}; consider k={k} -> '{new_anchor}' score={score}"
                )
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 0:
            if debug:
                debug(f"[peel] drop last {best_k} token(s); base→best {base}→{best_score}")
                debug(
                    f"[peel] tokens now: {' '.join(out[: -best_k])}"
                )
            out = out[: -best_k]
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
                        f"[exact]{'  '*i} ACCEPT[unique_blocked] uprn={node.uprn} i={i}/{n}"
                    )
                return node.uprn
            # B) Exact-exhausted terminal
            if accept_terminal_if_exhausted and i >= n:
                if debug:
                    debug(
                        f"[exact]{'  '*i} ACCEPT[terminal_exhausted] uprn={node.uprn} i={i}/{n}"
                    )
                return node.uprn

        if i >= n:
            return None

        nxt = t[i]
        child = node.child(nxt)
        if child is None:
            if debug:
                debug(f"[exact]{'  '*i} BLOCK on '{nxt}' (no child)")
            return None

        if debug:
            debug(
                f"[exact]{'  '*i} EXACT '{nxt}' -> child(count={child.count}, uprn={child.uprn})"
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


def match_stage1_with_skips(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    max_cost: int = 2,
    min_exact_hits: int = 2,
    require_numeric: bool = True,
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

    def accept(node: TrieNode, i: int, exact_hits: int, saw_num: bool) -> Optional[tuple[str, str]]:
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

    # (cost, seq, node, i, exact_hits, saw_numeric)
    heap: list[tuple[int, int, TrieNode, int, int, bool]] = []
    seq = 0
    heapq.heappush(heap, (0, seq, root, 0, 0, False))
    best_cost = float("inf")
    best_uprn: Optional[int] = None
    runner_cost = float("inf")

    # visited pruning: keep best cost per (node_id, i, exact_hits, saw_num)
    seen: dict[tuple[int, int, int, bool], int] = {}

    while heap:
        cost, _, node, i, exact_hits, saw_num = heapq.heappop(heap)

        if cost > max_cost:
            break

        key = (id(node), i, exact_hits, saw_num)
        prev = seen.get(key)
        if prev is not None and prev <= cost:
            continue
        seen[key] = cost

        # Acceptance check at current node
        acc = accept(node, i, exact_hits, saw_num)
        if acc:
            acc_type, reason = acc
            if debug:
                debug(
                    f"[search]{'  '*i} ACCEPT[{acc_type}] uprn={node.uprn} cost={cost} hits={exact_hits} numeric={saw_num} progress={i}/{n} | {reason}"
                )
            if cost < best_cost:
                runner_cost = best_cost
                best_cost = cost
                best_uprn = node.uprn
            elif cost < runner_cost:
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
                indent = '  ' * i
                debug(
                    f"[search]{indent} EXACT consume '{tok}': child exists → descend"
                )
                debug(
                    f"[search]{indent}   child(count={child.count}, uprn={child.uprn}); cost stays {cost}; progress {i}/{n}→{i+1}/{n}"
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
                    saw_num or is_numeric(tok),
                ),
            )

        # Skip messy
        s_cost = skip_cost(node, tok)
        if debug:
            indent = '  ' * i
            c_anchor = int(node.count)
            c_combo = int(node.child_count(tok))
            ratio = c_anchor / max(1, c_combo)
            reason = (
                f"redundant (ratio={ratio:.2f} ≥ {skip_redundant_ratio})"
                if c_anchor > c_combo and ratio >= float(skip_redundant_ratio)
                else f"kept (ratio={ratio:.2f} < {skip_redundant_ratio})"
            )
            debug(
                f"[search]{indent} SKIP '{tok}': cost +{s_cost} — {reason}; progress {i}/{n}→{i+1}/{n}"
            )
            debug(
                f"[search]{indent}   counts: anchor={c_anchor}, child_count={c_combo}"
            )
            debug(
                f"[search]{indent}   remaining L2R: '{rem_l2r(i)}' → '{rem_l2r_after(i)}'"
            )
        seq += 1
        heapq.heappush(heap, (cost + s_cost, seq, node, i + 1, exact_hits, saw_num))

    if best_uprn is not None and best_cost <= max_cost and (
        runner_cost == float("inf") or runner_cost >= best_cost + 1
    ):
        return best_uprn
    return None
