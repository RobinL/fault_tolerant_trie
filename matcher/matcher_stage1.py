from __future__ import annotations

from typing import Callable, List, Sequence, Optional
import re

from .trie_builder import TrieNode, count_tail_L2R


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
    for _ in range(max(0, int(steps))):
        if len(out) <= 1:
            break

        base = count_tail([out[-1]])
        best_k = 0
        best_score = base

        max_try = min(int(max_k), len(out) - 1)
        for k in range(1, max_try + 1):
            new_last = out[-k - 1]
            score = count_tail([new_last])
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 0:
            out = out[: -best_k]
        else:
            break

    return out


def peel_end_tokens_with_trie(
    tokens: Sequence[str],
    root: TrieNode,
    steps: int = 4,
    max_k: int = 2,
) -> List[str]:
    """Thin wrapper wiring peel_end_tokens to the trie count helper."""

    def _count_tail(tail: Sequence[str]) -> int:
        return count_tail_L2R(root, tail)

    return peel_end_tokens(tokens, _count_tail, steps=steps, max_k=max_k)


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


def match_stage1_with_skips(
    tokens_L2R: Sequence[str],
    root: TrieNode,
    *,
    max_cost: int = 2,
    min_exact_hits: int = 2,
    require_numeric: bool = True,
    skip_redundant_ratio: float = 2.0,
    accept_terminal_if_exhausted: bool = True,
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
        if accept(node, i, exact_hits, saw_num):
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
        seq += 1
        heapq.heappush(heap, (cost + s_cost, seq, node, i + 1, exact_hits, saw_num))

    if best_uprn is not None and best_cost <= max_cost and (
        runner_cost == float("inf") or runner_cost >= best_cost + 1
    ):
        return best_uprn
    return None
